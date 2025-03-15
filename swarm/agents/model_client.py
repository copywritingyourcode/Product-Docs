"""
Ollama Model Client for AI Assistant Swarm.

This module provides a client for interacting with Ollama models,
handling model loading, inference, and error handling.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger("model_client")

class OllamaClient:
    """
    Client for interacting with Ollama models.
    
    Handles communication with the Ollama API for model inference,
    with support for streaming, error handling, and retries.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        request_timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: The base URL of the Ollama API
            request_timeout: Timeout for requests in seconds
            max_retries: Maximum number of retries on failure
        """
        self.base_url = base_url
        self.timeout = request_timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized Ollama client with base URL: {base_url}")
        
    async def ensure_session(self) -> None:
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in Ollama.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            True if the model exists, False otherwise
        """
        await self.ensure_session()
        try:
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to get model list: {response.status}")
                    return False
                
                data = await response.json()
                models = data.get("models", [])
                
                for model in models:
                    if model.get("name") == model_name:
                        logger.info(f"Model {model_name} exists")
                        return True
                        
                logger.warning(f"Model {model_name} not found in Ollama")
                return False
        except Exception as e:
            logger.error(f"Error checking model existence: {str(e)}")
            raise
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> AsyncGenerator[str, None] if stream else str:
        """
        Generate text using an Ollama model.
        
        Args:
            model_name: The name of the Ollama model to use
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for the model
            temperature: Temperature for model inference
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            If stream=True, an async generator yielding response chunks
            If stream=False, the full response as a string
        """
        await self.ensure_session()
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            start_time = time.time()
            url = f"{self.base_url}/api/generate"
            
            if stream:
                return self._stream_response(url, payload)
            else:
                async with self.session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    logger.info(f"Generated response in {time.time() - start_time:.2f}s")
                    return data.get("response", "")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def _stream_response(self, url: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream response from Ollama API.
        
        Args:
            url: The API URL to call
            payload: The request payload
            
        Yields:
            Response chunks as they arrive
        """
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            logger.error(f"Ollama streaming error: {data['error']}")
                            raise RuntimeError(f"Ollama streaming error: {data['error']}")
                            
                        if "response" in data:
                            yield data["response"]
                            
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from stream: {line}")
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise 