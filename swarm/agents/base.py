"""
Base agent class for the AI Assistant Swarm.

This module defines the base agent class that all specialized agents inherit from.
"""

import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from swarm.agents.model_client import OllamaClient


@dataclass
class Message:
    """Class representing a message in the agent context."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


class BaseAgent:
    """
    Base agent class for all specialized agents in the AI Assistant Swarm.
    
    This abstract class provides common functionality for all agents,
    including context management and interface with the language model.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.7,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            model_name: Name of the model to use
            system_prompt: System prompt defining the agent's role
            temperature: Temperature parameter for generation (0.0 to 1.0)
        """
        self.name = name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.context: List[Message] = []
        self.client = OllamaClient()
        self.logger = logging.getLogger(__name__)
        
        # Initialize context with system prompt
        self.reset_context()
    
    def reset_context(self):
        """Reset the agent's conversation context."""
        self.context = [Message(role="system", content=self.system_prompt)]
    
    def add_to_context(self, role: str, content: str):
        """
        Add a message to the agent's context.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.context.append(Message(role=role, content=content))
    
    def format_prompt(self, query: str) -> str:
        """
        Format the context and query into a prompt for the model.
        
        Args:
            query: User query to append to context
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add all context messages
        for message in self.context:
            prompt_parts.append(f"{message.role}: {message.content}")
        
        # Add the current query
        prompt_parts.append(f"user: {query}")
        
        return "\n\n".join(prompt_parts)
    
    async def generate_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Generate a response to the user query.
        
        Args:
            query: User query
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            Generated response
        
        Note:
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_response")
    
    async def _call_model(self, prompt: str) -> str:
        """
        Call the language model with the given prompt.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Model response
        """
        try:
            start_time = time.time()
            response = await self.client.generate(
                model_name=self.model_name,
                prompt=prompt,
                temperature=self.temperature
            )
            elapsed = time.time() - start_time
            
            self.logger.debug(f"Generated response in {elapsed:.2f}s using {self.model_name}")
            return response
        except Exception as e:
            self.logger.error(f"Error calling model {self.model_name}: {e}")
            return f"Error generating response: {str(e)}" 