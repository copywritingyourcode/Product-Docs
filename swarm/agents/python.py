"""
Python Developer Agent for the AI Assistant Swarm.

This agent specializes in Python programming assistance,
using the DeepSeek-RAG model to provide coding help,
debugging assistance, and explanations suitable for beginners.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

from .base import BaseAgent
from .model_client import OllamaClient

# Configure logging
logger = logging.getLogger("agent.python")

class PythonAgent(BaseAgent):
    """
    Python Developer Agent specializing in programming assistance.
    
    This agent leverages the DeepSeek-RAG model to provide:
    - Python code examples and guidance
    - Debugging help
    - Explanations tailored for beginners
    """
    
    DEFAULT_SYSTEM_PROMPT = """
    You are an expert Python developer with years of experience, acting as a coding mentor.
    Your specialty is helping beginner Python users understand concepts, debug issues, and write clean code.
    
    When providing code examples:
    - Always use Python 3 syntax
    - Include detailed comments explaining what each part does
    - Start with simple solutions before introducing more complex ones
    - Verify that the code will actually run without errors
    - Explain how to run or test the code for beginners
    
    When explaining concepts:
    - Use simple language appropriate for beginners
    - Relate to real-world examples where possible
    - Build gradually from fundamentals to more complex ideas
    - Avoid jargon without explanation
    
    Remember that you're teaching as much as solving problems.
    Your goal is to ensure the user not only gets working code, but understands it thoroughly.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-rag",
        memory_manager = None,
        max_context_length: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize the Python Developer agent.
        
        Args:
            model_name: The Ollama model to use (default: deepseek-rag)
            memory_manager: Optional memory manager for long-term memory
            max_context_length: Maximum context length for the model
            temperature: Temperature for model inference
        """
        super().__init__(
            name="Senior Python Developer",
            model_name=model_name,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            memory_manager=memory_manager,
            max_context_length=max_context_length,
            temperature=temperature,
        )
        self.ollama_client = OllamaClient()
        
    async def generate_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Generate a Python-related response to the user's query.
        
        Args:
            query: The user's query about Python programming
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            The agent's response with Python guidance or code
        """
        # Check if memory retrieval is needed and available
        relevant_context = ""
        if retrieve_from_memory and self.memory_manager:
            try:
                documents = await self.memory_manager.retrieve(query, limit=3)
                if documents:
                    relevant_context = "Relevant information from my knowledge base:\n\n"
                    for i, doc in enumerate(documents, 1):
                        relevant_context += f"{i}. {doc.page_content}\n\n"
                    logger.info(f"Retrieved {len(documents)} documents from memory")
            except Exception as e:
                logger.error(f"Error retrieving from memory: {str(e)}")
                
        # Prepare the enhanced prompt with any retrieved context
        enhanced_query = query
        if relevant_context:
            enhanced_query = f"{relevant_context}\n\nUser query: {query}\n\nPlease use the above relevant information if it helps with answering the question."
            
        # Add the query to context (without the enhanced part)
        self.add_to_context("user", query)
        
        try:
            # Generate response using DeepSeek-RAG model
            prompt = self.format_prompt(enhanced_query)
            response = await self.ollama_client.generate(
                model_name=self.model_name,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
            )
            
            # Add the response to the context
            self.add_to_context("assistant", response)
            
            # Trim the context if needed to avoid growing too large
            self._truncate_context_if_needed()
            
            # If memory manager exists, store this interaction
            if self.memory_manager:
                try:
                    await self.memory_manager.add_interaction(query, response, "python_code")
                    logger.info("Stored interaction in memory")
                except Exception as e:
                    logger.error(f"Error storing in memory: {str(e)}")
                    
            return response
        except Exception as e:
            logger.error(f"Error generating Python response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response. Please check if the DeepSeek-RAG model is properly installed in Ollama, or try again with a different query. Error details: {str(e)}" 