"""
Fallback Agent for the AI Assistant Swarm.

This agent handles queries that don't fit specialized domains,
providing generalist knowledge and assistance for miscellaneous topics.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

from .base import BaseAgent
from .model_client import OllamaClient

# Configure logging
logger = logging.getLogger("agent.fallback")

class FallbackAgent(BaseAgent):
    """
    Fallback Agent for general-purpose queries.
    
    This agent handles queries that don't fit into the specialized domains
    of other agents, providing a safety net for the assistant swarm.
    """
    
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, knowledgeable assistant capable of answering a wide range of general questions.
    While you don't specialize in any particular field, you can provide useful information on many topics.
    
    Your strengths include:
    - Providing general knowledge on various subjects
    - Giving practical advice for everyday questions
    - Explaining simple concepts in an accessible way
    - Helping with basic tasks and questions
    
    When you're unsure about specialized topics:
    - Acknowledge your limitations
    - Provide general information that might be helpful
    - Suggest that the user might want to consult a specialist for medical or coding questions
    
    Be conversational, helpful, and friendly while remaining informative.
    """
    
    def __init__(
        self,
        model_name: str = "qwen",  # Could be the same as orchestrator or a smaller model
        memory_manager = None,
        max_context_length: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize the Fallback agent.
        
        Args:
            model_name: The Ollama model to use (default: qwen)
            memory_manager: Optional memory manager for long-term memory
            max_context_length: Maximum context length for the model
            temperature: Temperature for model inference
        """
        super().__init__(
            name="General Assistant",
            model_name=model_name,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            memory_manager=memory_manager,
            max_context_length=max_context_length,
            temperature=temperature,
        )
        self.ollama_client = OllamaClient()
        
    async def generate_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Generate a general response to the user's query.
        
        Args:
            query: The user's general query
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            The agent's response with general information
        """
        # Check if memory retrieval is needed and available
        relevant_context = ""
        if retrieve_from_memory and self.memory_manager:
            try:
                # For fallback, we can try any document type
                documents = await self.memory_manager.retrieve(query, limit=3)
                if documents:
                    relevant_context = "I found some potentially relevant information:\n\n"
                    for i, doc in enumerate(documents, 1):
                        relevant_context += f"{i}. {doc.page_content}\n\n"
                    logger.info(f"Retrieved {len(documents)} documents from memory")
            except Exception as e:
                logger.error(f"Error retrieving from memory: {str(e)}")
                
        # Prepare the enhanced prompt with any retrieved context
        enhanced_query = query
        if relevant_context:
            enhanced_query = f"{relevant_context}\n\nUser query: {query}\n\nPlease use the above information if it's relevant to the question."
            
        # Add the query to context (without the enhanced part)
        self.add_to_context("user", query)
        
        try:
            # Generate response using model
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
                    await self.memory_manager.add_interaction(query, response, "general_query")
                    logger.info("Stored general interaction in memory")
                except Exception as e:
                    logger.error(f"Error storing in memory: {str(e)}")
                    
            return response
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response. Please check if the required model is properly installed in Ollama, or try again with a different query. Error details: {str(e)}" 