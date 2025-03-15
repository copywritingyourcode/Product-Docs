"""
Medical Specialist Agent for the AI Assistant Swarm.

This agent specializes in medical and health-related queries,
using the Gemma3 27B model to provide accurate, beginner-friendly
explanations for medical questions.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

from .base import BaseAgent
from .model_client import OllamaClient

# Configure logging
logger = logging.getLogger("agent.medical")

class MedicalAgent(BaseAgent):
    """
    Medical Specialist Agent for health-related inquiries.
    
    This agent leverages the Gemma3 27B model to provide:
    - Detailed yet beginner-friendly medical explanations
    - Analysis of medical documents
    - Health-related guidance
    
    Note: This agent is not a replacement for professional medical advice
    and should be used for informational purposes only.
    """
    
    DEFAULT_SYSTEM_PROMPT = """
    You are a knowledgeable Medical Specialist with expertise in explaining medical concepts in an accessible way.
    Your role is to provide clear, accurate information about medical topics, while being mindful of your limitations.
    
    When responding to medical questions:
    - Explain medical terms in simple language for beginners
    - Provide factual, evidence-based information
    - Be clear about the general nature of your advice
    - Avoid making definitive diagnoses or prescribing treatments
    - Include appropriate disclaimers when necessary
    
    When discussing medical research or documents:
    - Summarize key findings accurately
    - Explain the implications in plain language
    - Note limitations or uncertainties in the research
    - Provide context for medical statistics
    
    Always clarify that your information is educational and does not replace professional medical advice.
    Be helpful but responsible in providing health information.
    """
    
    def __init__(
        self,
        model_name: str = "gemma3:27b",
        memory_manager = None,
        max_context_length: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize the Medical Specialist agent.
        
        Args:
            model_name: The Ollama model to use (default: gemma3:27b)
            memory_manager: Optional memory manager for long-term memory
            max_context_length: Maximum context length for the model
            temperature: Temperature for model inference
        """
        super().__init__(
            name="Medical Specialist",
            model_name=model_name,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            memory_manager=memory_manager,
            max_context_length=max_context_length,
            temperature=temperature,
        )
        self.ollama_client = OllamaClient()
        
    async def generate_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Generate a medical response to the user's query.
        
        Args:
            query: The user's query about medical topics
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            The agent's response with medical information
        """
        # Check if memory retrieval is needed and available
        relevant_context = ""
        if retrieve_from_memory and self.memory_manager:
            try:
                # Medical information requires extra care - specify document type
                documents = await self.memory_manager.retrieve(
                    query, 
                    limit=5, 
                    doc_types=["medical_document", "research_paper"]
                )
                if documents:
                    relevant_context = "Relevant information from trusted medical sources:\n\n"
                    for i, doc in enumerate(documents, 1):
                        # Include source metadata if available for verification
                        source = doc.metadata.get("source", "Unknown source")
                        date = doc.metadata.get("date", "Unknown date")
                        relevant_context += f"{i}. From {source} ({date}):\n{doc.page_content}\n\n"
                    logger.info(f"Retrieved {len(documents)} medical documents from memory")
            except Exception as e:
                logger.error(f"Error retrieving from memory: {str(e)}")
                
        # Prepare the enhanced prompt with any retrieved context
        enhanced_query = query
        if relevant_context:
            enhanced_query = f"{relevant_context}\n\nUser medical query: {query}\n\nPlease incorporate the above relevant medical information if appropriate, and remember to provide beginner-friendly explanations."
            
        # Add the query to context (without the enhanced part)
        self.add_to_context("user", query)
        
        try:
            # Generate response using Gemma3 model
            prompt = self.format_prompt(enhanced_query)
            response = await self.ollama_client.generate(
                model_name=self.model_name,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
            )
            
            # Add medical disclaimer if not already included
            if "not a substitute for professional medical advice" not in response.lower():
                response += "\n\nRemember: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment."
            
            # Add the response to the context
            self.add_to_context("assistant", response)
            
            # Trim the context if needed to avoid growing too large
            self._truncate_context_if_needed()
            
            # If memory manager exists, store this interaction with medical tag
            if self.memory_manager:
                try:
                    await self.memory_manager.add_interaction(query, response, "medical_inquiry")
                    logger.info("Stored medical interaction in memory")
                except Exception as e:
                    logger.error(f"Error storing in memory: {str(e)}")
                    
            return response
        except Exception as e:
            logger.error(f"Error generating medical response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response. Please check if the Gemma3 27B model is properly installed in Ollama, or try again with a different query. Error details: {str(e)}" 