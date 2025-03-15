"""
Swarm Manager for AI Assistant Swarm.

This module provides the central manager for the entire system,
coordinating agents, memory, and handling user queries.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from pathlib import Path

from .agents.base import BaseAgent
from .agents.medical import MedicalAgent
from .agents.python import PythonAgent
from .agents.orchestrator import OrchestratorAgent
from .agents.fallback import FallbackAgent
from .agents.model_client import OllamaClient
from .memory.vector_store import VectorStore, DocumentType
from .memory.retention import RetentionManager
from .memory.document import Document

# Configure logging
logger = logging.getLogger("swarm_manager")

class SwarmManager:
    """
    Manager for the AI Assistant Swarm system.
    
    This class coordinates the different agents, handles memory management,
    and provides the main API for interacting with the swarm.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        temperature: float = 0.7,
        models_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Swarm Manager.
        
        Args:
            data_dir: Directory for storing data (vector DB, etc.)
            temperature: Temperature for model inference
            models_config: Optional custom model names configuration
        """
        self.temperature = temperature
        
        # Set up data directory
        if data_dir is None:
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
        else:
            self.data_dir = Path(data_dir)
            
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up vector storage
        vector_db_path = self.data_dir / "vector_db"
        self.vector_store = VectorStore(
            persist_directory=str(vector_db_path)
        )
        
        # Set up retention manager
        self.retention_manager = RetentionManager(
            vector_store=self.vector_store
        )
        
        # Configure models
        self.models_config = models_config or {
            "medical": "gemma3:27b",
            "python": "deepseek-rag",
            "orchestrator": "qwen",
            "fallback": "qwen"  # Can use same model as orchestrator
        }
        
        # Initialize model client
        self.model_client = OllamaClient()
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Start background tasks
        self.background_tasks = []
        self._start_background_tasks()
        
        logger.info("Swarm Manager initialized")
    
    def _initialize_agents(self) -> None:
        """Initialize all the specialized agents."""
        try:
            # Create the agent instances
            self.agents["medical"] = MedicalAgent(
                model_name=self.models_config["medical"],
                memory_manager=self,
                temperature=self.temperature
            )
            
            self.agents["python"] = PythonAgent(
                model_name=self.models_config["python"],
                memory_manager=self,
                temperature=self.temperature
            )
            
            self.agents["fallback"] = FallbackAgent(
                model_name=self.models_config["fallback"],
                memory_manager=self,
                temperature=self.temperature
            )
            
            # Create the orchestrator last so it can coordinate the others
            self.orchestrator = OrchestratorAgent(
                model_name=self.models_config["orchestrator"],
                memory_manager=self,
                temperature=self.temperature
            )
            
            # Register agents with the orchestrator
            self.orchestrator.register_agent("medical", self.agents["medical"])
            self.orchestrator.register_agent("python", self.agents["python"])
            self.orchestrator.register_agent("fallback", self.agents["fallback"])
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _start_background_tasks(self) -> None:
        """Start background tasks like retention management."""
        # Start retention cleanup scheduler
        retention_task = asyncio.create_task(self.retention_manager.start_cleanup_scheduler())
        self.background_tasks.append(retention_task)
        
        logger.info("Background tasks started")
    
    async def check_models(self) -> Dict[str, bool]:
        """
        Check if required models are available in Ollama.
        
        Returns:
            Dictionary of model names to availability status
        """
        results = {}
        
        # Get unique model names
        model_names = set(self.models_config.values())
        
        # Check each model
        for model in model_names:
            try:
                available = await self.model_client.check_model_exists(model)
                results[model] = available
                logger.info(f"Model {model} available: {available}")
            except Exception as e:
                logger.error(f"Error checking model {model}: {str(e)}")
                results[model] = False
                
        return results
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for all agents.
        
        Args:
            temperature: The temperature value (0.0 to 1.0)
        """
        self.temperature = max(0.1, min(1.0, temperature))  # Clamp to valid range
        
        # Update temperature for all agents
        for agent in self.agents.values():
            agent.temperature = self.temperature
        
        self.orchestrator.temperature = self.temperature
        
        logger.info(f"Set temperature to {self.temperature} for all agents")
    
    async def process_query(
        self,
        query: str,
        agent_type: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process a user query through the swarm.
        
        Args:
            query: The user's query
            agent_type: Optional agent type to use directly
            
        Returns:
            Tuple of (response, agent_name)
        """
        if not query.strip():
            return "Please provide a query.", None
            
        try:
            # Use a specific agent if requested
            if agent_type:
                if agent_type in self.agents:
                    agent = self.agents[agent_type]
                    response = await agent.generate_response(query)
                    return response, agent.name
                else:
                    logger.warning(f"Requested agent '{agent_type}' not found, falling back to orchestrator")
            
            # Otherwise use the orchestrator to determine the best agent
            response = await self.orchestrator.generate_response(query)
            
            # Get the name of the agent that actually generated the response
            # (This works because the orchestrator adds this info to its context)
            agent_name = None
            for msg in reversed(self.orchestrator.context):
                if msg.role == "assistant" and msg.content == response:
                    # The agent name is stored in the context from delegation
                    agent_name = self.orchestrator.name
                    if "[Delegated to " in msg.content:
                        try:
                            # Extract the agent name from the delegated message
                            agent_name = msg.content.split("[Delegated to ")[1].split("]")[0]
                        except:
                            pass
                    break
            
            return response, agent_name
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}", "Error"
    
    async def process_query_stream(
        self,
        query: str,
        agent_type: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, bool, Dict[str, Any]], None]:
        """
        Process a query and stream the response.
        
        Args:
            query: The user's query
            agent_type: Optional agent type to use directly
            
        Yields:
            Tuples of (response_chunk, is_done, metadata)
        """
        if not query.strip():
            yield "Please provide a query.", True, {"agent_name": None}
            return
            
        try:
            # For now, we'll simulate streaming by getting the full response
            # and chunking it. A real implementation would use the streaming API
            # of each model directly.
            response, agent_name = await self.process_query(query, agent_type)
            
            # Simulate streaming by yielding in chunks
            chunk_size = 10  # characters
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i+chunk_size]
                is_done = (i + chunk_size >= len(response))
                metadata = {"agent_name": agent_name} if is_done else {}
                yield chunk, is_done, metadata
                await asyncio.sleep(0.01)  # Small delay for simulated typing
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield f"I encountered an error: {str(e)}", True, {"agent_name": "Error"}
    
    async def add_file_to_memory(self, file_path: str) -> str:
        """
        Add a file to the memory system.
        
        Args:
            file_path: Path to the file to add
            
        Returns:
            Document ID of the added file
        """
        try:
            document_id = await self.vector_store.add_file(file_path)
            logger.info(f"Added file to memory: {file_path} with ID: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Error adding file to memory: {str(e)}")
            raise
    
    async def search_memory(self, query: str, limit: int = 5) -> List[Any]:
        """
        Search the memory for relevant documents.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            documents = await self.vector_store.retrieve(query, limit=limit)
            logger.info(f"Memory search for '{query}' found {len(documents)} results")
            return documents
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            raise
    
    async def add_interaction(self, query: str, response: str, interaction_type: str = "general") -> str:
        """
        Add a user-agent interaction to memory.
        
        Args:
            query: The user's query
            response: The agent's response
            interaction_type: Type of interaction
            
        Returns:
            Interaction ID
        """
        try:
            interaction_id = await self.vector_store.add_interaction(query, response, interaction_type)
            return interaction_id
        except Exception as e:
            logger.error(f"Error adding interaction to memory: {str(e)}")
            # Don't raise - this is a non-critical operation
            return ""
    
    async def retrieve(self, query: str, limit: int = 3, doc_types: Optional[List[str]] = None) -> List[Any]:
        """
        Retrieve relevant documents from memory.
        
        This method is used by agents to get context for their responses.
        
        Args:
            query: The query to search for
            limit: Maximum number of documents to return
            doc_types: Optional list of document types to filter by
            
        Returns:
            List of relevant documents
        """
        try:
            documents = await self.vector_store.retrieve(query, doc_types=doc_types, limit=limit)
            return documents
        except Exception as e:
            logger.error(f"Error retrieving from memory: {str(e)}")
            return []
    
    async def shutdown(self) -> None:
        """Shut down the swarm manager gracefully."""
        logger.info("Shutting down Swarm Manager")
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                
        # Persist vector store
        try:
            self.vector_store.persist()
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")
            
        # Close model client
        try:
            await self.model_client.close()
        except Exception as e:
            logger.error(f"Error closing model client: {str(e)}")
        
        logger.info("Swarm Manager shutdown complete") 