"""
Master Orchestrator Agent for the AI Assistant Swarm.

This agent serves as the central coordinator, handling general user interactions
and delegating tasks to specialized agents based on the query content.
"""

import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Union, Tuple

from .base import BaseAgent
from .model_client import OllamaClient

# Configure logging
logger = logging.getLogger("agent.orchestrator")

class OrchestratorAgent(BaseAgent):
    """
    Master Orchestrator Agent for coordinating the AI Assistant Swarm.
    
    This agent:
    - Analyzes user queries to determine the most appropriate specialist agent
    - Delegates tasks to specialized agents
    - Handles general conversation and system management
    - Coordinates multi-agent collaboration when needed
    """
    
    DEFAULT_SYSTEM_PROMPT = """
    You are the Master Orchestrator of an AI Assistant Swarm system.
    Your job is to analyze user queries and determine which specialist agent should handle them,
    or to respond directly for general questions.
    
    You have these specialist agents at your disposal:
    1. Medical Specialist (for health and medical topics)
    2. Senior Python Developer (for Python programming, coding, and software development)
    3. General Assistant (fallback for other topics)
    
    When delegating:
    - For medical questions (symptoms, diseases, health advice, etc.), use the Medical Specialist
    - For Python coding (tutorials, debugging, concepts, etc.), use the Senior Python Developer
    - For other general questions, use the General Assistant or respond directly
    
    Some queries might require multiple agents. In this case, you should:
    1. Break down the task
    2. Delegate to each appropriate specialist
    3. Synthesize their answers into a coherent response
    
    Always prioritize giving the user the most accurate and helpful response.
    """
    
    def __init__(
        self,
        model_name: str = "qwen",
        memory_manager = None,
        max_context_length: int = 4096,
        temperature: float = 0.7,
        agents: Dict[str, BaseAgent] = None
    ):
        """
        Initialize the Master Orchestrator agent.
        
        Args:
            model_name: The Ollama model to use (default: qwen)
            memory_manager: Optional memory manager for long-term memory
            max_context_length: Maximum context length for the model
            temperature: Temperature for model inference
            agents: Dictionary of specialist agents to delegate to
        """
        super().__init__(
            name="Master Orchestrator",
            model_name=model_name,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            memory_manager=memory_manager,
            max_context_length=max_context_length,
            temperature=temperature,
        )
        self.ollama_client = OllamaClient()
        self.agents = agents or {}
        
    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """
        Register a specialist agent with the orchestrator.
        
        Args:
            agent_type: The type of agent ('medical', 'python', 'fallback')
            agent: The agent instance
        """
        self.agents[agent_type] = agent
        logger.info(f"Registered {agent.name} as {agent_type} agent")
        
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Get a registered agent by type.
        
        Args:
            agent_type: The type of agent to get
            
        Returns:
            The agent instance or None if not found
        """
        return self.agents.get(agent_type)
    
    async def _determine_agent_type(self, query: str) -> str:
        """
        Determine which agent should handle the query.
        
        This is a rule-based approach that could be enhanced with
        more sophisticated classification in future versions.
        
        Args:
            query: The user's query
            
        Returns:
            The agent type to use ('medical', 'python', or 'fallback')
        """
        # Check for explicit agent request in format "Agent: query"
        agent_prefix_match = re.match(r'^(medical|doctor|health|python|code|programming|general):\s*(.*)', query, re.IGNORECASE)
        if agent_prefix_match:
            prefix = agent_prefix_match.group(1).lower()
            if prefix in ('medical', 'doctor', 'health'):
                return 'medical'
            elif prefix in ('python', 'code', 'programming'):
                return 'python'
            elif prefix == 'general':
                return 'fallback'
        
        # Simple keyword-based routing - could be replaced with ML classification
        medical_keywords = [
            'health', 'medical', 'doctor', 'disease', 'symptom', 'pain', 
            'treatment', 'diagnosis', 'medicine', 'vaccine', 'virus',
            'infection', 'cancer', 'diabetes', 'heart', 'blood', 'pressure'
        ]
        
        python_keywords = [
            'python', 'code', 'programming', 'function', 'class', 'module',
            'error', 'exception', 'import', 'syntax', 'variable', 'list',
            'dictionary', 'tuple', 'debug', 'algorithm', 'library'
        ]
        
        # Count keyword hits for each category
        medical_score = sum(1 for kw in medical_keywords if kw in query.lower())
        python_score = sum(1 for kw in python_keywords if kw in query.lower())
        
        # Make decision based on keyword scores
        if medical_score > python_score and medical_score > 0:
            return 'medical'
        elif python_score > medical_score and python_score > 0:
            return 'python'
        
        # If no clear winner from keywords, use Qwen to classify
        try:
            classification_prompt = f"""
            Analyze this user query and determine the most appropriate category from these options:
            - medical (for health topics, symptoms, treatments, etc.)
            - python (for programming, coding, debugging, etc.)
            - general (for all other topics)
            
            Reply with ONLY ONE of these exact categories.
            
            User query: {query}
            """
            
            response = await self.ollama_client.generate(
                model_name=self.model_name,
                prompt=classification_prompt,
                temperature=0.3,  # Lower temperature for more deterministic classification
            )
            
            response = response.strip().lower()
            logger.info(f"Classification response: {response}")
            
            if 'medical' in response:
                return 'medical'
            elif 'python' in response or 'programming' in response or 'coding' in response:
                return 'python'
            else:
                return 'fallback'
                
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            # Default to fallback if classification fails
            return 'fallback'
    
    async def generate_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Process the query and generate a response, potentially delegating to specialist agents.
        
        Args:
            query: The user's query
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            The agent's response or the delegated response from a specialist
        """
        # Always add user query to orchestrator's context
        self.add_to_context("user", query)
        
        try:
            # Determine which agent should handle this query
            agent_type = await self._determine_agent_type(query)
            logger.info(f"Determined agent type: {agent_type} for query: {query[:50]}...")
            
            # Get the appropriate agent
            agent = self.get_agent(agent_type)
            if not agent:
                logger.warning(f"Agent type {agent_type} not registered, falling back to direct response")
                # Handle directly if no appropriate agent is registered
                return await self._direct_response(query, retrieve_from_memory)
            
            # Delegate to the specialist agent
            response = await agent.generate_response(query, retrieve_from_memory)
            
            # Add the delegated response to context
            self.add_to_context("assistant", f"[Delegated to {agent.name}] {response}")
            
            # Return the specialist's response
            return response
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            # Handle the error gracefully
            error_response = f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Error details: {str(e)}"
            self.add_to_context("assistant", error_response)
            return error_response
    
    async def _direct_response(self, query: str, retrieve_from_memory: bool = True) -> str:
        """
        Generate a direct response using the orchestrator's model.
        
        This is used when no specialist agent is available or for simple queries.
        
        Args:
            query: The user's query
            retrieve_from_memory: Whether to retrieve relevant information from memory
            
        Returns:
            The orchestrator's direct response
        """
        # Check if memory retrieval is needed and available
        relevant_context = ""
        if retrieve_from_memory and self.memory_manager:
            try:
                documents = await self.memory_manager.retrieve(query, limit=2)
                if documents:
                    relevant_context = "Relevant information I know:\n\n"
                    for i, doc in enumerate(documents, 1):
                        relevant_context += f"{i}. {doc.page_content}\n\n"
                    logger.info(f"Retrieved {len(documents)} documents from memory")
            except Exception as e:
                logger.error(f"Error retrieving from memory: {str(e)}")
                
        # Prepare the enhanced prompt with any retrieved context
        enhanced_query = query
        if relevant_context:
            enhanced_query = f"{relevant_context}\n\nUser query: {query}\n\nPlease use the above information if it's relevant to the question."
        
        try:
            # Generate response using orchestrator's model
            prompt = self.format_prompt(enhanced_query)
            response = await self.ollama_client.generate(
                model_name=self.model_name,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
            )
            
            # Add the response to the context
            self.add_to_context("assistant", response)
            
            # Trim the context if needed
            self._truncate_context_if_needed()
            
            # Store the interaction if memory manager exists
            if self.memory_manager:
                try:
                    await self.memory_manager.add_interaction(query, response, "direct_query")
                except Exception as e:
                    logger.error(f"Error storing in memory: {str(e)}")
                    
            return response
        except Exception as e:
            logger.error(f"Error generating direct response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response. Please try again or rephrase your question. Error details: {str(e)}"
    
    async def multi_agent_response(self, query: str, agent_types: List[str]) -> str:
        """
        Generate a response using multiple agents and combine their outputs.
        
        Args:
            query: The user's query
            agent_types: List of agent types to use
            
        Returns:
            Combined response from multiple agents
        """
        responses = {}
        tasks = []
        
        # Create tasks for each agent
        for agent_type in agent_types:
            agent = self.get_agent(agent_type)
            if agent:
                tasks.append(self._get_agent_response(agent, query, agent_type))
        
        # Run all agent tasks concurrently
        if tasks:
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for agent_type, result in agent_results:
                if isinstance(result, Exception):
                    logger.error(f"Error from {agent_type} agent: {str(result)}")
                    responses[agent_type] = f"Error from {agent_type} agent: {str(result)}"
                else:
                    responses[agent_type] = result
        
        # Combine responses
        if not responses:
            return "I couldn't get responses from any agents. Please try again."
        
        combined_response = "I consulted multiple specialists for your question:\n\n"
        for agent_type, response in responses.items():
            agent_name = self.get_agent(agent_type).name if self.get_agent(agent_type) else agent_type
            combined_response += f"== {agent_name} ==\n{response}\n\n"
        
        return combined_response
    
    async def _get_agent_response(self, agent: BaseAgent, query: str, agent_type: str) -> Tuple[str, str]:
        """
        Get response from a specific agent.
        
        Args:
            agent: The agent to query
            query: The user's query
            agent_type: The type of agent (for result tracking)
            
        Returns:
            Tuple of (agent_type, response)
        """
        try:
            response = await agent.generate_response(query)
            return (agent_type, response)
        except Exception as e:
            return (agent_type, e) 