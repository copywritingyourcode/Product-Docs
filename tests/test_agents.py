"""
Tests for the AI Assistant Swarm agents.

This module provides tests to verify that the agents
in the system are working correctly.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import logging

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Add the parent directory to the path to import the swarm package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm.agents.base import BaseAgent, Message
from swarm.agents.medical import MedicalAgent
from swarm.agents.python import PythonAgent
from swarm.agents.orchestrator import OrchestratorAgent
from swarm.agents.fallback import FallbackAgent
from swarm.agents.model_client import OllamaClient

class TestBaseAgent(unittest.TestCase):
    """Tests for the BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete subclass for testing the abstract base class
        class ConcreteAgent(BaseAgent):
            async def generate_response(self, query, retrieve_from_memory=True):
                return f"Response to: {query}"
                
        self.agent = ConcreteAgent(
            name="Test Agent",
            model_name="test-model",
            system_prompt="You are a test agent.",
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "Test Agent")
        self.assertEqual(self.agent.model_name, "test-model")
        self.assertEqual(self.agent.system_prompt, "You are a test agent.")
        self.assertEqual(self.agent.temperature, 0.7)  # Default value
        
    def test_context_management(self):
        """Test adding to and resetting context."""
        # Add messages to context
        self.agent.add_to_context("user", "Hello")
        self.agent.add_to_context("assistant", "Hi there")
        
        # Check context
        self.assertEqual(len(self.agent.context), 2)
        self.assertEqual(self.agent.context[0].role, "user")
        self.assertEqual(self.agent.context[0].content, "Hello")
        self.assertEqual(self.agent.context[1].role, "assistant")
        self.assertEqual(self.agent.context[1].content, "Hi there")
        
        # Reset context
        self.agent.reset_context()
        self.assertEqual(len(self.agent.context), 0)
        
    def test_format_prompt(self):
        """Test prompt formatting."""
        self.agent.add_to_context("user", "Hello")
        self.agent.add_to_context("assistant", "Hi there")
        
        prompt = self.agent.format_prompt("How are you?")
        
        # Check that the prompt contains all the expected parts
        self.assertIn("system: You are a test agent.", prompt)
        self.assertIn("user: Hello", prompt)
        self.assertIn("assistant: Hi there", prompt)
        self.assertIn("user: How are you?", prompt)

class TestSpecialistAgents(unittest.TestCase):
    """Tests for the specialist agents."""
    
    def setUp(self):
        """Set up test fixtures with mocked Ollama client."""
        # Create a mock for the OllamaClient
        self.mock_client_patcher = patch('swarm.agents.model_client.OllamaClient')
        self.mock_client = self.mock_client_patcher.start()
        
        # Configure the mock to return a predefined response
        instance = self.mock_client.return_value
        async def mock_generate(*args, **kwargs):
            model_name = kwargs.get('model_name', args[0] if args else 'unknown')
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else 'unknown')
            
            if 'medical' in prompt.lower() or 'health' in prompt.lower():
                return "This is a medical response about health."
            elif 'python' in prompt.lower() or 'code' in prompt.lower():
                return "Here's some Python code: print('Hello World')"
            else:
                return f"General response from {model_name}"
                
        instance.generate.side_effect = mock_generate
        
        # Initialize the agents
        self.medical_agent = MedicalAgent()
        self.python_agent = PythonAgent()
        self.fallback_agent = FallbackAgent()
        
        # Create a mock memory manager
        self.mock_memory = MagicMock()
        async def mock_retrieve(*args, **kwargs):
            return []  # No memory results for simplicity
        self.mock_memory.retrieve.side_effect = mock_retrieve
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_client_patcher.stop()
    
    async def test_medical_agent(self):
        """Test the Medical Specialist agent."""
        response = await self.medical_agent.generate_response("What causes headaches?")
        self.assertIn("medical response", response.lower())
        
    async def test_python_agent(self):
        """Test the Python Developer agent."""
        response = await self.python_agent.generate_response("How do I write a Python function?")
        self.assertIn("python", response.lower())
        self.assertIn("code", response.lower())
        
    async def test_fallback_agent(self):
        """Test the Fallback agent."""
        response = await self.fallback_agent.generate_response("What's the weather like today?")
        self.assertIn("general response", response.lower())
        
class TestOrchestratorAgent(unittest.TestCase):
    """Tests for the Orchestrator agent."""
    
    def setUp(self):
        """Set up test fixtures with mocked agents."""
        # Create a mock for the OllamaClient
        self.mock_client_patcher = patch('swarm.agents.model_client.OllamaClient')
        self.mock_client = self.mock_client_patcher.start()
        
        # Configure the mock
        instance = self.mock_client.return_value
        async def mock_generate(*args, **kwargs):
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else 'unknown')
            
            # For classification in the orchestrator
            if "analyze this user query" in prompt.lower():
                if "headache" in prompt.lower() or "pain" in prompt.lower():
                    return "medical"
                elif "python" in prompt.lower() or "function" in prompt.lower():
                    return "python"
                else:
                    return "general"
            else:
                return "Orchestrator response"
                
        instance.generate.side_effect = mock_generate
        
        # Create mock specialist agents
        self.mock_medical = MagicMock()
        async def mock_medical_response(*args, **kwargs):
            return "Medical Specialist: This is a response about health."
        self.mock_medical.generate_response.side_effect = mock_medical_response
        self.mock_medical.name = "Medical Specialist"
        
        self.mock_python = MagicMock()
        async def mock_python_response(*args, **kwargs):
            return "Python Developer: Here's some code."
        self.mock_python.generate_response.side_effect = mock_python_response
        self.mock_python.name = "Python Developer"
        
        self.mock_fallback = MagicMock()
        async def mock_fallback_response(*args, **kwargs):
            return "General Assistant: General information."
        self.mock_fallback.generate_response.side_effect = mock_fallback_response
        self.mock_fallback.name = "General Assistant"
        
        # Initialize the orchestrator
        self.orchestrator = OrchestratorAgent()
        
        # Register mock agents
        self.orchestrator.register_agent("medical", self.mock_medical)
        self.orchestrator.register_agent("python", self.mock_python)
        self.orchestrator.register_agent("fallback", self.mock_fallback)
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_client_patcher.stop()
        
    async def test_agent_delegation_medical(self):
        """Test delegation to Medical Specialist."""
        await self.orchestrator.generate_response("I have a headache. What could cause this?")
        self.mock_medical.generate_response.assert_called_once()
        self.mock_python.generate_response.assert_not_called()
        
    async def test_agent_delegation_python(self):
        """Test delegation to Python Developer."""
        await self.orchestrator.generate_response("How do I write a Python function?")
        self.mock_python.generate_response.assert_called_once()
        self.mock_medical.generate_response.assert_not_called()
        
    async def test_agent_delegation_fallback(self):
        """Test delegation to Fallback agent."""
        await self.orchestrator.generate_response("What's the capital of France?")
        self.mock_fallback.generate_response.assert_called_once()
        self.mock_medical.generate_response.assert_not_called()
        self.mock_python.generate_response.assert_not_called()

# Run the tests
if __name__ == '__main__':
    # Run synchronous tests
    unittest.main(exit=False)
    
    # Run async tests
    loop = asyncio.get_event_loop()
    
    # TestSpecialistAgents
    specialist_suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecialistAgents)
    for test in specialist_suite:
        loop.run_until_complete(test._callTestMethod())
        
    # TestOrchestratorAgent
    orchestrator_suite = unittest.TestLoader().loadTestsFromTestCase(TestOrchestratorAgent)
    for test in orchestrator_suite:
        loop.run_until_complete(test._callTestMethod())
        
    loop.close()
    print("All tests completed.") 