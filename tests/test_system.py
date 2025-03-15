"""
System tests for the AI Assistant Swarm.

This module provides end-to-end tests to verify that 
the complete swarm system works as expected, including
the SwarmManager, CLI, and Web interfaces.
"""

import os
import sys
import asyncio
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import logging
import time
import json
import shutil

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Add the parent directory to the path to import the swarm package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm.swarm_manager import SwarmManager
from swarm.agents.model_client import OllamaClient
from swarm.cli import CommandLineInterface
from swarm.webapp import WebInterface
from swarm.memory.document import Document, DocumentMetadata

class TestSwarmSystemIntegration(unittest.TestCase):
    """End-to-end system tests for the AI Assistant Swarm."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the swarm data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the OllamaClient
        self.mock_client_patcher = patch('swarm.agents.model_client.OllamaClient')
        self.mock_client = self.mock_client_patcher.start()
        
        # Configure the mock to return predefined responses
        instance = self.mock_client.return_value
        async def mock_generate(*args, **kwargs):
            model_name = kwargs.get('model_name', args[0] if args else 'unknown')
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else 'unknown')
            
            # For the orchestrator's agent classification
            if "analyze this user query" in prompt.lower():
                if "headache" in prompt.lower() or "health" in prompt.lower():
                    return "medical"
                elif "python" in prompt.lower() or "function" in prompt.lower():
                    return "python"
                else:
                    return "general"
            
            # For the medical agent
            elif "gemma" in model_name and ("headache" in prompt.lower() or "health" in prompt.lower()):
                return "Medical Specialist: Headaches can be caused by stress, dehydration, or other health factors."
            
            # For the Python agent
            elif "deepseek" in model_name and ("python" in prompt.lower() or "function" in prompt.lower()):
                return "Python Developer: Here's a function example:\n```python\ndef hello():\n    print('Hello, world!')\n```"
            
            # For the fallback agent
            else:
                return f"General Assistant: I can help with various topics. You asked about {prompt.split('user:')[-1].strip()}"
                
        instance.generate.side_effect = mock_generate
        
        # For model checking, return all models as available
        async def mock_list_models():
            return {
                "models": [
                    {"name": "gemma:27b", "size": 27000000000},
                    {"name": "deepseek-coder:33b", "size": 33000000000},
                    {"name": "qwen:14b", "size": 14000000000}
                ]
            }
        instance.list_models.side_effect = mock_list_models
        
        # Initialize the SwarmManager with the temp directory
        self.swarm_manager = SwarmManager(data_dir=self.temp_dir)
        
        # Mock the vector store's query method to return predictable results
        async def mock_query(query_text, **kwargs):
            if "python" in query_text.lower():
                return [
                    Document(
                        text="Python is a high-level programming language.",
                        metadata=DocumentMetadata(
                            source="python_info.txt",
                            doc_type="text",
                            creation_date=time.time(),
                            last_accessed=time.time(),
                            access_count=1,
                            retention_policy="standard"
                        )
                    )
                ]
            elif "health" in query_text.lower() or "headache" in query_text.lower():
                return [
                    Document(
                        text="Headaches can be symptoms of various conditions.",
                        metadata=DocumentMetadata(
                            source="health_info.txt",
                            doc_type="text",
                            creation_date=time.time(),
                            last_accessed=time.time(),
                            access_count=1,
                            retention_policy="standard"
                        )
                    )
                ]
            else:
                return []
                
        self.swarm_manager.vector_store.query = mock_query
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Stop the mock
        self.mock_client_patcher.stop()
    
    async def test_swarm_manager_initialization(self):
        """Test the SwarmManager initializes correctly."""
        # Verify the SwarmManager has created all necessary agents
        self.assertIsNotNone(self.swarm_manager.orchestrator_agent)
        
        # Check that the orchestrator has registered the specialized agents
        self.assertGreaterEqual(len(self.swarm_manager.orchestrator_agent.agents), 3)
        
        # Verify the data directory exists
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Check models availability
        missing_models = await self.swarm_manager.check_models()
        self.assertEqual(len(missing_models), 0, "Expected all models to be available")
    
    async def test_swarm_manager_process_query(self):
        """Test the SwarmManager processes queries correctly."""
        # Medical query
        response, agent_name = await self.swarm_manager.process_query(
            "I've been having frequent headaches. What could be causing this?"
        )
        self.assertIn("Medical Specialist", agent_name)
        self.assertIn("headache", response.lower())
        
        # Python query
        response, agent_name = await self.swarm_manager.process_query(
            "How do I define a Python function?"
        )
        self.assertIn("Python Developer", agent_name)
        self.assertIn("function", response.lower())
        
        # General query
        response, agent_name = await self.swarm_manager.process_query(
            "What's the weather like today?"
        )
        self.assertIn("General Assistant", agent_name)
        self.assertIn("weather", response.lower())
    
    async def test_swarm_manager_streaming(self):
        """Test the SwarmManager streams responses correctly."""
        # Test streaming with a medical query
        async for chunk in self.swarm_manager.process_query_stream(
            "I've been having frequent headaches. What could be causing this?"
        ):
            self.assertIsInstance(chunk, dict)
            self.assertIn("text", chunk)
            self.assertIn("agent", chunk)
    
    async def test_swarm_manager_memory_integration(self):
        """Test the SwarmManager integrates with memory system."""
        # Create a test file
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file_path, "w") as f:
            f.write("This is a test document about Python programming.")
        
        # Add the file to memory
        await self.swarm_manager.add_file_to_memory(test_file_path)
        
        # Search memory
        results = await self.swarm_manager.search_memory("Python programming")
        
        # Verify results include the test document
        self.assertGreaterEqual(len(results), 1)
        
        # Clean up
        os.remove(test_file_path)
    
    async def test_cli_interface(self):
        """Test the CLI interface works correctly."""
        # Create a CLI interface with the test swarm manager
        cli = CommandLineInterface(swarm_manager=self.swarm_manager)
        
        # Mock stdin/stdout
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        
        # Mock readline behavior for multiple inputs then exit
        mock_inputs = [
            "Hello, how can you help me?",  # General greeting
            "/agent python",               # Switch to Python agent
            "How do I write a function?",  # Python-specific query
            "/exit"                        # Exit command
        ]
        
        # Configure stdin.readline to return each input in sequence
        mock_stdin.readline.side_effect = mock_inputs
        
        # Replace the real stdin/stdout and run CLI start in a non-blocking way
        original_stdin = sys.stdin
        original_stdout = sys.stdout
        sys.stdin = mock_stdin
        sys.stdout = mock_stdout
        
        # Run CLI in a task that we'll cancel after our mocked interactions
        async def run_cli():
            await cli.start()
            
        task = asyncio.create_task(run_cli())
        
        # Give it a moment to process our mocked inputs
        await asyncio.sleep(0.1)
        
        # Cancel the CLI task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
            
        # Restore stdin/stdout
        sys.stdin = original_stdin
        sys.stdout = original_stdout
        
        # Verify the CLI processed the agent switch command
        self.assertEqual(cli.current_agent_type, "python")
    
    async def test_web_interface_initialization(self):
        """Test the Web interface initializes correctly."""
        # Create a Web interface with the test swarm manager
        web = WebInterface(swarm_manager=self.swarm_manager)
        
        # Mock the gradio.Blocks to prevent actual server startup
        with patch('gradio.Blocks') as mock_blocks:
            # Configure the mock
            mock_instance = mock_blocks.return_value
            mock_instance.launch = MagicMock(return_value=None)
            
            # Start the web interface (non-blocking)
            try:
                await asyncio.wait_for(web.start(), timeout=0.5)
            except asyncio.TimeoutError:
                # Expected - the interface would normally run indefinitely
                pass
                
            # Verify the interface tried to launch
            mock_instance.launch.assert_called_once()
            
            # Shut down the interface
            await web.shutdown()

# Run the tests
if __name__ == '__main__':
    # Create and run an async test suite
    async def run_async_tests():
        # TestSwarmSystemIntegration
        system_suite = unittest.TestLoader().loadTestsFromTestCase(TestSwarmSystemIntegration)
        for test in system_suite:
            await test._callTestMethod()
            
    # Run the async tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_async_tests())
    loop.close()
    print("All system tests completed.") 