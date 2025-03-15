"""
AI agents subpackage.

This module contains all the specialized agents used by the AI Assistant Swarm system.

Agent implementations for the AI Assistant Swarm System.

This package contains specialized agents that form the swarm:
- Medical Specialist: Powered by Gemma3 27B for medical and health-related queries
- Senior Python Developer: Powered by DeepSeek-RAG for coding assistance
- Master Orchestrator: Powered by Qwen for task coordination
- Fallback Agent: Handles queries outside the specialty domains
"""

from .base import BaseAgent
from .medical import MedicalAgent
from .python import PythonAgent
from .orchestrator import OrchestratorAgent
from .fallback import FallbackAgent

__all__ = [
    'BaseAgent',
    'MedicalAgent',
    'PythonAgent',
    'OrchestratorAgent',
    'FallbackAgent',
] 