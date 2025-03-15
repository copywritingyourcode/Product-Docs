"""
User interfaces for the AI Assistant Swarm.

This package includes two main interfaces:
- Command Line Interface (CLI): For terminal-based interaction
- Web Interface (Gradio): For browser-based interaction with a more visual experience

Both interfaces access the same underlying system but provide different user experiences.
"""

from .common import UserInterface, Message, Role

__all__ = [
    'UserInterface',
    'Message',
    'Role',
] 