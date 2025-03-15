"""
Common Interface Components for AI Assistant Swarm.

This module provides shared functionality and data structures
for both the CLI and web interfaces.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logger = logging.getLogger("interfaces.common")

class Role(str, Enum):
    """Roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    """A message in a conversation."""
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    agent_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "time": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S"),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary."""
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            agent_name=data.get("agent_name"),
        )

@dataclass
class Conversation:
    """A conversation between a user and the assistant swarm."""
    messages: List[Message] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"conv_{int(time.time())}")
    title: Optional[str] = None
    
    def add_message(self, role: Union[Role, str], content: str, agent_name: Optional[str] = None) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender
            content: The content of the message
            agent_name: Optional name of the agent (for assistant messages)
            
        Returns:
            The added message
        """
        # Convert string role to enum if needed
        if isinstance(role, str):
            role = Role(role)
            
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=agent_name,
        )
        self.messages.append(message)
        return message
    
    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message in the conversation."""
        for message in reversed(self.messages):
            if message.role == Role.USER:
                return message
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the last assistant message in the conversation."""
        for message in reversed(self.messages):
            if message.role == Role.ASSISTANT:
                return message
        return None
    
    def get_context_window(self, max_messages: int = 10) -> List[Message]:
        """
        Get the recent context window of messages.
        
        Args:
            max_messages: Maximum number of messages to include
            
        Returns:
            List of recent messages
        """
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title or f"Conversation {self.id}",
            "messages": [m.to_dict() for m in self.messages],
            "timestamp": self.messages[0].timestamp if self.messages else time.time(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a Conversation from a dictionary."""
        conv = cls(
            id=data.get("id", f"conv_{int(time.time())}"),
            title=data.get("title"),
        )
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return conv

class UserInterface(ABC):
    """
    Abstract base class for user interfaces.
    
    This defines the common interface that both CLI and Web UI
    must implement, ensuring consistency between interfaces.
    """
    
    def __init__(self, swarm_manager):
        """
        Initialize the user interface.
        
        Args:
            swarm_manager: The manager for the assistant swarm
        """
        self.swarm_manager = swarm_manager
        self.current_conversation = Conversation()
        self.conversations: Dict[str, Conversation] = {}
        self.logger = logging.getLogger(f"interfaces.{self.__class__.__name__}")
        
    @abstractmethod
    async def start(self):
        """Start the user interface."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Shut down the user interface."""
        pass
    
    async def process_message(self, message: str, agent_type: Optional[str] = None) -> str:
        """
        Process a user message and get a response.
        
        Args:
            message: The user's message
            agent_type: Optional specific agent to use
            
        Returns:
            The assistant's response
        """
        # Add user message to conversation
        self.current_conversation.add_message(Role.USER, message)
        
        self.logger.info(f"Processing message: '{message[:50]}...' for agent_type={agent_type}")
        
        try:
            # Get response from swarm manager
            response, agent_name = await self.swarm_manager.process_query(message, agent_type)
            
            # Add assistant message to conversation
            self.current_conversation.add_message(Role.ASSISTANT, response, agent_name)
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            error_response = f"I apologize, but I encountered an error while processing your request. Error: {str(e)}"
            self.current_conversation.add_message(Role.ASSISTANT, error_response)
            return error_response
    
    def new_conversation(self) -> Conversation:
        """
        Start a new conversation.
        
        Returns:
            The new conversation
        """
        # Save current conversation if it has messages
        if self.current_conversation.messages:
            self.conversations[self.current_conversation.id] = self.current_conversation
            
        # Create new conversation
        self.current_conversation = Conversation()
        self.logger.info(f"Starting new conversation: {self.current_conversation.id}")
        return self.current_conversation
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load an existing conversation.
        
        Args:
            conversation_id: The ID of the conversation to load
            
        Returns:
            The loaded conversation or None if not found
        """
        if conversation_id in self.conversations:
            self.current_conversation = self.conversations[conversation_id]
            self.logger.info(f"Loaded conversation: {conversation_id}")
            return self.current_conversation
        else:
            self.logger.warning(f"Conversation not found: {conversation_id}")
            return None
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        Get a list of available conversations.
        
        Returns:
            List of conversation summaries
        """
        # Include current conversation if it has messages
        all_conversations = dict(self.conversations)
        if self.current_conversation.messages:
            all_conversations[self.current_conversation.id] = self.current_conversation
            
        # Convert to list of summary dictionaries
        return [
            {
                "id": conv_id,
                "title": conv.title or f"Conversation {conv_id}",
                "message_count": len(conv.messages),
                "last_updated": max([m.timestamp for m in conv.messages]) if conv.messages else 0,
                "preview": conv.messages[-1].content[:50] + "..." if conv.messages else "",
            }
            for conv_id, conv in all_conversations.items()
        ] 