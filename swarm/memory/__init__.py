"""
Memory management subpackage.

This module contains the vector storage and retention management for the AI Assistant Swarm.

This package handles long-term memory storage using a vector database (ChromaDB),
implementing various retention policies for different types of data:
- Research papers: Stored indefinitely in compressed form
- Chat logs and PDFs: Kept for 1 year
- Medical documents: Kept forever in hot storage with multi-layer validation
"""

from .vector_store import VectorStore, DocumentType
from .retention import RetentionPolicy, RetentionManager

__all__ = [
    'VectorStore',
    'DocumentType',
    'RetentionPolicy',
    'RetentionManager',
] 