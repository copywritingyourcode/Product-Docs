"""
Vector Store for AI Assistant Swarm's Memory System.

This module provides functionality for storing and retrieving
documents and conversations in a vector database (ChromaDB)
for semantic search.
"""

import os
import time
import logging
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logger = logging.getLogger("memory.vector_store")

class DocumentType(str, Enum):
    """Enum representing different types of documents in the memory system."""
    CHAT_LOG = "chat_log"
    RESEARCH_PAPER = "research_paper"
    MEDICAL_DOCUMENT = "medical_document"
    PYTHON_CODE = "python_code"
    GENERAL_DOCUMENT = "general_document"
    USER_UPLOAD = "user_upload"
    
    @classmethod
    def from_str(cls, doc_type: str) -> 'DocumentType':
        """Convert string to DocumentType."""
        try:
            return cls(doc_type)
        except ValueError:
            logger.warning(f"Unknown document type: {doc_type}. Using GENERAL_DOCUMENT.")
            return cls.GENERAL_DOCUMENT

class VectorStore:
    """
    Vector database for the AI Assistant Swarm's memory system.
    
    This class handles storing, retrieving, and managing documents
    and conversation history in a vector database (ChromaDB).
    """
    
    def __init__(
        self,
        persist_directory: Union[str, Path] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory where the vector DB is stored. If None, uses in-memory DB.
            embedding_model_name: The name of the HuggingFace model to use for embeddings
        """
        self.persist_directory = persist_directory
        
        # Initialize embedding function
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Set up ChromaDB
        if persist_directory:
            logger.info(f"Initializing persistent ChromaDB at {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model,
            )
        else:
            logger.info("Initializing in-memory ChromaDB")
            self.db = Chroma(
                embedding_function=self.embedding_model,
            )
        
        # Set up text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info("Vector store initialized")
        
    async def add_document(
        self,
        document_text: str,
        document_type: Union[str, DocumentType],
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            document_text: The text content of the document
            document_type: The type of document
            document_id: Optional ID for the document
            metadata: Optional metadata for the document
            source: Optional source information
            
        Returns:
            The document ID
        """
        if isinstance(document_type, str):
            document_type = DocumentType.from_str(document_type)
            
        # Create a unique ID if none provided
        if document_id is None:
            document_id = f"{document_type.value}_{int(time.time())}_{hash(document_text) % 10000}"
            
        # Prepare metadata
        doc_metadata = metadata or {}
        doc_metadata.update({
            "doc_type": document_type.value,
            "added_at": datetime.now().isoformat(),
            "source": source or "direct_input",
        })
        
        try:
            # Split document into chunks
            docs = self.text_splitter.create_documents(
                texts=[document_text],
                metadatas=[doc_metadata]
            )
            
            # Update document IDs to ensure uniqueness
            for i, doc in enumerate(docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["parent_id"] = document_id
            
            # Add to vector store
            self.db.add_documents(docs)
            
            logger.info(f"Added document {document_id} of type {document_type.value} to vector store")
            return document_id
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    async def add_file(
        self,
        file_path: Union[str, Path],
        document_type: Optional[Union[str, DocumentType]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a file to the vector store.
        
        Args:
            file_path: Path to the file
            document_type: The type of document (if None, inferred from file extension)
            metadata: Optional metadata for the document
            
        Returns:
            The document ID
        """
        file_path = Path(file_path)
        
        # Infer document type from file extension if not provided
        if document_type is None:
            if file_path.suffix.lower() in ['.pdf']:
                document_type = DocumentType.RESEARCH_PAPER
            elif file_path.suffix.lower() in ['.py', '.ipynb']:
                document_type = DocumentType.PYTHON_CODE
            elif file_path.suffix.lower() in ['.txt', '.md', '.docx', '.doc', '.rtf']:
                document_type = DocumentType.GENERAL_DOCUMENT
            else:
                document_type = DocumentType.USER_UPLOAD
        
        # Convert string to enum if needed
        if isinstance(document_type, str):
            document_type = DocumentType.from_str(document_type)
            
        try:
            # Load document based on file type
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            else:
                # Default to text loader
                loader = TextLoader(str(file_path))
                
            # Load the document
            docs = loader.load()
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "doc_type": document_type.value,
                "added_at": datetime.now().isoformat(),
                "source": str(file_path),
                "filename": file_path.name,
            })
            
            # Apply metadata to all doc chunks
            for doc in docs:
                doc.metadata.update(doc_metadata)
                
            # Split into chunks
            split_docs = self.text_splitter.split_documents(docs)
            
            # Generate document ID
            document_id = f"{document_type.value}_{int(time.time())}_{file_path.stem}"
            
            # Update document IDs
            for i, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["parent_id"] = document_id
                
            # Add to vector store
            self.db.add_documents(split_docs)
            
            logger.info(f"Added file {file_path} as document {document_id} to vector store")
            return document_id
        except Exception as e:
            logger.error(f"Error adding file to vector store: {str(e)}")
            raise
    
    async def add_interaction(
        self,
        query: str,
        response: str,
        interaction_type: str = "general",
    ) -> str:
        """
        Add a user-agent interaction to the vector store.
        
        Args:
            query: The user's query
            response: The agent's response
            interaction_type: Type of interaction (general, medical, python)
            
        Returns:
            The interaction ID
        """
        # Prepare document content
        interaction_text = f"USER: {query}\n\nASSISTANT: {response}"
        
        # Determine document type
        if interaction_type == "medical_inquiry":
            doc_type = DocumentType.MEDICAL_DOCUMENT
        elif interaction_type == "python_code":
            doc_type = DocumentType.PYTHON_CODE
        else:
            doc_type = DocumentType.CHAT_LOG
            
        # Add to vector store
        interaction_id = await self.add_document(
            document_text=interaction_text,
            document_type=doc_type,
            metadata={
                "interaction_type": interaction_type,
                "query": query,
                "response": response,
            },
            source="chat_interaction"
        )
        
        return interaction_id
        
    async def retrieve(
        self,
        query: str,
        doc_types: Optional[List[Union[str, DocumentType]]] = None,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: The query to search for
            doc_types: Optional list of document types to filter by
            limit: Maximum number of documents to return
            filter_metadata: Additional metadata filters
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Prepare filter
            filter_dict = filter_metadata or {}
            
            # Add document type filter if provided
            if doc_types:
                # Convert to enum values if needed
                doc_type_values = []
                for dt in doc_types:
                    if isinstance(dt, DocumentType):
                        doc_type_values.append(dt.value)
                    else:
                        doc_type_values.append(DocumentType.from_str(dt).value)
                
                filter_dict["doc_type"] = {"$in": doc_type_values}
            
            # Query the vector store
            results = self.db.similarity_search_with_score(
                query=query,
                k=limit,
                filter=filter_dict if filter_dict else None
            )
            
            # Process and log results
            documents = []
            for doc, score in results:
                logger.debug(f"Retrieved document with score {score}: {doc.page_content[:50]}...")
                documents.append(doc)
                
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving from vector store: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all chunks with matching parent_id
            self.db.delete({"parent_id": document_id})
            logger.info(f"Deleted document {document_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
            
    def persist(self) -> None:
        """Persist the vector store to disk if using persistent storage."""
        if self.persist_directory:
            try:
                self.db.persist()
                logger.info("Vector store persisted to disk")
            except Exception as e:
                logger.error(f"Error persisting vector store: {str(e)}")
                
    def __len__(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            return len(self.db.get()["ids"])
        except Exception:
            return 0
            
    def __str__(self) -> str:
        """String representation of the vector store."""
        return f"VectorStore(persist_directory={self.persist_directory}, document_count={len(self)})" 