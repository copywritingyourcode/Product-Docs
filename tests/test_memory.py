"""
Tests for the AI Assistant Swarm memory system.

This module provides tests to verify that the memory system
in the swarm works correctly, including document storage,
retrieval, and retention policies.
"""

import os
import sys
import time
import asyncio
import shutil
import unittest
from unittest.mock import MagicMock, patch
import logging
import tempfile

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Add the parent directory to the path to import the swarm package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm.memory.vector_store import VectorStore
from swarm.memory.document import Document, DocumentMetadata
from swarm.memory.retention import RetentionManager

class TestDocument(unittest.TestCase):
    """Tests for the Document class."""
    
    def test_document_creation(self):
        """Test document creation with metadata."""
        metadata = DocumentMetadata(
            source="test_file.txt",
            doc_type="text",
            creation_date=1234567890.0,
            last_accessed=1234567890.0,
            access_count=0,
            retention_policy="standard"
        )
        
        doc = Document(
            text="This is a test document.",
            metadata=metadata
        )
        
        self.assertEqual(doc.text, "This is a test document.")
        self.assertEqual(doc.metadata.source, "test_file.txt")
        self.assertEqual(doc.metadata.doc_type, "text")
        self.assertEqual(doc.metadata.creation_date, 1234567890.0)
        self.assertEqual(doc.metadata.retention_policy, "standard")
        
    def test_document_to_dict(self):
        """Test document serialization to dictionary."""
        metadata = DocumentMetadata(
            source="test_file.txt",
            doc_type="text",
            creation_date=1234567890.0,
            last_accessed=1234567890.0,
            access_count=0,
            retention_policy="standard"
        )
        
        doc = Document(
            text="This is a test document.",
            metadata=metadata
        )
        
        doc_dict = doc.to_dict()
        
        self.assertEqual(doc_dict["text"], "This is a test document.")
        self.assertEqual(doc_dict["metadata"]["source"], "test_file.txt")
        self.assertEqual(doc_dict["metadata"]["doc_type"], "text")
        self.assertEqual(doc_dict["metadata"]["creation_date"], 1234567890.0)
        
    def test_document_from_dict(self):
        """Test document deserialization from dictionary."""
        doc_dict = {
            "text": "This is a test document.",
            "metadata": {
                "source": "test_file.txt",
                "doc_type": "text",
                "creation_date": 1234567890.0,
                "last_accessed": 1234567890.0,
                "access_count": 0,
                "retention_policy": "standard"
            }
        }
        
        doc = Document.from_dict(doc_dict)
        
        self.assertEqual(doc.text, "This is a test document.")
        self.assertEqual(doc.metadata.source, "test_file.txt")
        self.assertEqual(doc.metadata.doc_type, "text")
        self.assertEqual(doc.metadata.creation_date, 1234567890.0)
        
    def test_document_access(self):
        """Test document access tracking."""
        metadata = DocumentMetadata(
            source="test_file.txt",
            doc_type="text",
            creation_date=time.time() - 3600,  # 1 hour ago
            last_accessed=time.time() - 3600,  # 1 hour ago
            access_count=0,
            retention_policy="standard"
        )
        
        doc = Document(
            text="This is a test document.",
            metadata=metadata
        )
        
        original_last_accessed = doc.metadata.last_accessed
        original_access_count = doc.metadata.access_count
        
        time.sleep(0.1)  # Small delay to ensure time difference
        
        # Record access
        doc.record_access()
        
        self.assertGreater(doc.metadata.last_accessed, original_last_accessed)
        self.assertEqual(doc.metadata.access_count, original_access_count + 1)

class TestVectorStore(unittest.TestCase):
    """Tests for the VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(self.temp_dir)
        
        # Test documents
        self.doc1 = Document(
            text="Python is a programming language.",
            metadata=DocumentMetadata(
                source="python_info.txt",
                doc_type="text",
                creation_date=time.time(),
                last_accessed=time.time(),
                access_count=0,
                retention_policy="standard"
            )
        )
        
        self.doc2 = Document(
            text="JavaScript is used for web development.",
            metadata=DocumentMetadata(
                source="js_info.txt",
                doc_type="text",
                creation_date=time.time(),
                last_accessed=time.time(),
                access_count=0,
                retention_policy="standard"
            )
        )
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
    async def test_add_and_retrieve_document(self):
        """Test adding and retrieving documents."""
        # Add documents
        await self.vector_store.add_document(self.doc1)
        await self.vector_store.add_document(self.doc2)
        
        # Retrieve documents
        results = await self.vector_store.query("programming language")
        
        # Verify results
        self.assertGreaterEqual(len(results), 1)
        found_python = False
        for doc in results:
            if "Python" in doc.text:
                found_python = True
                break
        self.assertTrue(found_python, "Python document not found in results")
        
    async def test_update_document_metadata(self):
        """Test updating document metadata."""
        # Add document
        await self.vector_store.add_document(self.doc1)
        
        # Get document ID
        results = await self.vector_store.query("Python")
        self.assertGreaterEqual(len(results), 1)
        doc = results[0]
        
        # Update access count
        doc.metadata.access_count += 1
        await self.vector_store.update_document_metadata(doc)
        
        # Retrieve document again
        results = await self.vector_store.query("Python")
        self.assertGreaterEqual(len(results), 1)
        updated_doc = results[0]
        
        # Verify updated metadata
        self.assertEqual(updated_doc.metadata.access_count, 1)
        
    async def test_delete_document(self):
        """Test deleting a document."""
        # Add documents
        await self.vector_store.add_document(self.doc1)
        await self.vector_store.add_document(self.doc2)
        
        # Get document IDs
        python_results = await self.vector_store.query("Python")
        self.assertGreaterEqual(len(python_results), 1)
        python_doc = python_results[0]
        
        # Delete Python document
        await self.vector_store.delete_document(python_doc)
        
        # Verify Python document is gone
        new_python_results = await self.vector_store.query("Python")
        self.assertEqual(len(new_python_results), 0, "Python document was not deleted")
        
        # Verify JavaScript document is still there
        js_results = await self.vector_store.query("JavaScript")
        self.assertGreaterEqual(len(js_results), 1)

class TestRetentionManager(unittest.TestCase):
    """Tests for the RetentionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(self.temp_dir)
        
        # Create the retention manager with mocked time
        self.retention_manager = RetentionManager(self.vector_store)
        
        # Create test documents with different retention policies
        self.standard_doc = Document(
            text="This is a standard document.",
            metadata=DocumentMetadata(
                source="standard.txt",
                doc_type="text",
                creation_date=time.time() - 86400 * 31,  # 31 days old
                last_accessed=time.time() - 86400 * 31,  # 31 days old
                access_count=0,
                retention_policy="standard"
            )
        )
        
        self.important_doc = Document(
            text="This is an important document.",
            metadata=DocumentMetadata(
                source="important.txt",
                doc_type="text",
                creation_date=time.time() - 86400 * 91,  # 91 days old
                last_accessed=time.time() - 86400 * 91,  # 91 days old
                access_count=0,
                retention_policy="important"
            )
        )
        
        self.permanent_doc = Document(
            text="This is a permanent document.",
            metadata=DocumentMetadata(
                source="permanent.txt",
                doc_type="text",
                creation_date=time.time() - 86400 * 366,  # 366 days old
                last_accessed=time.time() - 86400 * 366,  # 366 days old
                access_count=0,
                retention_policy="permanent"
            )
        )
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
    async def test_apply_retention_policies(self):
        """Test application of retention policies."""
        # Add documents
        await self.vector_store.add_document(self.standard_doc)
        await self.vector_store.add_document(self.important_doc)
        await self.vector_store.add_document(self.permanent_doc)
        
        # Mock the vector_store.get_all_documents method
        original_get_all = self.vector_store.get_all_documents
        
        async def mock_get_all_documents():
            return [self.standard_doc, self.important_doc, self.permanent_doc]
            
        self.vector_store.get_all_documents = mock_get_all_documents
        
        # Mock the delete_document method to track deletions
        original_delete = self.vector_store.delete_document
        deleted_docs = []
        
        async def mock_delete_document(doc):
            deleted_docs.append(doc)
            return await original_delete(doc)
            
        self.vector_store.delete_document = mock_delete_document
        
        # Apply retention policies
        await self.retention_manager.apply_retention_policies()
        
        # Verify standard document was deleted (over 30 days old)
        self.assertIn(self.standard_doc, deleted_docs, "Standard document was not deleted")
        
        # Verify important document was not deleted (under 90 days old for important)
        self.assertNotIn(self.important_doc, deleted_docs, "Important document was incorrectly deleted")
        
        # Verify permanent document was not deleted (permanent policy)
        self.assertNotIn(self.permanent_doc, deleted_docs, "Permanent document was incorrectly deleted")
        
        # Restore original methods
        self.vector_store.get_all_documents = original_get_all
        self.vector_store.delete_document = original_delete

# Run the tests
if __name__ == '__main__':
    # Run synchronous tests
    unittest.main(exit=False)
    
    # Run async tests
    loop = asyncio.get_event_loop()
    
    # TestVectorStore
    vector_store_suite = unittest.TestLoader().loadTestsFromTestCase(TestVectorStore)
    for test in vector_store_suite:
        loop.run_until_complete(test._callTestMethod())
        
    # TestRetentionManager
    retention_suite = unittest.TestLoader().loadTestsFromTestCase(TestRetentionManager)
    for test in retention_suite:
        loop.run_until_complete(test._callTestMethod())
        
    loop.close()
    print("All tests completed.") 