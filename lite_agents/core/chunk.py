from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class DocumentChunk:
    """Represents a document chunk with its contextual information.
    
    Args:
        content (str): the chunk content
        document_name (str): the document filename (without extension)
        chunk_index (int): the index of the chunk in the document
        total_chunks (int): the total number of chunks in the document
        context (str | None): the generated context for the chunk
        document_title (str | None, optional): the document title. Defaults to None.
        document_summary (str | None, optional): the document summary. Defaults to None.
        section_header (str | None, optional): the section header. Defaults to None.
    """
    content: str
    document_name: str
    chunk_index: int
    total_chunks: int
    context: str | None = None
    section_header: str | None = None
    document_title: str | None = None
    document_summary: str | None = None
    
    def get_contextualized_content(self) -> str:
        """Return the content with context prepended.
        
        Returns:
            str: the contextualized content
        """
        if self.context:
            return f"{self.context}\n\n{self.content}"
        return self.content
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dictionary for vector database.
        
        Returns:
            Dict[str, Any]: the metadata dictionary
        """
        metadata = {
            "document_name": self.document_name,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }
        
        if self.context:
            metadata["context"] = self.context
        
        if self.document_title:
            metadata["document_title"] = self.document_title
            
        if self.document_summary:
            metadata["document_summary"] = self.document_summary
        
        if self.section_header:
            metadata["section_header"] = self.section_header
        
        return metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary.
        
        Returns:
            Dict[str, Any]: the dictionary representation
        """
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DocumentChunk:
        """Create DocumentChunk from dictionary.
        
        Args:
            data (Dict[str, Any]): the dictionary data
            
        Returns:
            DocumentChunk: the DocumentChunk instance
        """
        return DocumentChunk(**data)
