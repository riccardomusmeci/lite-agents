from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List


class BaseReader(ABC):
    """Abstract base class for document readers."""
    
    @abstractmethod
    def read(self, file_path: Path) -> Tuple[str, str, str]:
        """Read a file and return document metadata and content.
        
        Args:
            file_path (Path): path to the file
            
        Returns:
            Tuple[str, str, str]: (document_name, document_title, content)
        """
        pass

    @abstractmethod
    def split(self, content: str) -> List[Tuple[str, str]]:
        """Split content into sections.
        
        Args:
            content (str): the content to split
            
        Returns:
            List[Tuple[str, str]]: list of (header, section_content) tuples
        """
        pass
