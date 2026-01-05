from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple, List
from .base import BaseReader
from .registry import register_reader


@register_reader([".md", ".markdown"])
class MarkdownReader(BaseReader):
    """Reader implementation for Markdown files."""
    
    def read(self, file_path: Path) -> Tuple[str, str, str]:
        """Read a markdown file and extract document information.
        
        Args:
            file_path (Path): path to the markdown file
            
        Returns:
            Tuple[str, str, str]: (document_name, document_title, content)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title (first line with #)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        return file_path.stem, title, content

    def split(self, content: str) -> List[Tuple[str, str]]:
        """Split content by markdown sections.
        
        Args:
            content (str): the markdown content
            
        Returns:
            List[Tuple[str, str]]: list of (header, section_content) tuples
        """
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_header = ""
        current_content = []
        
        for line in content.split('\n'):
            match = re.match(header_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                # Start new section
                current_header = match.group(2)
                current_content = [line]
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        
        return sections
