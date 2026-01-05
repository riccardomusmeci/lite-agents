from __future__ import annotations
from typing import Dict, Type, List, Callable
from pathlib import Path
from .base import BaseReader

_READERS_REGISTRY: Dict[str, Type[BaseReader]] = {}


def register_reader(extensions: List[str]) -> Callable[[Type[BaseReader]], Type[BaseReader]]:
    """Decorator to register a reader class for specific file extensions.
    
    Args:
        extensions (List[str]): list of file extensions (e.g. ['.md', '.markdown'])
        
    Returns:
        Callable[[Type[BaseReader]], Type[BaseReader]]: the decorated class
    """
    def decorator(cls: Type[BaseReader]) -> Type[BaseReader]:
        for ext in extensions:
            _READERS_REGISTRY[ext] = cls
        return cls
    return decorator


def get_reader_for_file(file_path: Path) -> BaseReader:
    """Get the appropriate reader instance for a file based on extension.
    
    Args:
        file_path (Path): path to the file
        
    Returns:
        BaseReader: an instance of the appropriate reader
        
    Raises:
        ValueError: if no reader is registered for the file extension
    """
    ext = file_path.suffix.lower()
    reader_cls = _READERS_REGISTRY.get(ext)
    if not reader_cls:
        raise ValueError(f"No reader registered for extension: {ext}")
    return reader_cls()
