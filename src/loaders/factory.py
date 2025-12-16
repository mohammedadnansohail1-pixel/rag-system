"""Factory for creating document loaders with registry pattern."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Callable

from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import UnsupportedFileTypeError

logger = logging.getLogger(__name__)

# Registry to hold loader classes
_LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {}


def register_loader(name: str) -> Callable:
    """
    Decorator to register a loader class.
    
    Usage:
        @register_loader("pdf")
        class PDFLoader(BaseLoader):
            ...
    """
    def decorator(cls: Type[BaseLoader]) -> Type[BaseLoader]:
        if name in _LOADER_REGISTRY:
            logger.warning(f"Overwriting existing loader: {name}")
        _LOADER_REGISTRY[name] = cls
        logger.debug(f"Registered loader: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_registered_loaders() -> List[str]:
    """Return list of registered loader names."""
    return list(_LOADER_REGISTRY.keys())


class LoaderFactory:
    """
    Factory that creates appropriate loader based on file type.
    
    Usage:
        factory = LoaderFactory()
        docs = factory.load(Path("document.pdf"))
        
        # Or load entire directory
        docs = factory.load_directory(Path("./data/raw"))
    """

    def __init__(self):
        """Initialize with registered loaders."""
        self._loaders: List[BaseLoader] = []
        
        # Instantiate all registered loaders
        for name, loader_class in _LOADER_REGISTRY.items():
            self._loaders.append(loader_class())
            logger.debug(f"Initialized loader: {name}")

    def register_loader(self, loader: BaseLoader) -> None:
        """Add a custom loader instance at runtime."""
        self._loaders.append(loader)
        logger.info(f"Registered runtime loader: {loader.__class__.__name__}")

    def get_loader(self, file_path: Path) -> BaseLoader:
        """
        Get appropriate loader for file type.
        
        Args:
            file_path: Path to file
            
        Returns:
            Loader that can handle this file
            
        Raises:
            UnsupportedFileTypeError: If no loader found
        """
        for loader in self._loaders:
            if loader.can_load(file_path):
                return loader
        
        raise UnsupportedFileTypeError(
            f"No loader found for file type: {file_path.suffix}"
        )

    def load(self, file_path: Path) -> List[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Documents
        """
        file_path = Path(file_path)
        loader = self.get_loader(file_path)
        logger.info(f"Loading {file_path} with {loader.__class__.__name__}")
        return loader.load(file_path)

    def load_directory(
        self, 
        directory: Path, 
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported files from directory.
        
        Args:
            directory: Path to directory
            recursive: Search subdirectories
            file_types: Filter by extensions (e.g., ['.pdf', '.md'])
            
        Returns:
            List of all Documents
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            
            # Filter by file types if specified
            if file_types and file_path.suffix.lower() not in file_types:
                continue
            
            try:
                loader = self.get_loader(file_path)
                docs = loader.load(file_path)
                documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} docs from {file_path}")
            except UnsupportedFileTypeError:
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
