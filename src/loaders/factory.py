"""Factory for creating document loaders with registry pattern."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Callable

from src.loaders.base import BaseLoader, Document
from src.loaders.exceptions import UnsupportedFileTypeError

logger = logging.getLogger(__name__)

# Registry to hold loader classes
_LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {}

# Cache for loader instances
_LOADER_INSTANCES: Dict[str, BaseLoader] = {}


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


def _get_loader_instances() -> List[BaseLoader]:
    """Get or create loader instances from registry."""
    for name, loader_class in _LOADER_REGISTRY.items():
        if name not in _LOADER_INSTANCES:
            _LOADER_INSTANCES[name] = loader_class()
    return list(_LOADER_INSTANCES.values())


class LoaderFactory:
    """
    Factory that creates appropriate loader based on file type.

    Usage:
        docs = LoaderFactory.load(Path("document.pdf"))

        # Or load entire directory
        docs = LoaderFactory.load_directory(Path("./data/raw"))
        
        # Load with section extraction (for SEC filings)
        docs = LoaderFactory.load(Path("10k.txt"), use_sections=True)
        
        # Auto-detect: SEC filings automatically use sections
        docs = LoaderFactory.load(Path("10k.txt"), use_sections="auto")
    """

    # Loaders that support and should default to section extraction
    SECTION_LOADERS = {'SECLoader'}

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        """
        Get appropriate loader for file type.

        Args:
            file_path: Path to file

        Returns:
            Loader that can handle this file

        Raises:
            UnsupportedFileTypeError: If no loader found
        """
        file_path = Path(file_path)
        loaders = _get_loader_instances()

        for loader in loaders:
            if loader.can_load(file_path):
                return loader

        raise UnsupportedFileTypeError(
            f"No loader found for file type: {file_path.suffix}"
        )

    @classmethod
    def load(
        cls, 
        file_path: Path, 
        use_sections: bool | str = "auto",
    ) -> List[Document]:
        """
        Load a single file.

        Args:
            file_path: Path to file
            use_sections: 
                - True: Always extract sections
                - False: Never extract sections
                - "auto": Extract sections for supported loaders (default)

        Returns:
            List of Documents
        """
        file_path = Path(file_path)
        loader = cls.get_loader(file_path)
        loader_name = loader.__class__.__name__
        
        # Determine if we should use sections
        should_use_sections = False
        if use_sections == "auto":
            should_use_sections = (
                loader_name in cls.SECTION_LOADERS 
                and hasattr(loader, 'load_with_sections')
            )
        elif use_sections:
            should_use_sections = hasattr(loader, 'load_with_sections')
        
        if should_use_sections:
            logger.info(f"Loading {file_path} with {loader_name} (section extraction)")
            return loader.load_with_sections(str(file_path))
        
        logger.info(f"Loading {file_path} with {loader_name}")
        return loader.load(file_path)

    @classmethod
    def load_directory(
        cls,
        directory: Path,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        use_sections: bool | str = "auto",
    ) -> List[Document]:
        """
        Load all supported files from directory.

        Args:
            directory: Path to directory
            recursive: Search subdirectories
            file_types: Filter by extensions (e.g., ['.pdf', '.md'])
            use_sections:
                - True: Always extract sections
                - False: Never extract sections  
                - "auto": Extract sections for supported loaders (default)

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
                docs = cls.load(file_path, use_sections=use_sections)
                documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} docs from {file_path}")
            except UnsupportedFileTypeError:
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
