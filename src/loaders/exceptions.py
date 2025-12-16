"""Custom exceptions for document loaders."""


class LoaderError(Exception):
    """Raised when a document cannot be loaded or parsed."""
    pass


class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported by any loader."""
    pass
