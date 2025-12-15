"""Abstract base class for secret backends."""

from abc import ABC, abstractmethod


class SecretBackend(ABC):
    """
    Abstract base class that all secret backends must implement.
    
    Ensures consistent interface across:
    - Environment variables
    - File-based secrets
    - HashiCorp Vault
    - AWS Secrets Manager
    """

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """
        Retrieve a secret by key.
        
        Args:
            key: The secret key name (exact match)
            
        Returns:
            The secret value as string
            
        Raises:
            SecretNotFoundError: If secret doesn't exist
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if backend is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
