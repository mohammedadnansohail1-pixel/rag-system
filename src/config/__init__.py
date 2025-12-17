"""Configuration module for domain-specific RAG settings."""
from src.config.domain_config import (
    DomainConfig,
    DomainRegistry,
    get_domain_config,
    register_domain,
    FINANCIAL,
    TECHNICAL,
    LEGAL,
    GENERAL,
)

__all__ = [
    "DomainConfig",
    "DomainRegistry", 
    "get_domain_config",
    "register_domain",
    "FINANCIAL",
    "TECHNICAL",
    "LEGAL",
    "GENERAL",
]
