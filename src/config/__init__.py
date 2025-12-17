"""Configuration module for domain-specific RAG settings."""
from src.config.domain_config import (
    DomainConfig,
    DomainRegistry,
    EnrichmentConfig,
    get_domain_config,
    register_domain,
    FINANCIAL,
    TECHNICAL,
    LEGAL,
    GENERAL,
)
from src.config.chunk_enricher import ChunkEnricher

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
