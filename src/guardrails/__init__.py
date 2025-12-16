"""Guardrails module - production-ready safety for RAG."""

from src.guardrails.config import (
    GuardrailsConfig,
    PRODUCTION_CONFIG,
    PERMISSIVE_CONFIG,
)
from src.guardrails.validator import (
    GuardrailsValidator,
    ValidationResult,
    get_uncertainty_response,
)
from src.guardrails.prompts import (
    SYSTEM_PROMPT_STRICT,
    SYSTEM_PROMPT_BALANCED,
    SYSTEM_PROMPT_WITH_CONFIDENCE,
    get_confidence_guidance,
    build_prompt_with_context,
)

__all__ = [
    "GuardrailsConfig",
    "PRODUCTION_CONFIG",
    "PERMISSIVE_CONFIG",
    "GuardrailsValidator",
    "ValidationResult",
    "get_uncertainty_response",
    "SYSTEM_PROMPT_STRICT",
    "SYSTEM_PROMPT_BALANCED",
    "SYSTEM_PROMPT_WITH_CONFIDENCE",
    "get_confidence_guidance",
    "build_prompt_with_context",
]
