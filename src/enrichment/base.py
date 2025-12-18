"""Base class for metadata enrichers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EnrichmentResult:
    """
    Result of enrichment process.
    
    Attributes:
        entities: Extracted entities by type
        topics: Key topics/themes
        summary: Short summary of content
        potential_questions: Questions this content can answer
        keywords: Important keywords
        metadata: Additional metadata
    """
    entities: Dict[str, List[str]] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    potential_questions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {}
        
        if self.entities:
            result["entities"] = self.entities
        if self.topics:
            result["topics"] = self.topics
        if self.summary:
            result["summary"] = self.summary
        if self.potential_questions:
            result["potential_questions"] = self.potential_questions
        if self.keywords:
            result["keywords"] = self.keywords
        if self.metadata:
            result.update(self.metadata)
        
        return result
    
    def merge(self, other: "EnrichmentResult") -> "EnrichmentResult":
        """Merge with another enrichment result."""
        merged_entities = {**self.entities}
        for key, values in other.entities.items():
            if key in merged_entities:
                merged_entities[key] = list(set(merged_entities[key] + values))
            else:
                merged_entities[key] = values
        
        return EnrichmentResult(
            entities=merged_entities,
            topics=list(set(self.topics + other.topics)),
            summary=other.summary or self.summary,
            potential_questions=list(set(self.potential_questions + other.potential_questions)),
            keywords=list(set(self.keywords + other.keywords)),
            metadata={**self.metadata, **other.metadata},
        )


class BaseEnricher(ABC):
    """
    Abstract base class for metadata enrichers.
    
    Implement this to create custom enrichment strategies:
    - Entity extraction
    - Topic modeling
    - LLM-based summarization
    - Question generation
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Enricher identifier."""
        pass
    
    @abstractmethod
    def enrich(self, content: str, metadata: Optional[Dict] = None) -> EnrichmentResult:
        """
        Enrich content with metadata.
        
        Args:
            content: Text content to enrich
            metadata: Existing metadata for context
            
        Returns:
            EnrichmentResult with extracted metadata
        """
        pass
    
    def enrich_batch(
        self, 
        contents: List[str], 
        metadatas: Optional[List[Dict]] = None
    ) -> List[EnrichmentResult]:
        """
        Enrich multiple contents.
        
        Args:
            contents: List of text contents
            metadatas: Optional list of existing metadata
            
        Returns:
            List of EnrichmentResults
        """
        metadatas = metadatas or [None] * len(contents)
        return [
            self.enrich(content, meta) 
            for content, meta in zip(contents, metadatas)
        ]
