"""Enrichment pipeline combining multiple enrichers."""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from src.enrichment.base import BaseEnricher, EnrichmentResult
from src.enrichment.entity_extractor import EntityExtractor
from src.enrichment.topic_extractor import TopicExtractor
from src.enrichment.llm_enricher import LLMEnricher
from src.chunkers.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """Configuration for enrichment pipeline."""
    
    # Entity extraction
    extract_entities: bool = True
    extract_money: bool = True
    extract_percentages: bool = True
    extract_dates: bool = True
    extract_organizations: bool = True
    
    # Topic extraction
    extract_topics: bool = True
    max_topics: int = 5
    max_keywords: int = 10
    
    # LLM enrichment (optional, slower)
    use_llm: bool = False
    generate_summary: bool = True
    generate_questions: bool = False
    
    @classmethod
    def fast(cls) -> "EnrichmentConfig":
        """Fast config: entities + topics only, no LLM."""
        return cls(
            extract_entities=True,
            extract_topics=True,
            use_llm=False,
        )
    
    @classmethod
    def full(cls) -> "EnrichmentConfig":
        """Full config: all enrichment including LLM."""
        return cls(
            extract_entities=True,
            extract_topics=True,
            use_llm=True,
            generate_summary=True,
            generate_questions=True,
        )
    
    @classmethod
    def minimal(cls) -> "EnrichmentConfig":
        """Minimal config: only essential entities."""
        return cls(
            extract_entities=True,
            extract_money=True,
            extract_percentages=True,
            extract_dates=False,
            extract_organizations=False,
            extract_topics=False,
            use_llm=False,
        )


class EnrichmentPipeline:
    """
    Pipeline that combines multiple enrichers.
    
    Configurable to use:
    - Fast extraction (entities + topics) - default
    - Full enrichment (+ LLM summaries/questions) - optional
    
    Usage:
        # Fast mode (default)
        pipeline = EnrichmentPipeline()
        enriched_chunks = pipeline.enrich_chunks(chunks)
        
        # Full mode with LLM
        from src.generation.ollama_llm import OllamaLLM
        
        pipeline = EnrichmentPipeline(
            config=EnrichmentConfig.full(),
            llm=OllamaLLM(model="llama3.2")
        )
        enriched_chunks = pipeline.enrich_chunks(chunks)
        
        # From YAML config
        pipeline = EnrichmentPipeline.from_config({
            "extract_entities": True,
            "extract_topics": True,
            "use_llm": False,
        })
    """
    
    def __init__(
        self,
        config: Optional[EnrichmentConfig] = None,
        llm = None,
        custom_entities: Optional[Dict[str, set]] = None,
        custom_topics: Optional[Dict[str, set]] = None,
    ):
        """
        Args:
            config: Enrichment configuration
            llm: LLM instance for summaries/questions (required if use_llm=True)
            custom_entities: Additional entity sets to extract
            custom_topics: Additional topic sets to detect
        """
        self.config = config or EnrichmentConfig.fast()
        self.enrichers: List[BaseEnricher] = []
        
        # Add entity extractor
        if self.config.extract_entities:
            self.enrichers.append(EntityExtractor(
                extract_money=self.config.extract_money,
                extract_percentages=self.config.extract_percentages,
                extract_dates=self.config.extract_dates,
                extract_organizations=self.config.extract_organizations,
                custom_entities=custom_entities,
            ))
        
        # Add topic extractor
        if self.config.extract_topics:
            self.enrichers.append(TopicExtractor(
                max_topics=self.config.max_topics,
                max_keywords=self.config.max_keywords,
                custom_topics=custom_topics,
            ))
        
        # Add LLM enricher
        if self.config.use_llm:
            if llm is None:
                raise ValueError("LLM required when use_llm=True")
            self.enrichers.append(LLMEnricher(
                llm=llm,
                generate_summary=self.config.generate_summary,
                generate_questions=self.config.generate_questions,
            ))
        
        logger.info(
            f"Initialized EnrichmentPipeline with {len(self.enrichers)} enrichers: "
            f"{[e.name for e in self.enrichers]}"
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], llm=None) -> "EnrichmentPipeline":
        """Create pipeline from config dict."""
        enrichment_config = EnrichmentConfig(
            extract_entities=config.get("extract_entities", True),
            extract_money=config.get("extract_money", True),
            extract_percentages=config.get("extract_percentages", True),
            extract_dates=config.get("extract_dates", True),
            extract_organizations=config.get("extract_organizations", True),
            extract_topics=config.get("extract_topics", True),
            max_topics=config.get("max_topics", 5),
            max_keywords=config.get("max_keywords", 10),
            use_llm=config.get("use_llm", False),
            generate_summary=config.get("generate_summary", True),
            generate_questions=config.get("generate_questions", False),
        )
        return cls(config=enrichment_config, llm=llm)
    
    def enrich(self, content: str, metadata: Optional[Dict] = None) -> EnrichmentResult:
        """
        Enrich a single piece of content.
        
        Args:
            content: Text content to enrich
            metadata: Existing metadata for context
            
        Returns:
            Combined EnrichmentResult from all enrichers
        """
        combined = EnrichmentResult()
        
        for enricher in self.enrichers:
            try:
                result = enricher.enrich(content, metadata)
                combined = combined.merge(result)
            except Exception as e:
                logger.warning(f"Enricher {enricher.name} failed: {e}")
        
        return combined
    
    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """
        Enrich a single chunk, updating its metadata.
        
        Args:
            chunk: Chunk to enrich
            
        Returns:
            Chunk with enriched metadata
        """
        result = self.enrich(chunk.content, chunk.metadata)
        
        # Start with existing metadata
        enriched_metadata = {**chunk.metadata}
        
        # Add section info to metadata (for retrieval filtering)
        if chunk.section:
            enriched_metadata['section'] = chunk.section
        if chunk.section_hierarchy:
            enriched_metadata['section_hierarchy'] = chunk.section_hierarchy
        if chunk.chunk_type:
            enriched_metadata['chunk_type'] = chunk.chunk_type
        if chunk.chunk_id:
            enriched_metadata['chunk_id'] = chunk.chunk_id
        if chunk.parent_id:
            enriched_metadata['parent_id'] = chunk.parent_id
        
        # Merge enrichment results
        enriched_metadata.update(result.to_dict())
        
        # Create new chunk with enriched metadata
        return Chunk(
            content=chunk.content,
            metadata=enriched_metadata,
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            chunk_type=chunk.chunk_type,
            section=chunk.section,
            section_hierarchy=chunk.section_hierarchy,
        )
    
    def enrich_chunks(
        self, 
        chunks: List[Chunk],
        show_progress: bool = True,
    ) -> List[Chunk]:
        """
        Enrich multiple chunks.
        
        Args:
            chunks: List of chunks to enrich
            show_progress: Log progress updates
            
        Returns:
            List of enriched chunks
        """
        enriched = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 50 == 0:
                logger.info(f"Enrichment progress: {i + 1}/{total}")
            
            enriched_chunk = self.enrich_chunk(chunk)
            enriched.append(enriched_chunk)
        
        if show_progress:
            logger.info(f"Enrichment complete: {total} chunks processed")
        
        return enriched
    
    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about enriched chunks."""
        stats = {
            "total_chunks": len(chunks),
            "chunks_with_entities": 0,
            "chunks_with_topics": 0,
            "chunks_with_summary": 0,
            "chunks_with_section": 0,
            "entity_types": set(),
            "all_topics": set(),
            "sections": set(),
        }
        
        for chunk in chunks:
            meta = chunk.metadata
            
            if meta.get("entities"):
                stats["chunks_with_entities"] += 1
                stats["entity_types"].update(meta["entities"].keys())
            
            if meta.get("topics"):
                stats["chunks_with_topics"] += 1
                stats["all_topics"].update(meta["topics"])
            
            if meta.get("summary"):
                stats["chunks_with_summary"] += 1
            
            if meta.get("section"):
                stats["chunks_with_section"] += 1
                stats["sections"].add(meta["section"][:30])
        
        # Convert sets to lists for JSON serialization
        stats["entity_types"] = list(stats["entity_types"])
        stats["all_topics"] = list(stats["all_topics"])[:20]
        stats["sections"] = list(stats["sections"])[:15]
        
        return stats
