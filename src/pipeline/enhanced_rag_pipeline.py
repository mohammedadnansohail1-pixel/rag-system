"""Enhanced RAG pipeline with structure-aware chunking, enrichment, and parent-child retrieval."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.loaders.factory import LoaderFactory
from src.chunkers.factory import ChunkerFactory
from src.chunkers import StructureAwareChunker
from src.enrichment import EnrichmentPipeline, EnrichmentConfig
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever
from src.retrieval.parent_child_retriever import ParentChildRetriever
from src.generation.base import BaseLLM
from src.guardrails.config import GuardrailsConfig, PRODUCTION_CONFIG
from src.guardrails.validator import GuardrailsValidator, get_uncertainty_response
from src.guardrails.prompts import (
    SYSTEM_PROMPT_STRICT,
    build_prompt_with_context,
    get_confidence_guidance,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRAGConfig:
    """Configuration for enhanced RAG pipeline."""
    
    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 150
    chunking_strategy: str = "structure_aware"  # or "recursive", "fixed"
    generate_parent_chunks: bool = True
    parent_chunk_size: int = 4000
    
    # Enrichment
    enable_enrichment: bool = True
    enrichment_mode: str = "fast"  # "fast", "full", "minimal"
    
    # Retrieval
    enable_parent_child: bool = True
    parent_weight: float = 0.95
    top_k: int = 10
    
    # Generation
    include_metadata_in_context: bool = True
    max_context_length: int = 8000
    
    @classmethod
    def default(cls) -> "EnhancedRAGConfig":
        """Default production config."""
        return cls()
    
    @classmethod
    def fast(cls) -> "EnhancedRAGConfig":
        """Fast config - minimal processing."""
        return cls(
            enable_enrichment=False,
            enable_parent_child=False,
            generate_parent_chunks=False,
        )
    
    @classmethod
    def full(cls) -> "EnhancedRAGConfig":
        """Full config - all features enabled."""
        return cls(
            enable_enrichment=True,
            enrichment_mode="full",
            enable_parent_child=True,
            generate_parent_chunks=True,
        )


@dataclass
class EnhancedRAGResponse:
    """Enhanced response with full metadata and analytics."""
    
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: str = "low"
    avg_score: float = 0.0
    is_uncertain: bool = False
    validation_passed: bool = True
    rejection_reason: Optional[str] = None
    
    # Enhanced metadata
    sections_used: List[str] = field(default_factory=list)
    topics_found: List[str] = field(default_factory=list)
    entities_found: Dict[str, List[str]] = field(default_factory=dict)
    parent_chunks_used: int = 0
    context_length: int = 0

    @property
    def confidence_emoji(self) -> str:
        return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(self.confidence, "⚪")

    def summary(self) -> str:
        """Get response summary."""
        return (
            f"{self.confidence_emoji} Confidence: {self.confidence}\n"
            f"Sources: {len(self.sources)} | Score: {self.avg_score:.2f}\n"
            f"Sections: {self.sections_used[:3]}\n"
            f"Topics: {self.topics_found[:5]}"
        )


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with all improvements.
    
    Features:
    - Structure-aware chunking (96% reduction)
    - Metadata enrichment (entities, topics)
    - Parent-child retrieval (context expansion)
    - Guardrails and confidence scoring
    - Full observability
    
    Usage:
        pipeline = EnhancedRAGPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
        )
        
        # Ingest with all enhancements
        pipeline.ingest_file("document.pdf")
        
        # Query with rich metadata
        response = pipeline.query("What is the revenue?")
        print(response.summary())
        print(f"Topics: {response.topics_found}")
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        llm: BaseLLM,
        config: Optional[EnhancedRAGConfig] = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            embeddings: Embedding provider
            vectorstore: Vector store
            retriever: Base retriever (will be wrapped if parent-child enabled)
            llm: LLM for generation
            config: Enhanced RAG configuration
            guardrails_config: Guardrails configuration
            system_prompt: Custom system prompt
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config or EnhancedRAGConfig.default()
        self.guardrails = GuardrailsValidator(guardrails_config or PRODUCTION_CONFIG)
        self.system_prompt = system_prompt or SYSTEM_PROMPT_STRICT
        
        # Setup chunker
        if self.config.chunking_strategy == "structure_aware":
            self.chunker = StructureAwareChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                generate_parent_chunks=self.config.generate_parent_chunks,
                parent_chunk_size=self.config.parent_chunk_size,
            )
        else:
            self.chunker = ChunkerFactory.create(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        
        # Setup enrichment pipeline
        if self.config.enable_enrichment:
            if self.config.enrichment_mode == "full":
                enrichment_config = EnrichmentConfig.full()
            elif self.config.enrichment_mode == "minimal":
                enrichment_config = EnrichmentConfig.minimal()
            else:
                enrichment_config = EnrichmentConfig.fast()
            self.enrichment = EnrichmentPipeline(config=enrichment_config, llm=llm if self.config.enrichment_mode == "full" else None)
        else:
            self.enrichment = None
        
        # Setup retriever (wrap with parent-child if enabled)
        if self.config.enable_parent_child:
            self.retriever = ParentChildRetriever(
                base_retriever=retriever,
                include_parents=True,
                parent_weight=self.config.parent_weight,
                replace_children_with_parents=True,
            )
        else:
            self.retriever = retriever
        
        # Store base retriever reference for add_documents
        self._base_retriever = retriever
        
        # Stats
        self._ingestion_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "parent_chunks": 0,
            "enriched_chunks": 0,
        }
        
        logger.info(
            f"Initialized EnhancedRAGPipeline: "
            f"chunking={self.config.chunking_strategy}, "
            f"enrichment={self.config.enable_enrichment}, "
            f"parent_child={self.config.enable_parent_child}"
        )
    
    def ingest_file(self, file_path: str) -> Dict[str, int]:
        """
        Ingest a file with structure-aware chunking and enrichment.
        
        Returns:
            Stats about ingestion
        """
        path = Path(file_path)
        documents = LoaderFactory.load(str(path))
        
        return self._ingest_documents(documents, source=str(path))
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Ingest all documents from a directory."""
        documents = LoaderFactory.load_directory(
            directory,
            recursive=recursive,
            file_types=file_types
        )
        
        return self._ingest_documents(documents, source=directory)
    
    def _ingest_documents(self, documents, source: str) -> Dict[str, int]:
        """Internal method to ingest documents."""
        stats = {
            "documents": len(documents),
            "chunks": 0,
            "parent_chunks": 0,
            "sections": set(),
        }
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning(f"No chunks created from {source}")
            return stats
        
        stats["chunks"] = len(all_chunks)
        
        # Enrich chunks
        if self.enrichment:
            all_chunks = self.enrichment.enrich_chunks(all_chunks, show_progress=True)
            stats["enriched"] = len(all_chunks)
        
        # Count parents and sections
        for chunk in all_chunks:
            if chunk.chunk_type == "parent":
                stats["parent_chunks"] += 1
            if chunk.section:
                stats["sections"].add(chunk.section[:30])
        
        stats["sections"] = list(stats["sections"])[:10]
        
        # Prepare for ingestion
        texts = [c.content for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        
        # Add to retriever
        self.retriever.add_documents(texts=texts, metadatas=metadatas)
        
        # Update global stats
        self._ingestion_stats["total_documents"] += stats["documents"]
        self._ingestion_stats["total_chunks"] += stats["chunks"]
        self._ingestion_stats["parent_chunks"] += stats["parent_chunks"]
        
        logger.info(
            f"Ingested {stats['chunks']} chunks from {source} "
            f"(parents={stats['parent_chunks']}, sections={len(stats['sections'])})"
        )
        
        return stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_section: Optional[str] = None,
        filter_topics: Optional[List[str]] = None,
    ) -> EnhancedRAGResponse:
        """
        Query with enhanced retrieval and metadata.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            filter_section: Only use chunks from this section
            filter_topics: Only use chunks with these topics
            
        Returns:
            EnhancedRAGResponse with answer and rich metadata
        """
        top_k = top_k or self.config.top_k
        
        # Step 1: Retrieve
        results = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Apply metadata filters
        if filter_section or filter_topics:
            results = self._filter_results(results, filter_section, filter_topics)
        
        # Step 3: Validate through guardrails
        validation = self.guardrails.validate(results)
        
        # Step 4: Handle validation failure
        if not validation.is_valid:
            logger.warning(f"Guardrails rejected query: {validation.rejection_reason}")
            return EnhancedRAGResponse(
                answer=get_uncertainty_response(validation.rejection_reason or ""),
                sources=[],
                query=question,
                confidence="low",
                avg_score=validation.avg_score,
                is_uncertain=True,
                validation_passed=False,
                rejection_reason=validation.rejection_reason,
            )
        
        filtered_results = validation.filtered_results
        
        # Step 5: Extract metadata from results
        sections_used = []
        topics_found = []
        entities_found = {}
        parent_chunks_used = 0
        
        for r in filtered_results:
            meta = r.metadata
            
            if meta.get("section"):
                sections_used.append(meta["section"])
            
            if meta.get("topics"):
                topics_found.extend(meta["topics"])
            
            if meta.get("entities"):
                for etype, values in meta["entities"].items():
                    if etype not in entities_found:
                        entities_found[etype] = []
                    entities_found[etype].extend(values)
            
            if meta.get("retrieved_as") == "parent_of_match":
                parent_chunks_used += 1
        
        # Deduplicate
        sections_used = list(dict.fromkeys(sections_used))
        topics_found = list(dict.fromkeys(topics_found))
        for etype in entities_found:
            entities_found[etype] = list(dict.fromkeys(entities_found[etype]))[:5]
        
        # Step 6: Build context
        context = self._build_context(filtered_results)
        
        # Step 7: Build prompt
        prompt = build_prompt_with_context(
            query=question,
            context_chunks=filtered_results,
            confidence=validation.confidence,
            include_scores=True
        )
        
        # Step 8: Add section/topic context to prompt
        if self.config.include_metadata_in_context and sections_used:
            prompt = f"[Sections: {', '.join(sections_used[:3])}]\n\n" + prompt
        
        # Step 9: Generate answer
        confidence_guidance = get_confidence_guidance(validation.confidence)
        enhanced_system_prompt = f"{self.system_prompt}\n\nCONFIDENCE LEVEL: {validation.confidence.upper()}\n{confidence_guidance}"
        
        answer = self.llm.generate(prompt, system_prompt=enhanced_system_prompt)
        
        # Step 10: Build response
        sources = [
            {
                "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                "metadata": r.metadata,
                "score": r.score
            }
            for r in filtered_results
        ]
        
        logger.info(
            f"Generated response: confidence={validation.confidence}, "
            f"sources={len(sources)}, sections={len(sections_used)}, "
            f"topics={len(topics_found)}, parents_used={parent_chunks_used}"
        )
        
        return EnhancedRAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            confidence=validation.confidence,
            avg_score=validation.avg_score,
            is_uncertain=False,
            validation_passed=True,
            sections_used=sections_used,
            topics_found=topics_found[:10],
            entities_found=entities_found,
            parent_chunks_used=parent_chunks_used,
            context_length=len(context),
        )
    
    def _filter_results(
        self,
        results,
        filter_section: Optional[str],
        filter_topics: Optional[List[str]],
    ):
        """Filter results by section or topics."""
        filtered = results
        
        if filter_section:
            filtered = [
                r for r in filtered
                if filter_section.lower() in (r.metadata.get("section", "") or "").lower()
            ]
        
        if filter_topics:
            filtered = [
                r for r in filtered
                if any(t in r.metadata.get("topics", []) for t in filter_topics)
            ]
        
        return filtered if filtered else results  # Return original if filter removes all
    
    def _build_context(self, results) -> str:
        """Build context string from results."""
        context_parts = []
        total_length = 0
        
        for r in results:
            if total_length + len(r.content) > self.config.max_context_length:
                break
            context_parts.append(r.content)
            total_length += len(r.content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._ingestion_stats,
            "config": {
                "chunking_strategy": self.config.chunking_strategy,
                "enrichment_enabled": self.config.enable_enrichment,
                "parent_child_enabled": self.config.enable_parent_child,
            }
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "vectorstore": self.vectorstore.health_check(),
            "llm": self.llm.health_check(),
            "retriever": self.retriever.health_check(),
        }
