"""Enhanced RAG pipeline with structure-aware chunking, enrichment, parent-child, and hierarchical summaries."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.config import Config
from src.loaders.factory import LoaderFactory
from src.chunkers.factory import ChunkerFactory
from src.chunkers import StructureAwareChunker
from src.enrichment import EnrichmentPipeline, EnrichmentConfig
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever
from src.retrieval.parent_child_retriever import ParentChildRetriever
from src.summarization import SectionSummarizer, HierarchicalRetriever
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
    """
    Configuration for enhanced RAG pipeline.
    
    Loads from the 'enhanced' section of config/rag.yaml.
    
    Usage:
        # From existing Config system
        config = EnhancedRAGConfig.from_config()
        
        # From dict (for testing or custom configs)
        config = EnhancedRAGConfig.from_dict({...})
    """
    
    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 150
    chunking_strategy: str = "structure_aware"
    generate_parent_chunks: bool = True
    parent_chunk_size: int = 4000
    
    # Enrichment
    enable_enrichment: bool = True
    enrichment_mode: str = "fast"
    
    # Summarization
    enable_summarization: bool = False
    min_chunks_for_summary: int = 5
    summary_sections: Optional[List[str]] = None
    summary_boost: float = 1.5
    
    # Retrieval
    enable_parent_child: bool = True
    enable_hierarchical: bool = False
    parent_weight: float = 0.95
    top_k: int = 10
    
    # Generation
    include_metadata_in_context: bool = True
    max_context_length: int = 8000
    
    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> "EnhancedRAGConfig":
        """
        Load from the existing Config system (config/rag.yaml).
        
        Args:
            config: Existing Config instance, or loads default if None
        """
        if config is None:
            config = Config.load()
        
        enhanced = config.get_section("enhanced")
        if not enhanced:
            logger.warning("No 'enhanced' section in config, using defaults")
            return cls()
        
        chunking = enhanced.get("chunking", {})
        enrichment = enhanced.get("enrichment", {})
        summarization = enhanced.get("summarization", {})
        parent_child = enhanced.get("parent_child", {})
        hierarchical = enhanced.get("hierarchical", {})
        retrieval = config.get_section("retrieval")
        
        return cls(
            # Chunking
            chunk_size=chunking.get("chunk_size", 1500),
            chunk_overlap=chunking.get("chunk_overlap", 150),
            chunking_strategy=chunking.get("strategy", "structure_aware"),
            generate_parent_chunks=chunking.get("generate_parent_chunks", True),
            parent_chunk_size=chunking.get("parent_chunk_size", 4000),
            
            # Enrichment
            enable_enrichment=enrichment.get("enabled", True),
            enrichment_mode=enrichment.get("mode", "fast"),
            
            # Summarization
            enable_summarization=summarization.get("enabled", False),
            min_chunks_for_summary=summarization.get("min_chunks_for_summary", 5),
            summary_sections=summarization.get("sections"),
            summary_boost=summarization.get("summary_boost", 1.5),
            
            # Retrieval
            enable_parent_child=parent_child.get("enabled", True),
            parent_weight=parent_child.get("parent_weight", 0.95),
            enable_hierarchical=hierarchical.get("enabled", False),
            top_k=retrieval.get("retrieval_top_k", 10) if retrieval else 10,
        )
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EnhancedRAGConfig":
        """Create configuration from dictionary (for testing or custom configs)."""
        chunking = config.get("chunking", {})
        enrichment = config.get("enrichment", {})
        summarization = config.get("summarization", {})
        parent_child = config.get("parent_child", {})
        hierarchical = config.get("hierarchical", {})
        retrieval = config.get("retrieval", {})
        generation = config.get("generation", {})
        
        return cls(
            # Chunking
            chunk_size=chunking.get("chunk_size", 1500),
            chunk_overlap=chunking.get("chunk_overlap", 150),
            chunking_strategy=chunking.get("strategy", "structure_aware"),
            generate_parent_chunks=chunking.get("generate_parent_chunks", True),
            parent_chunk_size=chunking.get("parent_chunk_size", 4000),
            
            # Enrichment
            enable_enrichment=enrichment.get("enabled", True),
            enrichment_mode=enrichment.get("mode", "fast"),
            
            # Summarization
            enable_summarization=summarization.get("enabled", False),
            min_chunks_for_summary=summarization.get("min_chunks_for_summary", 5),
            summary_sections=summarization.get("sections"),
            summary_boost=summarization.get("summary_boost", 1.5),
            
            # Retrieval
            enable_parent_child=parent_child.get("enabled", True),
            parent_weight=parent_child.get("parent_weight", 0.95),
            enable_hierarchical=hierarchical.get("enabled", False),
            top_k=retrieval.get("top_k", 10),
            
            # Generation
            include_metadata_in_context=generation.get("include_metadata_in_context", True),
            max_context_length=generation.get("max_context_length", 8000),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "chunking": {
                "strategy": self.chunking_strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "generate_parent_chunks": self.generate_parent_chunks,
                "parent_chunk_size": self.parent_chunk_size,
            },
            "enrichment": {
                "enabled": self.enable_enrichment,
                "mode": self.enrichment_mode,
            },
            "summarization": {
                "enabled": self.enable_summarization,
                "min_chunks_for_summary": self.min_chunks_for_summary,
                "sections": self.summary_sections,
                "summary_boost": self.summary_boost,
            },
            "parent_child": {
                "enabled": self.enable_parent_child,
                "parent_weight": self.parent_weight,
            },
            "hierarchical": {
                "enabled": self.enable_hierarchical,
            },
            "retrieval": {
                "top_k": self.top_k,
            },
            "generation": {
                "include_metadata_in_context": self.include_metadata_in_context,
                "max_context_length": self.max_context_length,
            },
        }


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
    summaries_used: int = 0
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
            f"Topics: {self.topics_found[:5]}\n"
            f"Summaries used: {self.summaries_used}"
        )


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with all improvements.
    
    Features:
    - Structure-aware chunking (Phase 1)
    - Metadata enrichment (Phase 2)
    - Parent-child retrieval (Phase 3)
    - Hierarchical summaries (Phase 4)
    - Guardrails and confidence scoring
    
    Usage:
        # Load config from rag.yaml
        config = EnhancedRAGConfig.from_config()
        
        pipeline = EnhancedRAGPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            config=config,
        )
        
        # Ingest with all enhancements
        pipeline.ingest_file("document.pdf")
        
        # Query with rich metadata
        response = pipeline.query("Summarize the risk factors")
        print(response.summary())
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
            retriever: Base retriever (will be wrapped based on config)
            llm: LLM for generation
            config: Enhanced RAG configuration (loads from rag.yaml if None)
            guardrails_config: Guardrails configuration
            system_prompt: Custom system prompt
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config or EnhancedRAGConfig.from_config()
        self.guardrails = GuardrailsValidator(guardrails_config or PRODUCTION_CONFIG)
        self.system_prompt = system_prompt or SYSTEM_PROMPT_STRICT
        
        # Store base retriever reference
        self._base_retriever = retriever
        
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
            self.enrichment = EnrichmentPipeline(
                config=enrichment_config, 
                llm=llm if self.config.enrichment_mode == "full" else None
            )
        else:
            self.enrichment = None
        
        # Setup summarizer
        if self.config.enable_summarization:
            self.summarizer = SectionSummarizer(
                llm=llm,
                min_chunks_for_summary=self.config.min_chunks_for_summary,
            )
        else:
            self.summarizer = None
        
        # Setup retriever chain
        self.retriever = self._setup_retriever(retriever)
        
        # Stats
        self._ingestion_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "parent_chunks": 0,
            "summary_chunks": 0,
            "enriched_chunks": 0,
        }
        
        logger.info(
            f"Initialized EnhancedRAGPipeline: "
            f"chunking={self.config.chunking_strategy}, "
            f"enrichment={self.config.enable_enrichment}, "
            f"parent_child={self.config.enable_parent_child}, "
            f"summarization={self.config.enable_summarization}, "
            f"hierarchical={self.config.enable_hierarchical}"
        )
    
    def _setup_retriever(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """Setup retriever chain based on config."""
        retriever = base_retriever
        
        # Wrap with parent-child if enabled
        if self.config.enable_parent_child:
            retriever = ParentChildRetriever(
                base_retriever=retriever,
                include_parents=True,
                parent_weight=self.config.parent_weight,
                replace_children_with_parents=True,
            )
        
        # Wrap with hierarchical if enabled
        if self.config.enable_hierarchical:
            retriever = HierarchicalRetriever(
                base_retriever=retriever,
                summary_boost=self.config.summary_boost,
                max_summaries=2,
            )
        
        return retriever
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a file with all enhancements.
        
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
    ) -> Dict[str, Any]:
        """Ingest all documents from a directory."""
        documents = LoaderFactory.load_directory(
            directory,
            recursive=recursive,
            file_types=file_types
        )
        
        return self._ingest_documents(documents, source=directory)
    
    def _ingest_documents(self, documents, source: str) -> Dict[str, Any]:
        """Internal method to ingest documents with all enhancements."""
        stats = {
            "documents": len(documents),
            "chunks": 0,
            "parent_chunks": 0,
            "summary_chunks": 0,
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
        
        # Generate summaries if enabled
        summary_chunks = []
        if self.summarizer:
            logger.info("Generating section summaries...")
            summaries = self.summarizer.summarize_sections(
                all_chunks,
                sections_to_summarize=self.config.summary_sections,
            )
            summary_chunks = [s.to_chunk() for s in summaries]
            stats["summary_chunks"] = len(summary_chunks)
            logger.info(f"Generated {len(summary_chunks)} section summaries")
        
        # Combine all chunks
        all_chunks = all_chunks + summary_chunks
        
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
        self._ingestion_stats["summary_chunks"] += stats["summary_chunks"]
        
        logger.info(
            f"Ingested {stats['chunks']} chunks + {stats['summary_chunks']} summaries from {source}"
        )
        
        return stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_section: Optional[str] = None,
        filter_topics: Optional[List[str]] = None,
        prefer_summaries: Optional[bool] = None,
    ) -> EnhancedRAGResponse:
        """
        Query with enhanced retrieval and metadata.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            filter_section: Only use chunks from this section
            filter_topics: Only use chunks with these topics
            prefer_summaries: Force summary preference (auto-detected if None)
            
        Returns:
            EnhancedRAGResponse with answer and rich metadata
        """
        top_k = top_k or self.config.top_k
        
        # Step 1: Retrieve (with optional summary preference)
        if isinstance(self.retriever, HierarchicalRetriever) and prefer_summaries is not None:
            results = self.retriever.retrieve(
                question, 
                top_k=top_k,
                prefer_summaries=prefer_summaries,
            )
        else:
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
        summaries_used = 0
        
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
            
            if meta.get("is_summary"):
                summaries_used += 1
        
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
                "score": r.score,
                "is_summary": r.metadata.get("is_summary", False),
            }
            for r in filtered_results
        ]
        
        logger.info(
            f"Generated response: confidence={validation.confidence}, "
            f"sources={len(sources)}, sections={len(sections_used)}, "
            f"summaries_used={summaries_used}"
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
            summaries_used=summaries_used,
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
        
        return filtered if filtered else results
    
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
            "config": self.config.to_dict(),
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "vectorstore": self.vectorstore.health_check(),
            "llm": self.llm.health_check(),
            "retriever": self.retriever.health_check(),
        }
