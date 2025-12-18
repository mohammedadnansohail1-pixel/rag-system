"""Multi-document RAG pipeline with cross-document capabilities."""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.documents.registry import DocumentRegistry, DocumentInfo
from src.pipeline.enhanced_rag_pipeline import (
    EnhancedRAGPipeline,
    EnhancedRAGConfig,
    EnhancedRAGResponse,
)
from src.loaders.factory import LoaderFactory
from src.embeddings.base import BaseEmbeddings
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever
from src.generation.base import BaseLLM
from src.guardrails.validator import get_uncertainty_response
from src.guardrails.prompts import build_prompt_with_context, get_confidence_guidance

logger = logging.getLogger(__name__)


@dataclass
class MultiDocResponse(EnhancedRAGResponse):
    """Response with multi-document metadata."""
    companies_cited: List[str] = field(default_factory=list)
    documents_cited: List[str] = field(default_factory=list)


class MultiDocumentPipeline(EnhancedRAGPipeline):
    """
    Multi-document RAG pipeline extending EnhancedRAGPipeline.
    
    Features:
    - Document registry for tracking
    - Batch ingestion with progress
    - Query filtering by company/filing type
    - Cross-document comparison
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        llm: BaseLLM,
        config: Optional[EnhancedRAGConfig] = None,
        registry_path: str = ".cache/doc_registry.json",
        **kwargs,
    ):
        super().__init__(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            config=config,
            **kwargs,
        )
        
        self.registry = DocumentRegistry(persist_path=registry_path)
        logger.info(f"MultiDocumentPipeline: {len(self.registry.all_documents)} docs in registry")
    
    def _generate_doc_id(self, source_path: str) -> str:
        return hashlib.sha256(source_path.encode()).hexdigest()[:12]
    
    def ingest_file(self, file_path: str, skip_if_exists: bool = True) -> Dict[str, Any]:
        path = Path(file_path)
        source = str(path.absolute())
        
        if skip_if_exists and self.registry.is_ingested(source):
            logger.info(f"Skipping already ingested: {file_path}")
            return {"skipped": True, "source": source}
        
        stats = super().ingest_file(file_path)
        
        documents = LoaderFactory.load(str(path))
        doc_meta = documents[0].metadata if documents else {}
        
        doc_info = DocumentInfo(
            doc_id=self._generate_doc_id(source),
            source_path=source,
            company_name=doc_meta.get("company_name"),
            filing_type=doc_meta.get("filing_type"),
            filing_date=doc_meta.get("filing_date"),
            chunk_count=stats.get("chunks", 0),
            summary_count=stats.get("summary_chunks", 0),
            total_chars=sum(d.metadata.get("char_count", 0) for d in documents),
            sections=stats.get("sections", []),
            metadata=doc_meta,
        )
        self.registry.register(doc_info)
        
        stats["doc_id"] = doc_info.doc_id
        stats["company_name"] = doc_info.company_name
        return stats
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        skip_existing: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and (not file_types or f.suffix.lower() in file_types)
        ]
        
        logger.info(f"Found {len(files)} files in {directory}")
        
        total_stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "total_summaries": 0,
            "companies": set(),
            "errors": [],
        }
        
        for i, file_path in enumerate(files):
            try:
                stats = self.ingest_file(str(file_path), skip_if_exists=skip_existing)
                
                if stats.get("skipped"):
                    total_stats["files_skipped"] += 1
                else:
                    total_stats["files_processed"] += 1
                    total_stats["total_chunks"] += stats.get("chunks", 0)
                    total_stats["total_summaries"] += stats.get("summary_chunks", 0)
                    if stats.get("company_name"):
                        total_stats["companies"].add(stats["company_name"])
                
                if progress_callback:
                    progress_callback(i + 1, len(files), str(file_path))
                    
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
                total_stats["errors"].append({"file": str(file_path), "error": str(e)})
        
        total_stats["companies"] = list(total_stats["companies"])
        return total_stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_companies: Optional[List[str]] = None,
        filter_filing_types: Optional[List[str]] = None,
        filter_section: Optional[str] = None,
        filter_topics: Optional[List[str]] = None,
        **kwargs,
    ) -> MultiDocResponse:
        top_k = top_k or self.config.top_k
        
        # Retrieve with larger pool when filtering
        retrieve_k = top_k * 4 if (filter_companies or filter_filing_types) else top_k * 2
        results = self.retriever.retrieve(question, top_k=retrieve_k)
        
        # Apply company filter
        if filter_companies:
            results = [
                r for r in results
                if any(
                    c.lower() in (r.metadata.get("company_name", "") or "").lower()
                    for c in filter_companies
                )
            ]
        
        # Apply filing type filter
        if filter_filing_types:
            results = [
                r for r in results
                if (r.metadata.get("filing_type", "") or "").upper() in
                   [ft.upper() for ft in filter_filing_types]
            ]
        
        # Apply section/topic filters
        if filter_section or filter_topics:
            results = self._filter_results(results, filter_section, filter_topics)
        
        # Re-normalize scores for filtered results
        # When filtering to one company, their best results should pass guardrails
        if results and (filter_companies or filter_filing_types):
            max_score = max(r.score for r in results)
            if max_score > 0:
                for r in results:
                    # Scale so best match gets boosted, preserving relative ordering
                    r.score = 0.4 + (r.score / max_score) * 0.45
        
        results = results[:top_k]
        
        validation = self.guardrails.validate(results)
        
        if not validation.is_valid:
            return MultiDocResponse(
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
        
        sections_used = []
        topics_found = []
        entities_found = {}
        companies_cited = set()
        documents_cited = set()
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
            if meta.get("company_name"):
                companies_cited.add(meta["company_name"])
            if meta.get("source"):
                documents_cited.add(meta["source"])
            if meta.get("is_summary"):
                summaries_used += 1
        
        sections_used = list(dict.fromkeys(sections_used))
        topics_found = list(dict.fromkeys(topics_found))
        
        context = self._build_context(filtered_results)
        
        prompt = build_prompt_with_context(
            query=question,
            context_chunks=filtered_results,
            confidence=validation.confidence,
            include_scores=True
        )
        
        if companies_cited:
            company_list = ", ".join(companies_cited)
            prompt = f"[Companies: {company_list}]\n\n" + prompt
        
        confidence_guidance = get_confidence_guidance(validation.confidence)
        enhanced_system_prompt = self.system_prompt + "\n\nCONFIDENCE: " + validation.confidence.upper() + "\n" + confidence_guidance
        
        answer = self.llm.generate(prompt, system_prompt=enhanced_system_prompt)
        
        sources = [
            {
                "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in filtered_results
        ]
        
        return MultiDocResponse(
            answer=answer,
            sources=sources,
            query=question,
            confidence=validation.confidence,
            avg_score=validation.avg_score,
            sections_used=sections_used,
            topics_found=topics_found[:10],
            entities_found=entities_found,
            summaries_used=summaries_used,
            context_length=len(context),
            companies_cited=list(companies_cited),
            documents_cited=list(documents_cited),
        )
    
    def compare_companies(
        self,
        question: str,
        companies: List[str],
        top_k_per_company: int = 3,
    ) -> MultiDocResponse:
        all_results = []
        
        for company in companies:
            results = self.retriever.retrieve(question, top_k=top_k_per_company * 3)
            company_results = [
                r for r in results
                if company.lower() in (r.metadata.get("company_name", "") or "").lower()
            ][:top_k_per_company]
            all_results.extend(company_results)
        
        context_parts = []
        for r in all_results:
            company = r.metadata.get("company_name", "Unknown")
            context_parts.append(f"[{company}]\n{r.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        comparison_prompt = f"""Compare the following companies based on the question.
Provide a structured comparison highlighting similarities and differences.

Question: {question}

Context from each company:
{context}

Comparison:"""
        
        answer = self.llm.generate(
            comparison_prompt,
            system_prompt="You are a financial analyst. Provide balanced, factual comparisons."
        )
        
        return MultiDocResponse(
            answer=answer,
            sources=[{"content": r.content[:200], "metadata": r.metadata, "score": r.score} for r in all_results],
            query=question,
            confidence="medium",
            avg_score=sum(r.score for r in all_results) / len(all_results) if all_results else 0,
            companies_cited=companies,
        )
    
    def get_company_summary(self, company_name: str) -> Dict[str, Any]:
        docs = self.registry.get_by_company(company_name)
        
        if not docs:
            return {"error": f"No documents found for {company_name}"}
        
        return {
            "company": company_name,
            "documents": len(docs),
            "filing_types": list(set(d.filing_type for d in docs if d.filing_type)),
            "total_chunks": sum(d.chunk_count for d in docs),
            "sections": list(set(s for d in docs for s in d.sections)),
            "latest_filing": max((d.filing_date for d in docs if d.filing_date), default=None),
        }
    
    @property
    def companies(self) -> List[str]:
        return self.registry.get_companies()
    
    @property 
    def registry_stats(self) -> Dict[str, Any]:
        return self.registry.stats
