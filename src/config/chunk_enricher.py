"""
Chunk Enricher - Implements research-backed enrichment strategies.

Key techniques:
- SAC (Summary-Augmented Chunking): arXiv:2510.06999 - Legal RAG
- Keyword enrichment: arXiv:2402.05131 - Financial RAG  
- NL Descriptions: ICSE 2026 - Code RAG
"""
import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.domain_config import DomainConfig

logger = logging.getLogger(__name__)


class ChunkEnricher:
    """
    Enriches chunks based on domain-specific research findings.
    
    Usage:
        enricher = ChunkEnricher(domain_config, llm_client)
        enriched_chunks = enricher.enrich_document(chunks, document)
    """
    
    def __init__(self, config: "DomainConfig", llm=None):
        """
        Args:
            config: DomainConfig with enrichment settings
            llm: Optional LLM for generating summaries/descriptions
        """
        self.config = config
        self.llm = llm
        self._summary_cache: Dict[str, str] = {}
    
    def enrich_document(
        self,
        chunks: List[Any],
        document_content: str,
        document_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enrich all chunks from a document.
        
        For SAC, generates ONE summary and prepends to ALL chunks.
        This is the key insight from legal RAG research.
        
        Args:
            chunks: List of Chunk objects or strings
            document_content: Full document for summary generation
            document_metadata: Optional metadata
            
        Returns:
            List of dicts with 'content', 'enriched_content', 'metadata'
        """
        if document_metadata is None:
            document_metadata = {}
        
        enrichment = self.config.enrichment
        doc_id = self._get_doc_id(document_content, document_metadata)
        
        # Generate document summary ONCE (SAC)
        doc_summary = None
        if enrichment.add_document_summary:
            doc_summary = self._get_or_generate_summary(
                doc_id, 
                document_content,
                enrichment.summary_prompt,
                enrichment.summary_max_chars
            )
            logger.debug(f"Generated summary for {doc_id}: {doc_summary[:50]}...")
        
        # Enrich each chunk
        enriched = []
        for idx, chunk in enumerate(chunks):
            # Handle both Chunk objects and strings
            if hasattr(chunk, 'content'):
                content = chunk.content
                chunk_meta = getattr(chunk, 'metadata', {})
            else:
                content = str(chunk)
                chunk_meta = {}
            
            # Merge metadata
            metadata = {**document_metadata, **chunk_meta, 'chunk_index': idx}
            
            # Build enriched content
            enriched_content = self._enrich_chunk(
                content=content,
                doc_summary=doc_summary,
                metadata=metadata,
            )
            
            enriched.append({
                'content': content,
                'enriched_content': enriched_content,
                'metadata': metadata,
                'document_id': doc_id,
                'has_summary': doc_summary is not None,
            })
        
        logger.info(f"Enriched {len(enriched)} chunks (SAC={doc_summary is not None})")
        return enriched
    
    def _enrich_chunk(
        self,
        content: str,
        doc_summary: Optional[str],
        metadata: Dict[str, Any],
    ) -> str:
        """Build enriched content for a single chunk."""
        enrichment = self.config.enrichment
        parts = []
        
        # 1. Document summary (SAC) - CRITICAL for legal
        if doc_summary:
            parts.append(f"[Document: {doc_summary}]")
        
        # 2. Section context
        if enrichment.add_section_context:
            section = metadata.get('section') or metadata.get('heading', '')
            if section:
                parts.append(f"[Section: {section}]")
        
        # 3. Source path
        if enrichment.add_source_path:
            source = metadata.get('source') or metadata.get('file_path', '')
            if source:
                # Shorten if too long
                if len(source) > 60:
                    source = "..." + source[-57:]
                parts.append(f"[Source: {source}]")
        
        # 4. Keywords
        if enrichment.add_keywords:
            keywords = self._extract_keywords(content, enrichment.max_keywords)
            if keywords:
                parts.append(f"[Keywords: {', '.join(keywords)}]")
        
        # 5. NL description (for code)
        if enrichment.add_nl_description and self._is_code(content):
            desc = self._generate_nl_description(content)
            if desc:
                parts.append(f"[Description: {desc}]")
        
        # Combine
        if parts:
            prefix = "\n".join(parts)
            return f"{prefix}\n\n{content}"
        return content
    
    def _get_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID."""
        source = metadata.get('source', '')
        hash_input = f"{source}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _get_or_generate_summary(
        self,
        doc_id: str,
        content: str,
        prompt: Optional[str],
        max_chars: int
    ) -> str:
        """Get cached summary or generate new one."""
        if doc_id in self._summary_cache:
            return self._summary_cache[doc_id]
        
        summary = self._generate_summary(content, prompt, max_chars)
        self._summary_cache[doc_id] = summary
        return summary
    
    def _generate_summary(
        self,
        content: str,
        prompt: Optional[str],
        max_chars: int
    ) -> str:
        """Generate document summary."""
        if self.llm and prompt:
            try:
                full_prompt = f"{prompt}\n\nDocument:\n{content[:4000]}"
                response = self.llm.generate(full_prompt)
                return response.strip()[:max_chars]
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}, using fallback")
        
        # Fallback: extract first meaningful sentences
        return self._extract_summary_fallback(content, max_chars)
    
    def _extract_summary_fallback(self, content: str, max_chars: int) -> str:
        """Fallback summary without LLM."""
        # Handle single-line documents (like SEC filings)
        if content.count('\n') < 5:
            # Split by sentences instead
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content[:5000])
            lines = [s.strip() for s in sentences if len(s.strip()) > 20]
        else:
            lines = [l.strip() for l in content.split('\n') if l.strip() and len(l.strip()) > 20]
        
        # Skip boilerplate
        skip = ['navigation', 'copyright', 'skip to', 'table of contents', 'accession number']
        lines = [l for l in lines if not any(s in l.lower() for s in skip)]
        
        summary_parts = []
        length = 0
        for line in lines[:5]:
            if length + len(line) > max_chars:
                break
            summary_parts.append(line)
            length += len(line) + 1
        
        result = ' '.join(summary_parts)[:max_chars]
        
        # If still empty, just take first N chars
        if not result and content:
            result = content[:max_chars].strip()
        
        # Clean SEC header junk from summary
        import re
        # Remove accession numbers, SGML references, etc.
        result = re.sub(r'\d{10}-\d{2}-\d{6}', '', result)  # Accession numbers
        result = re.sub(r'\.hdr\.sgml[^\s]*', '', result)  # SGML refs
        result = re.sub(r'\.txt\s*:', '', result)  # .txt :
        result = re.sub(r'\s{2,}', ' ', result).strip()  # Multiple spaces
        
        return result[:max_chars]
    
    def _extract_keywords(self, content: str, max_keywords: int) -> List[str]:
        """Extract keywords from content."""
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{3,}\b', content)
        
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'will', 'been',
            'were', 'they', 'their', 'what', 'when', 'where', 'which',
            'there', 'these', 'those', 'then', 'than', 'into', 'only',
            'other', 'such', 'more', 'some', 'could', 'would', 'should',
        }
        
        freq = {}
        for word in words:
            lower = word.lower()
            if lower not in stopwords and len(lower) > 3:
                freq[lower] = freq.get(lower, 0) + 1
        
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _is_code(self, content: str) -> bool:
        """Detect if content is likely code."""
        indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'return ', '>>> ']
        count = sum(1 for ind in indicators if ind in content)
        return count >= 2
    
    def _generate_nl_description(self, content: str) -> Optional[str]:
        """Generate NL description for code."""
        if self.llm:
            try:
                prompt = self.config.enrichment.nl_description_prompt or \
                    "Describe what this code does in one sentence:"
                response = self.llm.generate(f"{prompt}\n\n{content[:1000]}")
                return response.strip()[:150]
            except:
                pass
        
        # Fallback: extract from docstring/comments
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                return line.strip('"').strip("'")[:150]
            if line.startswith('#') and not line.startswith('#!'):
                return line[1:].strip()[:150]
        return None
    
    def clear_cache(self):
        """Clear summary cache."""
        self._summary_cache.clear()
