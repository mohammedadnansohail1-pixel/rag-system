"""
Domain-specific configuration for RAG pipelines.

Each domain has optimal settings for:
- Chunking (size, overlap, strategy)
- Retrieval (k, sparse encoder, fusion weights)
- Query expansion (method, prompts)
- Generation (system prompts, temperature)

Usage:
    from src.config import get_domain_config, FINANCIAL
    
    config = get_domain_config(FINANCIAL)
    chunker = config.create_chunker()
    retriever = config.create_retriever(embeddings, vectorstore)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Domain identifiers
FINANCIAL = "financial"
TECHNICAL = "technical"
LEGAL = "legal"
HEALTHCARE = "healthcare"
ECOMMERCE = "ecommerce"
GENERAL = "general"


@dataclass
class ChunkingConfig:
    """Chunking settings for a domain."""
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Domain-specific separators
    separators: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.separators:
            d["separators"] = self.separators
        return d


@dataclass
class RetrievalConfig:
    """Retrieval settings for a domain."""
    top_k: int = 10
    sparse_encoder: str = "bm25"
    sparse_weight: float = 0.3  # For hybrid fusion
    dense_weight: float = 0.7
    use_reranking: bool = False
    rerank_top_k: int = 20  # Retrieve this many, rerank to top_k
    
    # BM25 tuning
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass  
class ExpansionConfig:
    """Query expansion settings for a domain."""
    method: str = "none"  # none, synonym, hyde, llm
    hyde_prompt: Optional[str] = None
    synonym_terms: Optional[Dict[str, List[str]]] = None


@dataclass
class GenerationConfig:
    """LLM generation settings for a domain."""
    system_prompt: str = "You are a helpful assistant. Answer based on the provided context."
    temperature: float = 0.1
    max_tokens: int = 1024
    citation_style: str = "inline"  # inline, footnote, none

@dataclass
class EnrichmentConfig:
    """
    Chunk enrichment settings based on research findings.
    
    Research sources:
    - SAC: arXiv:2510.06999 (Legal) - Reduces DRM by 50%+
    - Keywords: arXiv:2402.05131 (Financial) - Improves retrieval 5%+
    - NL Descriptions: ICSE 2026 (Code) - Bridges semantic gap
    """
    # Summary-Augmented Chunking (CRITICAL for legal - 95% DRM without it)
    add_document_summary: bool = False
    summary_max_chars: int = 150
    summary_prompt: Optional[str] = None
    
    # Keyword enrichment (useful for financial)
    add_keywords: bool = False
    max_keywords: int = 6
    
    # Section/heading context
    add_section_context: bool = False
    
    # Natural language descriptions (useful for code)
    add_nl_description: bool = False
    nl_description_prompt: Optional[str] = None
    
    # Source path tracking
    add_source_path: bool = True




@dataclass
class DomainConfig:
    """Complete configuration for a domain."""
    name: str
    description: str
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    expansion: ExpansionConfig
    generation: GenerationConfig
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    
    # Optional domain-specific patterns
    document_patterns: List[str] = field(default_factory=list)  # File patterns to match
    
    def create_chunker(self):
        """Create configured chunker for this domain."""
        from src.chunkers.factory import ChunkerFactory
        return ChunkerFactory.from_config(self.chunking.to_dict())
    
    def create_sparse_encoder(self):
        """Create configured sparse encoder."""
        from src.retrieval.sparse_encoder import SparseEncoderFactory
        
        if self.retrieval.sparse_encoder == "bm25":
            return SparseEncoderFactory.create(
                "bm25",
                k1=self.retrieval.bm25_k1,
                b=self.retrieval.bm25_b,
            )
        return SparseEncoderFactory.create(self.retrieval.sparse_encoder)
    
    def create_query_expander(self, llm=None):
        """Create configured query expander."""
        from src.retrieval.query_expansion import QueryExpanderFactory
        
        if self.expansion.method == "none":
            return None
        
        if self.expansion.method == "synonym":
            return QueryExpanderFactory.create(
                "synonym",
                custom_synonyms=self.expansion.synonym_terms,
            )
        
        if self.expansion.method == "hyde" and llm:
            from src.retrieval.query_expansion import HyDEExpander
            expander = HyDEExpander(llm=llm, domain=self.name)
            # Override prompt if custom
            if self.expansion.hyde_prompt:
                expander.prompt_template = self.expansion.hyde_prompt
            return expander
        
        return None
    
    def get_system_prompt(self) -> str:
        """Get generation system prompt."""
        return self.generation.system_prompt
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Domain: {self.name}
Description: {self.description}

Chunking:
  Strategy: {self.chunking.strategy}
  Size: {self.chunking.chunk_size}, Overlap: {self.chunking.chunk_overlap}

Retrieval:
  Top-K: {self.retrieval.top_k}
  Sparse: {self.retrieval.sparse_encoder} (weight={self.retrieval.sparse_weight})
  Reranking: {self.retrieval.use_reranking}

Query Expansion: {self.expansion.method}

Generation:
  Temperature: {self.generation.temperature}
"""


class DomainRegistry:
    """Registry of domain configurations."""
    
    _domains: Dict[str, DomainConfig] = {}
    
    @classmethod
    def register(cls, config: DomainConfig) -> None:
        """Register a domain configuration."""
        cls._domains[config.name] = config
        logger.info(f"Registered domain config: {config.name}")
    
    @classmethod
    def get(cls, name: str) -> DomainConfig:
        """Get domain configuration by name."""
        if name not in cls._domains:
            available = ", ".join(cls._domains.keys())
            raise ValueError(f"Unknown domain: {name}. Available: {available}")
        return cls._domains[name]
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """List all registered domains."""
        return list(cls._domains.keys())


# =============================================================================
# PRE-CONFIGURED DOMAINS
# =============================================================================

# FINANCIAL DOMAIN (10-K, earnings reports, financial statements)
_FINANCIAL_CONFIG = DomainConfig(
    name=FINANCIAL,
    description="SEC filings, earnings reports, financial statements",
    chunking=ChunkingConfig(
        strategy="recursive",  # TODO: element_type when implemented
        chunk_size=1500,  # Research: arXiv:2402.05131 - larger preserves financial context
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
        # Research: keep_tables_intact=True (to implement in chunker)
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        sparse_encoder="bm25",
        sparse_weight=0.2,  # Less sparse weight - numbers don't match well
        dense_weight=0.8,
        use_reranking=False,  # General rerankers hurt on financial
        bm25_k1=1.2,
        bm25_b=0.6,  # Less length normalization for tables
    ),
    expansion=ExpansionConfig(
        method="hyde",
        hyde_prompt='''Write a 2-3 sentence passage that would appear in a company's 
SEC 10-K filing to answer this question. Use formal financial language.

Question: {query}

Passage:''',
    ),
    generation=GenerationConfig(
        system_prompt="""You are a financial analyst assistant. Answer questions based on the provided SEC filings and financial documents.
- Cite specific numbers and figures when available
- Note the time period for any financial data
- Be precise with financial terminology
- If data is not in the context, say so clearly""",
        temperature=0.1,
        citation_style="inline",
    ),
    document_patterns=["10-K", "10-Q", "8-K", "earnings", "annual report"],
    enrichment=EnrichmentConfig(
        # Research: arXiv:2402.05131 - keywords improve financial retrieval
        add_document_summary=True,
        summary_max_chars=200,
        summary_prompt="Summarize: company name, fiscal period, key metrics, main events. Under 200 chars.",
        add_keywords=True,
        max_keywords=6,
        add_section_context=True,
        add_source_path=True,
    ),
)

# TECHNICAL DOMAIN (documentation, code, APIs)
_TECHNICAL_CONFIG = DomainConfig(
    name=TECHNICAL,
    description="Technical documentation, API docs, code repositories",
    chunking=ChunkingConfig(
        strategy="recursive",  # TODO: syntax_aware when implemented
        chunk_size=1000,  # Balance: code units + documentation context
        chunk_overlap=100,  # Moderate overlap
        separators=["\nclass ", "\ndef ", "\n\n", "\n```\n", "\n", ". "],  # Syntax-aware
        # Research: include imports/class definitions with methods (to implement)
    ),
    retrieval=RetrievalConfig(
        top_k=8,
        sparse_encoder="bm25",
        sparse_weight=0.35,  # Balanced: function names + semantic matching
        dense_weight=0.65,
        use_reranking=False,
        bm25_k1=1.5,
        bm25_b=0.75,
    ),
    expansion=ExpansionConfig(
        method="synonym",
        synonym_terms={
            "function": ["method", "def", "procedure"],
            "class": ["object", "type", "struct"],
            "error": ["exception", "bug", "issue", "failure"],
            "install": ["setup", "configure", "pip", "npm"],
            "import": ["include", "require", "use"],
            "return": ["output", "result", "response"],
            "parameter": ["argument", "param", "arg", "option"],
            "api": ["endpoint", "route", "interface"],
            "database": ["db", "sql", "storage", "table"],
            "async": ["asynchronous", "await", "concurrent"],
        },
    ),
    generation=GenerationConfig(
        system_prompt="""You are a technical documentation assistant. Answer questions based on the provided documentation.
- Include code examples when relevant
- Be precise with function names and parameters
- Note version-specific behavior if mentioned
- Link concepts to related documentation when helpful""",
        temperature=0.1,
        citation_style="inline",
    ),
    document_patterns=["README", "docs", "api", ".md", ".rst"],
    enrichment=EnrichmentConfig(
        # Research: ICSE 2026/Qodo - NL descriptions bridge code↔query gap
        add_document_summary=False,  # Less critical for code
        add_keywords=True,  # Function names, parameters
        max_keywords=8,
        add_section_context=True,  # File path, class name
        add_nl_description=True,  # Critical for code
        nl_description_prompt="Describe what this code does in 1-2 sentences. Focus on purpose, inputs, outputs.",
        add_source_path=True,
    ),
)

# LEGAL DOMAIN (contracts, terms, policies)
_LEGAL_CONFIG = DomainConfig(
    name=LEGAL,
    description="Contracts, terms of service, legal agreements, policies",
    chunking=ChunkingConfig(
        strategy="recursive",
        chunk_size=1000,  # Balanced: precise retrieval + enough content
        chunk_overlap=150,
        separators=["\n\n", "\nSection", "\nArticle", "\n", ". "],
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        sparse_encoder="bm25",
        sparse_weight=0.5,  # Legal docs have precise terminology
        dense_weight=0.5,
        use_reranking=False,
        bm25_k1=1.8,  # Higher term frequency importance
        bm25_b=0.4,  # Less length normalization
    ),
    expansion=ExpansionConfig(
        method="synonym",
        synonym_terms={
            "shall": ["must", "will", "is required to"],
            "liability": ["responsibility", "obligation", "damages"],
            "indemnify": ["compensate", "reimburse", "hold harmless"],
            "terminate": ["end", "cancel", "discontinue"],
            "breach": ["violation", "default", "non-compliance"],
            "parties": ["company", "user", "customer", "provider"],
            "agreement": ["contract", "terms", "arrangement"],
            "warranty": ["guarantee", "assurance", "representation"],
            "confidential": ["proprietary", "secret", "private"],
            "intellectual property": ["ip", "copyright", "trademark", "patent"],
        },
    ),
    generation=GenerationConfig(
        system_prompt="""You are a legal document assistant. Answer questions based on the provided legal documents.
- Quote exact language from the document when relevant
- Note which section or clause contains the information
- Do not provide legal advice - only summarize what the document says
- Highlight any ambiguities or conditions""",
        temperature=0.0,  # Deterministic for legal
        citation_style="footnote",
    ),
    document_patterns=["terms", "privacy", "agreement", "contract", "policy"],
    enrichment=EnrichmentConfig(
        # arXiv:2510.06999 - SAC reduces DRM for multi-document scenarios
        # Note: SAC most useful with multiple similar documents
        add_document_summary=True,
        summary_max_chars=80,  # Reduced to ~8% of chunk
        summary_prompt="Summarize: document type, parties involved, core subject matter, key identifiers. Under 150 chars.",
        add_keywords=False,  # Research: generic summary beats keyword enrichment
        add_section_context=True,  # Include clause/section numbers
        add_source_path=True,
    ),
)

# GENERAL DOMAIN (fallback)
_GENERAL_CONFIG = DomainConfig(
    name=GENERAL,
    description="General purpose documents",
    chunking=ChunkingConfig(
        strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200,
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        sparse_encoder="bm25",
        sparse_weight=0.3,
        dense_weight=0.7,
        use_reranking=False,
    ),
    expansion=ExpansionConfig(method="none"),
    generation=GenerationConfig(
        system_prompt="You are a helpful assistant. Answer based on the provided context.",
        temperature=0.1,
    ),
    enrichment=EnrichmentConfig(
        add_document_summary=False,
        add_keywords=False,
        add_section_context=False,
        add_source_path=True,
    ),
)

# Register all domains
for config in [_FINANCIAL_CONFIG, _TECHNICAL_CONFIG, _LEGAL_CONFIG, _GENERAL_CONFIG]:
    DomainRegistry.register(config)


# Convenience functions
def get_domain_config(domain: str) -> DomainConfig:
    """Get configuration for a domain."""
    return DomainRegistry.get(domain)


def register_domain(config: DomainConfig) -> None:
    """Register a custom domain configuration."""
    DomainRegistry.register(config)
