#!/usr/bin/env python
"""
Multi-Domain RAG Demo

Demonstrates RAG system handling multiple document types:
- SEC Filings (Finance) - 10-K annual reports
- Technical Docs - RAG guides  
- PDF Books (Education) - Machine learning textbook

Features:
- Automatic document type detection
- Section extraction for SEC filings
- Domain-based filtering
- Hybrid retrieval (Dense + BM25)
- LLM answer generation with citations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import LoaderFactory
from src.chunkers.recursive_chunker import RecursiveChunker
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.generation.ollama_llm import OllamaLLM


class MultiDomainRAG:
    """Multi-domain RAG system."""
    
    DOMAINS = {
        "finance": {
            "description": "SEC 10-K Filings (Netflix, Apple)",
            "files": [
                ("NFLX", "data/sec_filings/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt"),
                ("AAPL", "data/validation/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"),
            ]
        },
        "technical": {
            "description": "RAG System Documentation",
            "files": [
                ("rag_guide", "data/sample/rag_intro.txt"),
                ("chunking_guide", "data/sample/chunking.txt"),
            ]
        },
        "education": {
            "description": "Machine Learning Textbook (PDF)",
            "files": [
                ("ml_book", "data/books/machine_learning_basics.pdf"),
            ]
        },
    }
    
    def __init__(self, collection_name: str = "multi_domain_rag"):
        self.embeddings = OllamaEmbeddings(
            host="http://localhost:11434",
            model="nomic-embed-text",
            dimensions=768,
        )
        self.vectorstore = QdrantHybridStore(
            collection_name=collection_name,
            host="localhost",
            port=6333,
            dense_dimensions=768,
            recreate_collection=False,
        )
        self.retriever = HybridRetriever(
            embeddings=self.embeddings,
            vectorstore=self.vectorstore,
            sparse_encoder="bm25",
        )
        self.llm = OllamaLLM(
            host="http://localhost:11434",
            model="llama3.2:latest",
            temperature=0.1,
        )
        self.chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    
    def index_all(self, recreate: bool = True):
        """Index all domains."""
        if recreate:
            self.vectorstore = QdrantHybridStore(
                collection_name="multi_domain_rag",
                host="localhost",
                port=6333,
                dense_dimensions=768,
                recreate_collection=True,
            )
            self.retriever = HybridRetriever(
                embeddings=self.embeddings,
                vectorstore=self.vectorstore,
                sparse_encoder="bm25",
            )
        
        total = 0
        for domain_name, domain_info in self.DOMAINS.items():
            print(f"\n📁 {domain_name.upper()}: {domain_info['description']}")
            
            for source_id, filepath in domain_info['files']:
                path = Path(filepath)
                if not path.exists():
                    print(f"   ⚠ {source_id}: Not found")
                    continue
                
                try:
                    docs = LoaderFactory.load(path)
                    chunks = []
                    for doc in docs:
                        for chunk in self.chunker.chunk(doc):
                            chunk.metadata['domain'] = domain_name
                            chunk.metadata['source'] = source_id
                            chunks.append(chunk)
                    
                    if chunks:
                        texts = [c.content for c in chunks]
                        metadatas = [c.metadata for c in chunks]
                        ids = self.retriever.add_documents(texts, metadatas)
                        total += len(ids)
                        print(f"   ✓ {source_id}: {len(ids)} chunks")
                except Exception as e:
                    print(f"   ✗ {source_id}: {e}")
        
        return total
    
    def query(self, question: str, domain: str = None, source: str = None, top_k: int = 3):
        """Query the RAG system."""
        # Build filter
        meta_filter = {}
        if domain:
            meta_filter['domain'] = domain
        if source:
            meta_filter['source'] = source
        
        # Retrieve
        results = self.retriever.retrieve(
            question, 
            top_k=top_k, 
            metadata_filter=meta_filter if meta_filter else None
        )
        
        # Generate
        context = [r.content for r in results]
        answer = self.llm.generate_with_context(question, context)
        
        return answer, results
    
    def interactive(self):
        """Interactive query mode."""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("\nCommands:")
        print("  <question>                    - Query all domains")
        print("  <question> @finance           - Query finance domain only")
        print("  <question> @technical         - Query technical domain only")
        print("  <question> @education         - Query education domain only")
        print("  <question> @NFLX              - Query Netflix docs only")
        print("  <question> @AAPL              - Query Apple docs only")
        print("  domains                       - List available domains")
        print("  quit                          - Exit")
        
        while True:
            try:
                user_input = input("\n❯ ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'domains':
                    for name, info in self.DOMAINS.items():
                        print(f"  @{name}: {info['description']}")
                    continue
                
                # Parse domain filter
                domain = None
                source = None
                if '@' in user_input:
                    parts = user_input.rsplit('@', 1)
                    question = parts[0].strip()
                    filter_val = parts[1].strip().lower()
                    
                    if filter_val in self.DOMAINS:
                        domain = filter_val
                    elif filter_val.upper() in ['NFLX', 'AAPL']:
                        source = filter_val.upper()
                        domain = 'finance'
                    else:
                        print(f"Unknown filter: @{filter_val}")
                        continue
                else:
                    question = user_input
                
                # Query
                filter_desc = f"@{domain or source or 'all'}"
                print(f"\n🔍 Searching {filter_desc}...")
                
                answer, results = self.query(question, domain=domain, source=source)
                
                # Show sources
                print("\n📚 Sources:")
                for i, r in enumerate(results):
                    d = r.metadata.get('domain', '?')
                    s = r.metadata.get('source', '?')
                    sec = r.metadata.get('section', '')
                    loc = f"{d}/{s}" + (f"/{sec}" if sec else "")
                    print(f"   [{i+1}] {loc} (score={r.score:.2f})")
                
                # Show answer
                print(f"\n💬 Answer:\n{answer}")
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye! 👋")


def main():
    print("=" * 70)
    print("🚀 MULTI-DOMAIN RAG SYSTEM")
    print("=" * 70)
    print("\nSupported domains:")
    for name, info in MultiDomainRAG.DOMAINS.items():
        print(f"  • {name}: {info['description']}")
    
    rag = MultiDomainRAG()
    
    # Check if collection exists and has data
    import requests
    try:
        r = requests.get("http://localhost:6333/collections/multi_domain_rag")
        if r.status_code == 200:
            points = r.json().get('result', {}).get('points_count', 0)
            if points > 0:
                print(f"\n✓ Found existing index with {points:,} chunks")
                reindex = input("Re-index documents? (y/N): ").strip().lower()
                if reindex == 'y':
                    total = rag.index_all(recreate=True)
                    print(f"\n✓ Indexed {total:,} chunks")
            else:
                print("\n📥 Indexing documents...")
                total = rag.index_all(recreate=True)
                print(f"\n✓ Indexed {total:,} chunks")
        else:
            print("\n📥 Indexing documents...")
            total = rag.index_all(recreate=True)
            print(f"\n✓ Indexed {total:,} chunks")
    except:
        print("\n📥 Indexing documents...")
        total = rag.index_all(recreate=True)
        print(f"\n✓ Indexed {total:,} chunks")
    
    # Demo queries
    print("\n" + "=" * 70)
    print("DEMO QUERIES")
    print("=" * 70)
    
    demos = [
        ("What does Netflix do?", "finance", None),
        ("What is RAG?", "technical", None),
        ("What is gradient descent?", "education", None),
    ]
    
    for question, domain, source in demos:
        print(f"\n❯ {question} @{domain}")
        answer, results = rag.query(question, domain=domain, source=source)
        sources = [f"{r.metadata.get('source')}" for r in results[:2]]
        print(f"  Sources: {', '.join(sources)}")
        print(f"  Answer: {answer[:200]}...")
    
    # Interactive mode
    rag.interactive()


if __name__ == "__main__":
    main()
