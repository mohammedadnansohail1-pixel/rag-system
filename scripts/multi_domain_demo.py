#!/usr/bin/env python3
"""
Multi-Domain RAG Demo
=====================
Shows the system working across Legal, Technical, HR, and Research documents.
"""

import time
import sys

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

def section(title):
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{WHITE}{title}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")
    time.sleep(0.5)

def demo_domain(pipeline, domain_name, directory, queries, emoji):
    """Run demo for a specific domain."""
    section(f"{emoji} {domain_name.upper()} DOCUMENTS")
    
    # Ingest
    print(f"{YELLOW}Ingesting {domain_name} documents...{RESET}")
    start = time.time()
    stats = pipeline.ingest_directory(directory, recursive=True)
    ingest_time = time.time() - start
    
    print(f"{GREEN}✓ Ingested {stats['total_chunks']} chunks in {ingest_time:.1f}s{RESET}\n")
    
    # Query
    for q in queries:
        print(f"{WHITE}Q: {q}{RESET}")
        start = time.time()
        response = pipeline.query(q, top_k=3)
        qtime = time.time() - start
        
        conf_color = GREEN if response.confidence == "high" else YELLOW if response.confidence == "medium" else "\033[91m"
        emoji_conf = "🟢" if response.confidence == "high" else "🟡" if response.confidence == "medium" else "🔴"
        
        print(f"{emoji_conf} {conf_color}{response.confidence.upper()}{RESET} | {qtime:.2f}s | {len(response.sources)} sources")
        print(f"{CYAN}A: {response.answer[:300]}...{RESET}\n")
        time.sleep(1)


def main():
    print("\033[2J\033[H", end="")  # Clear screen
    
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║{RESET}{BOLD}{WHITE}              MULTI-DOMAIN DOCUMENT INTELLIGENCE                      {RESET}{CYAN}║
║{RESET}{WHITE}         Legal • Technical • HR • Research Documents                  {RESET}{CYAN}║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")
    time.sleep(1)
    
    # Initialize
    print(f"{YELLOW}Initializing system...{RESET}")
    
    from src.embeddings import OllamaEmbeddings, CachedEmbeddings
    from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
    from src.retrieval import HybridRetriever
    from src.generation.ollama_llm import OllamaLLM
    from src.documents import MultiDocumentPipeline
    
    embeddings = CachedEmbeddings(OllamaEmbeddings(model="nomic-embed-text"), enabled=True)
    vectorstore = QdrantHybridStore(
        collection_name="multi_domain_demo", 
        dense_dimensions=768, 
        recreate_collection=True
    )
    retriever = HybridRetriever(
        embeddings=embeddings, 
        vectorstore=vectorstore, 
        sparse_encoder="fastembed"
    )
    llm = OllamaLLM(model="llama3.2")
    
    pipeline = MultiDocumentPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        registry_path=".cache/multi_domain.json",
    )
    print(f"{GREEN}✓ System ready{RESET}")
    time.sleep(1)
    
    # Domain 1: Legal Contracts
    section("⚖️  LEGAL CONTRACTS")
    print(f"{YELLOW}Ingesting commercial contracts...{RESET}")
    start = time.time()
    stats = pipeline.ingest_directory("data/demo_docs/legal/", recursive=True)
    print(f"{GREEN}✓ {stats['total_chunks']} chunks in {time.time()-start:.1f}s{RESET}\n")
    
    legal_queries = [
        "What are the termination clauses in these contracts?",
        "What confidentiality obligations are specified?",
        "What are the payment terms?",
    ]
    
    for q in legal_queries:
        print(f"{WHITE}Q: {q}{RESET}")
        start = time.time()
        response = pipeline.query(q, top_k=3)
        qtime = time.time() - start
        emoji = "🟢" if response.confidence == "high" else "🟡" if response.confidence == "medium" else "🔴"
        print(f"{emoji} {response.confidence.upper()} | {qtime:.2f}s")
        print(f"{CYAN}A: {response.answer[:250]}...{RESET}\n")
        time.sleep(0.5)
    
    # Domain 2: HR Policies
    section("👥 HR POLICIES")
    print(f"{YELLOW}Ingesting HR documents...{RESET}")
    start = time.time()
    stats = pipeline.ingest_directory("data/demo_docs/hr/", recursive=True)
    print(f"{GREEN}✓ {stats['total_chunks']} chunks in {time.time()-start:.1f}s{RESET}\n")
    
    hr_queries = [
        "What is the PTO policy?",
        "What health benefits are offered?",
        "What is the remote work policy?",
    ]
    
    for q in hr_queries:
        print(f"{WHITE}Q: {q}{RESET}")
        start = time.time()
        response = pipeline.query(q, top_k=3)
        qtime = time.time() - start
        emoji = "🟢" if response.confidence == "high" else "🟡" if response.confidence == "medium" else "🔴"
        print(f"{emoji} {response.confidence.upper()} | {qtime:.2f}s")
        print(f"{CYAN}A: {response.answer[:250]}...{RESET}\n")
        time.sleep(0.5)
    
    # Domain 3: Technical Docs
    section("💻 TECHNICAL DOCUMENTATION")
    print(f"{YELLOW}Ingesting technical docs...{RESET}")
    start = time.time()
    stats = pipeline.ingest_directory("data/demo_docs/technical/", recursive=True)
    print(f"{GREEN}✓ {stats['total_chunks']} chunks in {time.time()-start:.1f}s{RESET}\n")
    
    tech_queries = [
        "How do I create a FastAPI endpoint?",
        "What is Qdrant used for?",
        "How does LangChain work?",
    ]
    
    for q in tech_queries:
        print(f"{WHITE}Q: {q}{RESET}")
        start = time.time()
        response = pipeline.query(q, top_k=3)
        qtime = time.time() - start
        emoji = "🟢" if response.confidence == "high" else "🟡" if response.confidence == "medium" else "🔴"
        print(f"{emoji} {response.confidence.upper()} | {qtime:.2f}s")
        print(f"{CYAN}A: {response.answer[:250]}...{RESET}\n")
        time.sleep(0.5)
    
    # Summary
    section("📊 DEMO SUMMARY")
    
    reg_stats = pipeline.registry_stats
    print(f"""
{WHITE}┌─────────────────────────────────────────────────────────────────────┐
│{RESET} {BOLD}DOCUMENTS PROCESSED{RESET}                                               {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} Total Documents:  {BOLD}{reg_stats['total_documents']}{RESET}                                            {WHITE}│
│{RESET} Total Chunks:     {BOLD}{reg_stats['total_chunks']}{RESET}                                            {WHITE}│
│{RESET} Cache Hit Rate:   {BOLD}{embeddings.stats['hit_rate']}{RESET}                                        {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} {BOLD}SUPPORTED DOMAINS{RESET}                                                 {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} ⚖️  Legal:       Contracts, agreements, terms                        {WHITE}│
│{RESET} 👥 HR:          Policies, handbooks, procedures                     {WHITE}│
│{RESET} 💻 Technical:   Documentation, READMEs, guides                      {WHITE}│
│{RESET} 📄 Financial:   SEC filings, reports, statements                    {WHITE}│
│{RESET} 📚 Research:    Papers, articles, studies                           {WHITE}│
└─────────────────────────────────────────────────────────────────────┘
{RESET}""")
    
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║{RESET}{BOLD}{WHITE}                    ONE SYSTEM. ANY DOCUMENTS.                        {RESET}{CYAN}║
║{RESET}                                                                      {CYAN}║
║{RESET}{WHITE}   → Contracts, policies, technical docs, research papers            {RESET}{CYAN}║
║{RESET}{WHITE}   → Instant answers with source citations                           {RESET}{CYAN}║
║{RESET}{WHITE}   → Confidence scoring to prevent hallucinations                    {RESET}{CYAN}║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")


if __name__ == "__main__":
    main()
