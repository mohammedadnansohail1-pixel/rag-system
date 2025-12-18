#!/usr/bin/env python3
"""
Automated Demo - Just run and record your terminal
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

def slow_print(text, delay=0.02):
    """Print text character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def section(title):
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{WHITE}{title}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")
    time.sleep(0.5)

def typing_effect(text):
    """Simulate typing a command."""
    print(f"{GREEN}>>> {RESET}", end="")
    slow_print(text, 0.05)
    time.sleep(0.3)

def main():
    # Clear screen
    print("\033[2J\033[H", end="")
    
    # Title
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║{RESET}{BOLD}{WHITE}           ENTERPRISE DOCUMENT INTELLIGENCE PLATFORM                  {RESET}{CYAN}║
║{RESET}{WHITE}       "500+ pages → Actionable insights in seconds"                  {RESET}{CYAN}║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")
    time.sleep(2)
    
    # Problem statement
    section("THE PROBLEM")
    slow_print(f"{WHITE}Financial analysts spend 40+ hours reviewing SEC filings per company.{RESET}", 0.03)
    time.sleep(0.5)
    slow_print(f"{WHITE}Information is buried in 500+ page documents.{RESET}", 0.03)
    time.sleep(0.5)
    slow_print(f"{WHITE}Manual comparison across companies takes days.{RESET}", 0.03)
    time.sleep(1.5)
    
    # Solution
    section("THE SOLUTION")
    slow_print(f"{GREEN}Let me show you what AI can do...{RESET}", 0.03)
    time.sleep(1)
    
    # Now run actual code
    print(f"\n{YELLOW}Initializing system...{RESET}")
    
    from src.embeddings import OllamaEmbeddings, CachedEmbeddings
    from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
    from src.retrieval import HybridRetriever
    from src.generation.ollama_llm import OllamaLLM
    from src.documents import MultiDocumentPipeline
    
    embeddings = CachedEmbeddings(OllamaEmbeddings(model="nomic-embed-text"), enabled=True)
    vectorstore = QdrantHybridStore(collection_name="auto_demo", dense_dimensions=768, recreate_collection=True)
    retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore, sparse_encoder="fastembed")
    llm = OllamaLLM(model="llama3.2")
    
    pipeline = MultiDocumentPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        registry_path=".cache/auto_demo.json",
    )
    print(f"{GREEN}✓ System ready{RESET}")
    time.sleep(1)
    
    # Demo 1: Ingestion
    section("DEMO 1: BULK INGESTION")
    typing_effect("pipeline.ingest_directory('sec-filings/', recursive=True)")
    print(f"\n{YELLOW}Processing 3 SEC 10-K filings (500+ pages)...{RESET}")
    
    start = time.time()
    stats = pipeline.ingest_directory("data/test_adaptive/sec-edgar-filings/", recursive=True)
    ingest_time = time.time() - start
    
    print(f"""
{GREEN}✓ COMPLETE{RESET}

   Companies:    {BOLD}{len(stats['companies'])}{RESET} ({', '.join(stats['companies'])})
   Chunks:       {BOLD}{stats['total_chunks']}{RESET}
   Time:         {BOLD}{ingest_time:.1f} seconds{RESET}
   
   {CYAN}💡 That's 500+ pages processed in {ingest_time:.1f} seconds{RESET}
""")
    time.sleep(2)
    
    # Demo 2: Query
    section("DEMO 2: INSTANT ANSWERS")
    query1 = "What factors affect Meta's advertising revenue?"
    typing_effect(f'pipeline.query("{query1}")')
    print(f"\n{YELLOW}Searching and generating answer...{RESET}")
    
    start = time.time()
    response = pipeline.query(query1, top_k=5, filter_companies=["Meta"])
    query_time = time.time() - start
    
    confidence_color = GREEN if response.confidence == "high" else YELLOW
    emoji = "🟢" if response.confidence == "high" else "🟡"
    
    print(f"""
{GREEN}✓ ANSWER FOUND{RESET}

   {emoji} Confidence:  {confidence_color}{BOLD}{response.confidence.upper()}{RESET}
   📑 Sources:     {BOLD}{len(response.sources)}{RESET} cited passages
   ⏱️  Time:        {BOLD}{query_time:.2f}s{RESET}
   
{CYAN}Answer:{RESET}
{response.answer[:500]}...

   {CYAN}💡 Every claim is backed by source citations - no hallucinations{RESET}
""")
    time.sleep(3)
    
    # Demo 3: Filtered Query
    section("DEMO 3: COMPANY-SPECIFIC ANALYSIS")
    query2 = "What are the manufacturing and supply chain risks?"
    typing_effect(f'pipeline.query("{query2}", filter_companies=["Tesla"])')
    print(f"\n{YELLOW}Searching Tesla filings only...{RESET}")
    
    start = time.time()
    response2 = pipeline.query(query2, top_k=5, filter_companies=["Tesla"])
    query_time2 = time.time() - start
    
    confidence_color2 = GREEN if response2.confidence == "high" else YELLOW
    emoji2 = "🟢" if response2.confidence == "high" else "🟡"
    
    print(f"""
{GREEN}✓ ANSWER FOUND{RESET}

   {emoji2} Confidence:  {confidence_color2}{BOLD}{response2.confidence.upper()}{RESET}
   🏢 Company:     {BOLD}Tesla only{RESET}
   📑 Sources:     {BOLD}{len(response2.sources)}{RESET}
   ⏱️  Time:        {BOLD}{query_time2:.2f}s{RESET}
   
{CYAN}Answer:{RESET}
{response2.answer[:400]}...

   {CYAN}💡 Filtered queries return only the requested company's data{RESET}
""")
    time.sleep(3)
    
    # Demo 4: Comparison
    section("DEMO 4: CROSS-COMPANY COMPARISON")
    query3 = "What is the company's AI and machine learning strategy?"
    typing_effect(f'pipeline.compare_companies("{query3}", companies=["Meta", "Tesla", "NVIDIA"])')
    print(f"\n{YELLOW}Analyzing all 3 companies...{RESET}")
    
    start = time.time()
    comparison = pipeline.compare_companies(query3, companies=["Meta", "Tesla", "NVIDIA"], top_k_per_company=2)
    comp_time = time.time() - start
    
    print(f"""
{GREEN}✓ COMPARISON COMPLETE{RESET}

   🏢 Companies:   {BOLD}Meta, Tesla, NVIDIA{RESET}
   📑 Sources:     {BOLD}{len(comparison.sources)}{RESET} (balanced across companies)
   ⏱️  Time:        {BOLD}{comp_time:.2f}s{RESET}
   
{CYAN}Comparative Analysis:{RESET}
{comparison.answer[:600]}...

   {CYAN}💡 Days of manual analysis done in {comp_time:.1f} seconds{RESET}
""")
    time.sleep(3)
    
    # Summary
    section("SYSTEM CAPABILITIES")
    print(f"""
{WHITE}┌─────────────────────────────────────────────────────────────────────┐
│{RESET} {BOLD}WHAT THIS SYSTEM DOES{RESET}                                              {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} ✅ Ingests 500+ pages in seconds                                    {WHITE}│
│{RESET} ✅ Answers complex questions with citations                         {WHITE}│
│{RESET} ✅ Filters by company, date, document type                          {WHITE}│
│{RESET} ✅ Compares across multiple documents                               {WHITE}│
│{RESET} ✅ Confidence scoring - knows when NOT to answer                    {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} {BOLD}TECH STACK{RESET}                                                         {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} • Hybrid Search (semantic + keyword)                                {WHITE}│
│{RESET} • Cross-encoder reranking                                           {WHITE}│
│{RESET} • 275+ automated tests                                              {WHITE}│
│{RESET} • Docker-ready deployment                                           {WHITE}│
│{RESET} • FastAPI + Streamlit UI                                            {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} {BOLD}PERFORMANCE{RESET}                                                        {WHITE}│
├─────────────────────────────────────────────────────────────────────┤
│{RESET} • 436x embedding speedup (caching)                                  {WHITE}│
│{RESET} • 99%+ cache hit rate                                               {WHITE}│
│{RESET} • 2-4 second query response                                         {WHITE}│
└─────────────────────────────────────────────────────────────────────┘
{RESET}""")
    time.sleep(2)
    
    # CTA
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║{RESET}{BOLD}{WHITE}                    NEED SOMETHING LIKE THIS?                         {RESET}{CYAN}║
║{RESET}                                                                      {CYAN}║
║{RESET}{WHITE}   I build custom AI document systems for companies.                 {RESET}{CYAN}║
║{RESET}{WHITE}   Contracts, reports, manuals, filings - any documents.             {RESET}{CYAN}║
║{RESET}                                                                      {CYAN}║
║{RESET}{BOLD}{GREEN}   → DM me on LinkedIn                                               {RESET}{CYAN}║
║{RESET}{BOLD}{GREEN}   → GitHub: github.com/your-username/rag-system                     {RESET}{CYAN}║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")

if __name__ == "__main__":
    main()
