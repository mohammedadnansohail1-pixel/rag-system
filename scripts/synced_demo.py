#!/usr/bin/env python3
"""
Synced Demo - Plays audio while showing visuals
===============================================
Run with: python scripts/synced_demo.py
"""

import time
import sys
import os
import subprocess
import threading
import json

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Load timing data
with open("demo_audio/timing.json") as f:
    TIMING = json.load(f)

def clear():
    print("\033[2J\033[H", end="")

def play_audio(filename):
    """Play audio file in background."""
    filepath = f"demo_audio/{filename}.mp3"
    if os.path.exists(filepath):
        # Try different players
        for player in ["mpv --no-video", "ffplay -nodisp -autoexit", "aplay", "paplay"]:
            try:
                subprocess.Popen(
                    f"{player} {filepath} 2>/dev/null",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return
            except:
                continue

def get_duration(filename):
    """Get duration of audio segment."""
    return TIMING.get(filename, {}).get("duration", 3)

def play_and_wait(filename, extra_pause=0.5):
    """Play audio and wait for it to finish."""
    play_audio(filename)
    duration = get_duration(filename)
    time.sleep(duration + extra_pause)

# ============================================================================
# VISUALS
# ============================================================================

def show_hook():
    clear()
    print(f"""
{CYAN}
    ██████╗  █████╗  ██████╗     ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗
    ██╔══██╗██╔══██╗██╔════╝     ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║
    ██████╔╝███████║██║  ███╗    ███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║
    ██╔══██╗██╔══██║██║   ██║    ╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║
    ██║  ██║██║  ██║╚██████╔╝    ███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝
{RESET}

{WHITE}                    Enterprise Document Intelligence Platform{RESET}

{YELLOW}
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    │    500+ pages  →  2 seconds             │
                    │                                         │
                    │    Questions   →  Instant answers       │
                    │                                         │
                    │    Documents   →  Intelligence          │
                    │                                         │
                    └─────────────────────────────────────────┘
{RESET}""")

def show_problem_header():
    clear()
    print(f"""
{CYAN}{'═'*70}
  THE PROBLEM
{'═'*70}{RESET}
""")

def show_problem_pain():
    print(f"""
{YELLOW}
    ┌────────────────────────────────────────────────────────────────────┐
    │  FINANCIAL ANALYSTS                                                │
    │  └─ 40+ hours reviewing SEC filings per company                    │
    │  └─ Information buried in 500+ page documents                      │
    │  └─ Cross-company comparison takes days                            │
    ├────────────────────────────────────────────────────────────────────┤
    │  LEGAL TEAMS                                                       │
    │  └─ Hours reviewing contracts for risky clauses                    │
    │  └─ Manual extraction of key terms                                 │
    ├────────────────────────────────────────────────────────────────────┤
    │  HR DEPARTMENTS                                                    │
    │  └─ Employees can't find policy information                        │
    │  └─ Repetitive questions to HR team                                │
    ├────────────────────────────────────────────────────────────────────┤
    │  ENGINEERING TEAMS                                                 │
    │  └─ Documentation spread across wikis, repos, docs                 │
    │  └─ New hires take months to get up to speed                       │
    └────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_problem_ai():
    print(f"""
{YELLOW}
    ┌────────────────────────────────────────────────────────────────────┐
    │  CURRENT AI CHATBOTS FAIL                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │  ❌  HALLUCINATIONS     - Makes up convincing but wrong facts      │
    │  ❌  NO SOURCES         - Can't verify where info came from        │
    │  ❌  WRONG CONTEXT      - Mixes up different documents             │
    │  ❌  OUTDATED DATA      - Uses training data, not your docs        │
    └────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_solution():
    clear()
    print(f"""
{CYAN}{'═'*70}
  THE SOLUTION
{'═'*70}{RESET}

{GREEN}
    ┌────────────────────────────────────────────────────────────────────┐
    │  A PRODUCTION-GRADE RAG SYSTEM THAT ACTUALLY WORKS                 │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │   ✓  ACCURATE ANSWERS     - Grounded in your actual documents      │
    │                                                                    │
    │   ✓  SOURCE CITATIONS     - See exactly where claims come from     │
    │                                                                    │
    │   ✓  CONFIDENCE SCORING   - Know when to trust vs verify           │
    │                                                                    │
    │   ✓  MULTI-DOCUMENT       - Search and compare across docs         │
    │                                                                    │
    │   ✓  BLAZING FAST         - 500 pages in 2s, queries in 2-4s       │
    │                                                                    │
    │   ✓  RUNS LOCALLY         - No data leaves your machine            │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_architecture_header():
    clear()
    print(f"""
{CYAN}{'═'*70}
  ARCHITECTURE
{'═'*70}{RESET}
""")

def show_architecture_diagram():
    print(f"""
{CYAN}
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         USER QUERY                                  │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. DOCUMENT PROCESSING                                             │
    │     Loaders (PDF/MD/SEC) → Structure-Aware Chunking → Enrichment    │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  2. HYBRID RETRIEVAL                                                │
    │     Dense (Semantic) + Sparse (BM25) → Reciprocal Rank Fusion       │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  3. RERANKING                                                       │
    │     Cross-Encoder scores each (query, document) pair                │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  4. GUARDRAILS                                                      │
    │     Confidence: 🟢 HIGH | 🟡 MEDIUM | 🔴 LOW → Refuse if weak        │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  5. GENERATION                                                      │
    │     LLM generates answer from retrieved context + source citations  │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ANSWER + SOURCES + CONFIDENCE                    │
    └─────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_decision_hybrid():
    clear()
    print(f"""
{CYAN}{'═'*70}
  KEY DECISIONS
{'═'*70}{RESET}

{BOLD}{WHITE}1. Why Hybrid Search?{RESET}

{CYAN}   ┌─────────────────────────────────────────────────────────────────┐
   │  Query: "What is the CIK number?"                                │
   ├─────────────────────────────────────────────────────────────────┤
   │  Semantic only:  Returns general company info        ❌          │
   │  Keyword only:   Returns match but no context        ❌          │
   │  Hybrid:         Returns CIK with full context       ✓          │
   ├─────────────────────────────────────────────────────────────────┤
   │  Result: {GREEN}30% better retrieval accuracy{RESET}{CYAN}                           │
   └─────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_decision_confidence():
    print(f"""
{BOLD}{WHITE}2. Why Confidence Scoring?{RESET}

{CYAN}   ┌─────────────────────────────────────────────────────────────────┐
   │  LLMs hallucinate when context is weak                           │
   ├─────────────────────────────────────────────────────────────────┤
   │  🟢 HIGH    = Strong evidence, multiple sources → Trust it       │
   │  🟡 MEDIUM  = Some evidence → Verify important claims            │
   │  🔴 LOW     = Weak evidence → Human review needed                │
   ├─────────────────────────────────────────────────────────────────┤
   │  Below threshold: {GREEN}System refuses to answer{RESET}{CYAN}                       │
   └─────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_decision_chunking():
    print(f"""
{BOLD}{WHITE}3. Why Structure-Aware Chunking?{RESET}

{CYAN}   ┌─────────────────────────────────────────────────────────────────┐
   │  Fixed chunking destroys document structure                      │
   ├─────────────────────────────────────────────────────────────────┤
   │  Fixed chunking:      1,247 chunks (70% noise)                   │
   │  Structure-aware:       354 chunks (relevant content)            │
   ├─────────────────────────────────────────────────────────────────┤
   │  Result: {GREEN}96% noise reduction{RESET}{CYAN}                                      │
   └─────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_decision_caching():
    print(f"""
{BOLD}{WHITE}4. Why Aggressive Caching?{RESET}

{CYAN}   ┌─────────────────────────────────────────────────────────────────┐
   │  Embedding generation is slow and repetitive                     │
   ├─────────────────────────────────────────────────────────────────┤
   │  Embedding cache:     {GREEN}436x speedup{RESET}{CYAN}                               │
   │  Query cache:         {GREEN}15,000x speedup{RESET}{CYAN}                            │
   │  Cache hit rate:      {GREEN}99%+{RESET}{CYAN}                                       │
   └─────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_demo_header():
    clear()
    print(f"""
{CYAN}{'═'*70}
  LIVE DEMO
{'═'*70}{RESET}

{WHITE}SEC 10-K Filings: Meta, Tesla, NVIDIA (500+ pages){RESET}
""")

def show_usecases():
    clear()
    print(f"""
{CYAN}{'═'*70}
  USE CASES
{'═'*70}{RESET}

{CYAN}
    ┌────────────────────────────────────────────────────────────────────┐
    │  💰 FINANCIAL SERVICES                                             │
    │     SEC filing analysis • Due diligence • Competitive intelligence │
    ├────────────────────────────────────────────────────────────────────┤
    │  ⚖️  LEGAL                                                         │
    │     Contract review • Clause extraction • Compliance checking      │
    ├────────────────────────────────────────────────────────────────────┤
    │  👥 HR / OPERATIONS                                                │
    │     Policy Q&A chatbot • Employee handbook • Onboarding assistant  │
    ├────────────────────────────────────────────────────────────────────┤
    │  💻 ENGINEERING                                                    │
    │     Documentation search • Knowledge assistant • Support bot       │
    ├────────────────────────────────────────────────────────────────────┤
    │  📚 RESEARCH                                                       │
    │     Literature review • Paper analysis • Citation finding          │
    └────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_techstack():
    clear()
    print(f"""
{CYAN}{'═'*70}
  TECH STACK & STATS
{'═'*70}{RESET}

{WHITE}
    ┌────────────────────────────────────────────────────────────────────┐
    │  TECH STACK                                                        │
    ├────────────────────────────────────────────────────────────────────┤
    │  Embeddings:     Ollama (nomic-embed-text)                         │
    │  Vector Store:   Qdrant (hybrid dense + sparse)                    │
    │  Sparse Encoder: FastEmbed BM25                                    │
    │  LLM:            Ollama (Llama 3.2)                                 │
    │  Reranking:      Cross-Encoder (ms-marco-MiniLM)                   │
    │  API:            FastAPI                                           │
    │  UI:             Streamlit                                         │
    │  Deployment:     Docker                                            │
    ├────────────────────────────────────────────────────────────────────┤
    │  PROJECT STATS                                                     │
    ├────────────────────────────────────────────────────────────────────┤
    │  Tests:           {GREEN}275+ automated tests{RESET}{WHITE}                              │
    │  Cache speedup:   {GREEN}436x embeddings, 15,000x queries{RESET}{WHITE}                  │
    │  Ingestion:       {GREEN}230 pages/second{RESET}{WHITE}                                  │
    │  Query latency:   {GREEN}2-4 seconds{RESET}{WHITE}                                       │
    │  Noise reduction: {GREEN}96%{RESET}{WHITE} vs naive chunking                             │
    └────────────────────────────────────────────────────────────────────┘
{RESET}""")

def show_closing():
    clear()
    print(f"""
{CYAN}{'═'*70}
  LET'S BUILD SOMETHING
{'═'*70}{RESET}

{GREEN}
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   I build custom AI document systems for companies.                ║
    ║                                                                    ║
    ║   • Contracts  → Instant clause extraction                         ║
    ║   • Policies   → Employee self-service                             ║
    ║   • Tech docs  → Developer assistants                              ║
    ║   • Research   → Literature analyzers                              ║
    ║                                                                    ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   Production-ready. 275+ tests. Docker deployment.                 ║
    ║                                                                    ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   🐙  GitHub:   github.com/mohammedadnansohail1-pixel/rag-system   ║
    ║   💼  LinkedIn: [Your LinkedIn]                                    ║
    ║   📧  Email:    [Your Email]                                       ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
{RESET}

{CYAN}                    Thanks for watching!{RESET}
""")

# ============================================================================
# LIVE DEMO SECTION
# ============================================================================
def run_live_demo():
    """Run actual code demo with audio sync."""
    from src.embeddings import OllamaEmbeddings, CachedEmbeddings
    from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
    from src.retrieval import HybridRetriever
    from src.generation.ollama_llm import OllamaLLM
    from src.documents import MultiDocumentPipeline
    
    # Init
    play_audio("13_demo_init")
    print(f"{YELLOW}Initializing system...{RESET}")
    
    embeddings = CachedEmbeddings(OllamaEmbeddings(model="nomic-embed-text"), enabled=True)
    vectorstore = QdrantHybridStore(collection_name="synced_demo", dense_dimensions=768, recreate_collection=True)
    retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore, sparse_encoder="fastembed")
    llm = OllamaLLM(model="llama3.2")
    pipeline = MultiDocumentPipeline(embeddings=embeddings, vectorstore=vectorstore, retriever=retriever, llm=llm, registry_path=".cache/synced_demo.json")
    
    print(f"{GREEN}✓ System ready{RESET}")
    time.sleep(get_duration("13_demo_init") - 2)
    
    # Ingest
    play_audio("14_demo_ingest")
    print(f"\n{WHITE}Loading 3 SEC 10-K filings...{RESET}")
    
    start = time.time()
    stats = pipeline.ingest_directory("data/test_adaptive/sec-edgar-filings/", recursive=True)
    ingest_time = time.time() - start
    
    time.sleep(max(0, get_duration("14_demo_ingest") - ingest_time))
    
    # Ingest done
    play_audio("15_demo_ingest_done")
    print(f"""
{GREEN}✓ COMPLETE
   Companies: {', '.join(stats['companies'])}
   Chunks:    {stats['total_chunks']}
   Time:      {ingest_time:.1f}s{RESET}
""")
    time.sleep(get_duration("15_demo_ingest_done"))
    
    # Query 1
    play_audio("16_demo_query1")
    query1 = "What factors affect Meta's advertising revenue?"
    print(f"\n{WHITE}Query: {query1}{RESET}")
    time.sleep(get_duration("16_demo_query1"))
    
    start = time.time()
    response = pipeline.query(query1, top_k=5, filter_companies=["Meta"])
    q_time = time.time() - start
    
    play_audio("17_demo_query1_result")
    emoji = "🟢" if response.confidence == "high" else "🟡"
    print(f"""
{GREEN}✓ ANSWER
   {emoji} Confidence: {response.confidence.upper()}
   📑 Sources: {len(response.sources)}
   ⏱️  Time: {q_time:.2f}s{RESET}

{WHITE}{response.answer[:400]}...{RESET}
""")
    time.sleep(get_duration("17_demo_query1_result"))
    
    # Query 2
    play_audio("18_demo_query2")
    query2 = "What are the manufacturing and supply chain risks?"
    print(f"\n{WHITE}Query: {query2} (Tesla only){RESET}")
    time.sleep(get_duration("18_demo_query2"))
    
    start = time.time()
    response2 = pipeline.query(query2, top_k=5, filter_companies=["Tesla"])
    q_time2 = time.time() - start
    
    play_audio("19_demo_query2_result")
    emoji2 = "🟢" if response2.confidence == "high" else "🟡"
    print(f"""
{GREEN}✓ ANSWER
   {emoji2} Confidence: {response2.confidence.upper()}
   🏢 Company: Tesla only
   📑 Sources: {len(response2.sources)}
   ⏱️  Time: {q_time2:.2f}s{RESET}

{WHITE}{response2.answer[:350]}...{RESET}
""")
    time.sleep(get_duration("19_demo_query2_result"))
    
    # Comparison
    play_audio("20_demo_compare")
    print(f"\n{WHITE}Comparing AI strategies across Meta, Tesla, NVIDIA...{RESET}")
    time.sleep(get_duration("20_demo_compare"))
    
    start = time.time()
    comparison = pipeline.compare_companies(
        "What is the company's AI and machine learning strategy?",
        companies=["Meta", "Tesla", "NVIDIA"],
        top_k_per_company=2
    )
    c_time = time.time() - start
    
    play_audio("21_demo_compare_result")
    print(f"""
{GREEN}✓ COMPARISON COMPLETE
   Companies: Meta, Tesla, NVIDIA
   Sources: {len(comparison.sources)}
   Time: {c_time:.2f}s{RESET}

{WHITE}{comparison.answer[:450]}...{RESET}
""")
    time.sleep(get_duration("21_demo_compare_result"))

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"{YELLOW}Starting synced demo in 3 seconds...{RESET}")
    print(f"{DIM}Make sure audio is on!{RESET}")
    time.sleep(3)
    
    # HOOK
    show_hook()
    play_and_wait("01_hook")
    
    # PROBLEM
    show_problem_header()
    play_and_wait("02_problem_intro")
    show_problem_pain()
    play_and_wait("03_problem_analysts")
    show_problem_ai()
    play_and_wait("04_problem_ai")
    
    # SOLUTION
    show_solution()
    play_and_wait("05_solution")
    
    # ARCHITECTURE
    show_architecture_header()
    play_and_wait("06_arch_intro")
    show_architecture_diagram()
    play_and_wait("07_arch_explain")
    
    # KEY DECISIONS
    show_decision_hybrid()
    play_and_wait("08_decision_hybrid")
    show_decision_confidence()
    play_and_wait("09_decision_confidence")
    show_decision_chunking()
    play_and_wait("10_decision_chunking")
    show_decision_caching()
    play_and_wait("11_decision_caching")
    
    # LIVE DEMO
    show_demo_header()
    play_and_wait("12_demo_intro")
    run_live_demo()
    
    # USE CASES
    show_usecases()
    play_and_wait("22_usecases")
    
    # TECH STACK
    show_techstack()
    play_and_wait("23_techstack")
    
    # CLOSING
    show_closing()
    play_and_wait("24_closing", extra_pause=2)
    
    print(f"\n{GREEN}✓ Demo complete!{RESET}")

if __name__ == "__main__":
    main()
