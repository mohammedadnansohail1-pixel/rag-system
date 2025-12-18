#!/usr/bin/env python3
"""
Generate AI voiceover and create synced demo script.
Calculates audio duration and matches pauses.
"""

import asyncio
import os
import json
from mutagen.mp3 import MP3

# Create directories
os.makedirs("demo_audio", exist_ok=True)

# All voiceover segments with section markers
VOICEOVER_SEGMENTS = {
    # HOOK
    "01_hook": {
        "section": "hook",
        "text": "What if you could search 500 pages of documents and get accurate answers in just 2 seconds? I built an AI-powered document intelligence system that does exactly that. Let me show you how it works.",
        "after": "title"
    },
    
    # PROBLEM
    "02_problem_intro": {
        "section": "problem",
        "text": "Every day, knowledge workers waste hours searching through documents.",
        "after": "header"
    },
    "03_problem_analysts": {
        "section": "problem",
        "text": "Financial analysts spend 40 or more hours reviewing SEC filings for a single company. Information is buried in 500 page documents. Cross-company comparison takes days of manual work.",
        "after": "pain_points"
    },
    "04_problem_ai": {
        "section": "problem",
        "text": "And when people try AI chatbots, they face hallucinations where AI makes up facts, no source citations, mixing up information between documents, and using outdated training data instead of your actual documents.",
        "after": "ai_problems"
    },
    
    # SOLUTION
    "05_solution": {
        "section": "solution",
        "text": "I built a production-grade RAG system that solves all of this. Accurate answers grounded in your actual documents. Source citations so you can verify every claim. Confidence scoring that tells you when to trust versus verify. Multi-document search and comparison. 500 pages ingested in 2 seconds, queries answered in 2 to 4 seconds. And it runs completely locally. No data leaves your machine, no API costs.",
        "after": "features"
    },
    
    # ARCHITECTURE
    "06_arch_intro": {
        "section": "architecture",
        "text": "Let me walk you through the architecture. This is production-ready from day one.",
        "after": "header"
    },
    "07_arch_explain": {
        "section": "architecture",
        "text": "First, documents go through processing. Loaders handle PDFs, markdown, SEC filings. Structure-aware chunking preserves document hierarchy. Then enrichment adds metadata. Next, hybrid retrieval combines semantic search with BM25 keyword search using Reciprocal Rank Fusion. A cross-encoder reranker then precisely scores each query-document pair. Guardrails assess confidence - high, medium, or low - and the system refuses to answer if evidence is weak. Finally, the LLM generates an answer with source citations from the retrieved context only.",
        "after": "diagram"
    },
    
    # KEY DECISIONS
    "08_decision_hybrid": {
        "section": "decisions",
        "text": "Why hybrid search? Pure semantic search misses exact keywords. Pure keyword search misses meaning and context. Hybrid combines both for 30% better retrieval accuracy.",
        "after": "hybrid"
    },
    "09_decision_confidence": {
        "section": "decisions",
        "text": "Why confidence scoring? Large language models hallucinate when context is weak. By scoring retrieval quality before generation, we can warn users or refuse to answer. Green means trust it. Yellow means verify important claims. Red means human review needed.",
        "after": "confidence"
    },
    "10_decision_chunking": {
        "section": "decisions",
        "text": "Why structure-aware chunking? Fixed-size chunking destroys document structure. SEC filings have Items, Sections, Parts. By preserving hierarchy, we get 96% noise reduction.",
        "after": "chunking"
    },
    "11_decision_caching": {
        "section": "decisions",
        "text": "Why aggressive caching? Embedding generation is slow. Same content gets re-embedded repeatedly. Our cache gives 436x speedup on embeddings and 15,000x on queries.",
        "after": "caching"
    },
    
    # LIVE DEMO
    "12_demo_intro": {
        "section": "demo",
        "text": "Let's see it in action with real SEC filings from Meta, Tesla, and NVIDIA.",
        "after": "header"
    },
    "13_demo_init": {
        "section": "demo",
        "text": "Initializing the system. This runs entirely locally using Ollama and Qdrant.",
        "after": "init"
    },
    "14_demo_ingest": {
        "section": "demo",
        "text": "Loading 3 complete 10-K filings. That's over 500 pages of financial disclosures.",
        "after": "ingest_start"
    },
    "15_demo_ingest_done": {
        "section": "demo",
        "text": "Done. 531 chunks indexed in just over 2 seconds. Every section, every risk factor, now searchable.",
        "after": "ingest_done"
    },
    "16_demo_query1": {
        "section": "demo",
        "text": "Let's ask about factors affecting Meta's advertising revenue.",
        "after": "query1_start"
    },
    "17_demo_query1_result": {
        "section": "demo",
        "text": "Answer in under 5 seconds. HIGH confidence means strong evidence. 5 source citations. Every claim backed by actual passages from the filing. No hallucinations.",
        "after": "query1_result"
    },
    "18_demo_query2": {
        "section": "demo",
        "text": "Now filtering to just Tesla. Asking about manufacturing and supply chain risks.",
        "after": "query2_start"
    },
    "19_demo_query2_result": {
        "section": "demo",
        "text": "Only Tesla's filing was searched. MEDIUM confidence - the system is being honest about evidence strength. No data mixing between companies.",
        "after": "query2_result"
    },
    "20_demo_compare": {
        "section": "demo",
        "text": "Here's where it gets powerful. Comparing AI and machine learning strategies across all three companies.",
        "after": "compare_start"
    },
    "21_demo_compare_result": {
        "section": "demo",
        "text": "Structured comparison in under 5 seconds. Similarities, differences, specific initiatives from each company. What would take days of manual analysis, done instantly.",
        "after": "compare_result"
    },
    
    # USE CASES
    "22_usecases": {
        "section": "usecases",
        "text": "This same system works for any document-heavy domain. Financial services for SEC filings and due diligence. Legal for contract review and compliance. HR for policy chatbots and onboarding. Engineering for documentation search. Research for literature review and paper analysis. Same core system, different documents.",
        "after": "list"
    },
    
    # TECH STACK
    "23_techstack": {
        "section": "techstack",
        "text": "The tech stack includes Ollama for embeddings and LLM, Qdrant for hybrid vector storage, FastEmbed for BM25, cross-encoder for reranking, FastAPI backend, Streamlit UI, and Docker deployment. 275 automated tests. 436x embedding cache speedup. 96% noise reduction versus naive chunking.",
        "after": "stats"
    },
    
    # CLOSING
    "24_closing": {
        "section": "closing",
        "text": "I build custom AI document systems for companies. You have contracts, I build instant clause extraction. You have policies, I build employee self-service. You have technical docs, I build developer assistants. Production-ready code with 275 tests and Docker deployment. Check out the GitHub repo linked below. If you have documents your team spends hours searching through, let's talk. Thanks for watching.",
        "after": "cta"
    },
}

async def generate_audio(text, filename, voice="en-US-GuyNeural"):
    """Generate audio file from text."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    filepath = f"demo_audio/{filename}.mp3"
    await communicate.save(filepath)
    return filepath

def get_audio_duration(filepath):
    """Get duration of audio file in seconds."""
    audio = MP3(filepath)
    return audio.info.length

async def main():
    print("=" * 60)
    print("GENERATING AI VOICEOVER FILES")
    print("=" * 60)
    print()
    
    timing_data = {}
    total_duration = 0
    
    for filename, data in VOICEOVER_SEGMENTS.items():
        # Generate audio
        filepath = await generate_audio(data["text"], filename)
        
        # Get duration
        duration = get_audio_duration(filepath)
        total_duration += duration
        
        timing_data[filename] = {
            "section": data["section"],
            "after": data["after"],
            "duration": round(duration, 2),
            "text": data["text"][:50] + "..."
        }
        
        print(f"✓ {filename}.mp3 | {duration:.2f}s | {data['section']}")
    
    # Save timing data
    with open("demo_audio/timing.json", "w") as f:
        json.dump(timing_data, f, indent=2)
    
    print()
    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print()
    
    # Group by section
    sections = {}
    for filename, data in timing_data.items():
        section = data["section"]
        if section not in sections:
            sections[section] = 0
        sections[section] += data["duration"]
    
    print(f"{'Section':<20} {'Duration':<15}")
    print("-" * 35)
    for section, duration in sections.items():
        mins = int(duration // 60)
        secs = duration % 60
        print(f"{section:<20} {mins}:{secs:05.2f}")
    
    print("-" * 35)
    total_mins = int(total_duration // 60)
    total_secs = total_duration % 60
    print(f"{'TOTAL':<20} {total_mins}:{total_secs:05.2f}")
    
    print()
    print(f"✅ Generated {len(VOICEOVER_SEGMENTS)} audio files")
    print(f"✅ Timing data saved to demo_audio/timing.json")
    print()
    print("Audio files in demo_audio/:")
    for f in sorted(os.listdir("demo_audio")):
        if f.endswith(".mp3"):
            print(f"  - {f}")

if __name__ == "__main__":
    asyncio.run(main())
