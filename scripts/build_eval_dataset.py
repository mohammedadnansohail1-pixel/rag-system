"""
Build comprehensive evaluation dataset from Netflix 10-K.

Creates Q&A pairs with relevant chunk IDs for systematic testing.
Uses stable MD5 hashes for consistent IDs across sessions.
"""
import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import SECLoader
from src.chunkers.factory import ChunkerFactory


def stable_hash(text: str) -> str:
    """Generate stable hash that persists across Python sessions."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def main():
    print("Building evaluation dataset from Netflix 10-K...")
    
    # Load and chunk the document
    loader = SECLoader(download_dir='data/test_docs')
    docs = loader.load(
        'data/test_docs/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt'
    )
    
    chunker = ChunkerFactory.from_config({
        'strategy': 'recursive',
        'chunk_size': 1000,
        'chunk_overlap': 200,
    })
    
    chunks = chunker.chunk(docs[0])
    
    # Clean chunks
    clean_chunks = []
    for chunk in chunks:
        content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
        if len(content) > 100:
            clean_chunks.append(content)
    
    print(f"Created {len(clean_chunks)} chunks")
    
    # Create chunk ID mapping (stable MD5 hash)
    chunk_ids = {stable_hash(c): c for c in clean_chunks}
    
    # Find relevant chunks for each question
    def find_relevant_chunks(keywords, max_chunks=3):
        """Find chunks containing keywords."""
        relevant = []
        keywords_lower = [k.lower() for k in keywords]
        
        for chunk_id, content in chunk_ids.items():
            content_lower = content.lower()
            if all(kw in content_lower for kw in keywords_lower):
                relevant.append(chunk_id)
                if len(relevant) >= max_chunks:
                    break
        
        return relevant
    
    # Define Q&A pairs with search keywords
    qa_pairs = [
        # Revenue questions
        {
            "question": "What was Netflix's total streaming revenue in 2024?",
            "keywords": ["streaming revenues", "39,000"],
            "category": "financial"
        },
        {
            "question": "How did Netflix revenue change from 2023 to 2024?",
            "keywords": ["streaming revenues", "2024", "2023"],
            "category": "financial"
        },
        
        # Subscriber questions
        {
            "question": "How many paid memberships does Netflix have?",
            "keywords": ["paid memberships", "million"],
            "category": "subscribers"
        },
        {
            "question": "What was Netflix's subscriber growth in 2024?",
            "keywords": ["paid net membership additions"],
            "category": "subscribers"
        },
        {
            "question": "What is the average revenue per membership?",
            "keywords": ["average monthly revenue per", "membership"],
            "category": "subscribers"
        },
        
        # Content & Strategy questions
        {
            "question": "What is Netflix's content strategy?",
            "keywords": ["content", "original", "programming"],
            "category": "content"
        },
        {
            "question": "Does Netflix produce original content?",
            "keywords": ["original", "content", "productions"],
            "category": "content"
        },
        
        # Risk factors
        {
            "question": "What are the main risk factors for Netflix?",
            "keywords": ["risk factors", "competition"],
            "category": "risk"
        },
        {
            "question": "What are Netflix's cybersecurity risks?",
            "keywords": ["cybersecurity", "security", "risk"],
            "category": "risk"
        },
        
        # Operations
        {
            "question": "What is Netflix's operating margin?",
            "keywords": ["operating margin", "percent"],
            "category": "operations"
        },
        {
            "question": "How many employees does Netflix have?",
            "keywords": ["employees", "full-time"],
            "category": "operations"
        },
        {
            "question": "Where is Netflix headquartered?",
            "keywords": ["headquarters", "los gatos"],
            "category": "operations"
        },
        
        # Technology
        {
            "question": "What technology infrastructure does Netflix use?",
            "keywords": ["aws", "amazon web services", "cloud"],
            "category": "technology"
        },
        {
            "question": "How does Netflix deliver streaming content?",
            "keywords": ["content delivery", "streaming", "network"],
            "category": "technology"
        },
        
        # Leadership
        {
            "question": "Who are Netflix's executive officers?",
            "keywords": ["executive officers", "chief"],
            "category": "leadership"
        },
        {
            "question": "Who is on Netflix's board of directors?",
            "keywords": ["board of directors", "director"],
            "category": "leadership"
        },
        
        # Financial metrics
        {
            "question": "What was Netflix's net income in 2024?",
            "keywords": ["net income", "2024"],
            "category": "financial"
        },
        {
            "question": "What is Netflix's total debt?",
            "keywords": ["debt", "notes", "billion"],
            "category": "financial"
        },
        {
            "question": "What was Netflix's free cash flow?",
            "keywords": ["free cash flow"],
            "category": "financial"
        },
        
        # Ad-supported tier
        {
            "question": "Does Netflix have an ad-supported plan?",
            "keywords": ["advertising", "ad-supported"],
            "category": "products"
        },
    ]
    
    # Build dataset
    test_cases = []
    found_count = 0
    
    for qa in qa_pairs:
        relevant = find_relevant_chunks(qa["keywords"])
        
        if relevant:
            found_count += 1
            test_cases.append({
                "query": qa["question"],
                "relevant_ids": relevant,
                "relevance_scores": {rid: 1.0 for rid in relevant},
                "metadata": {
                    "category": qa["category"],
                    "keywords": qa["keywords"],
                }
            })
            print(f"  ✓ {qa['question'][:50]}... ({len(relevant)} chunks)")
        else:
            print(f"  ✗ {qa['question'][:50]}... (no chunks found)")
    
    print(f"\nFound relevant chunks for {found_count}/{len(qa_pairs)} questions")
    
    # Save dataset
    dataset = {
        "name": "netflix_10k_comprehensive",
        "description": f"Comprehensive Q&A dataset from Netflix 2024 10-K filing. {len(test_cases)} questions across financial, operational, and strategic topics.",
        "test_cases": test_cases,
        "metadata": {
            "source": "Netflix 10-K 2024",
            "num_chunks": len(clean_chunks),
            "categories": list(set(qa["category"] for qa in qa_pairs)),
            "hash_method": "md5_16char",
        }
    }
    
    output_path = Path("data/eval_datasets/netflix_10k_comprehensive.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2))
    
    print(f"\nSaved dataset to {output_path}")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Categories: {dataset['metadata']['categories']}")


if __name__ == "__main__":
    main()
