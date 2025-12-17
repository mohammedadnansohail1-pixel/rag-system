"""Validate domain configs on unseen data."""
import sys
sys.path.insert(0, ".")

from src.rag_pipeline import RAGPipeline
from src.loaders import SECLoader, WebLoader, CrawlConfig
from src.config import get_domain_config, FINANCIAL, TECHNICAL, LEGAL, GENERAL

print("=" * 70)
print("VALIDATION ON UNSEEN DATA")
print("=" * 70)

# Test queries for each domain
VALIDATION_TESTS = {
    FINANCIAL: [
        ("What is Apple's total revenue?", ["revenue", "net sales", "billion"]),
        ("How many employees does Apple have?", ["employee", "headcount", "workforce"]),
        ("What are Apple's main products?", ["iphone", "mac", "ipad", "services"]),
    ],
    TECHNICAL: [
        ("How do I create a FastAPI application?", ["fastapi", "app", "import"]),
        ("What are path parameters?", ["path", "parameter", "item_id"]),
        ("How do query parameters work?", ["query", "parameter", "optional"]),
    ],
    LEGAL: [
        ("Can GitHub terminate my account?", ["terminat", "account", "suspend"]),
        ("Who owns content I post?", ["content", "license", "intellectual property", "own"]),
        ("What are my responsibilities?", ["responsib", "must", "agree", "comply"]),
    ],
}

def run_validation():
    results = {}
    
    # 1. FINANCIAL - Apple 10-K
    print("\n[1/3] FINANCIAL: Apple 10-K")
    print("-" * 40)
    loader = SECLoader(download_dir="data/validation")
    apple_docs = loader.load("data/validation/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt")
    print(f"  Loaded {len(apple_docs)} documents")
    
    for config_name, domain in [("Domain-specific", FINANCIAL), ("General", GENERAL)]:
        pipeline = RAGPipeline(domain=domain, collection_name=f"val_fin_{config_name}")
        pipeline.index_documents(apple_docs, show_progress=False)
        
        hits = 0
        for query, keywords in VALIDATION_TESTS[FINANCIAL]:
            result = pipeline.query(query, top_k=5, return_sources=True)
            sources = result.get("sources", [])
            found = any(
                any(kw.lower() in s.get("content", "").lower() for kw in keywords)
                for s in sources
            ) if sources else False
            if found:
                hits += 1
        
        hit_rate = hits / len(VALIDATION_TESTS[FINANCIAL]) * 100
        print(f"  {config_name}: {hit_rate:.0f}% hit rate ({hits}/{len(VALIDATION_TESTS[FINANCIAL])})")
        results[f"financial_{config_name}"] = hit_rate
        pipeline.cleanup()
    
    # 2. TECHNICAL - FastAPI docs
    print("\n[2/3] TECHNICAL: FastAPI docs")
    print("-" * 40)
    from src.loaders import TextLoader
    from src.loaders.base import Document
    from bs4 import BeautifulSoup
    
    tech_docs = []
    for html_file in ["first-steps.html", "path-params.html", "query-params.html"]:
        filepath = f"data/validation/{html_file}"
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        tech_docs.append(Document(content=text, metadata={"source": filepath}))
    print(f"  Loaded {len(tech_docs)} documents")
    
    for config_name, domain in [("Domain-specific", TECHNICAL), ("General", GENERAL)]:
        pipeline = RAGPipeline(domain=domain, collection_name=f"val_tech_{config_name}")
        pipeline.index_documents(tech_docs, show_progress=False)
        
        hits = 0
        for query, keywords in VALIDATION_TESTS[TECHNICAL]:
            result = pipeline.query(query, top_k=5, return_sources=True)
            sources = result.get("sources", [])
            found = any(
                any(kw.lower() in s.get("content", "").lower() for kw in keywords)
                for s in sources
            ) if sources else False
            if found:
                hits += 1
        
        hit_rate = hits / len(VALIDATION_TESTS[TECHNICAL]) * 100
        print(f"  {config_name}: {hit_rate:.0f}% hit rate ({hits}/{len(VALIDATION_TESTS[TECHNICAL])})")
        results[f"technical_{config_name}"] = hit_rate
        pipeline.cleanup()
    
    # 3. LEGAL - GitHub ToS
    print("\n[3/3] LEGAL: GitHub Terms of Service")
    print("-" * 40)
    filepath = "data/validation/github_tos.html"
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    legal_docs = [Document(content=text, metadata={"source": filepath})]
    print(f"  Loaded {len(legal_docs)} documents")
    
    for config_name, domain in [("Domain-specific", LEGAL), ("General", GENERAL)]:
        pipeline = RAGPipeline(domain=domain, collection_name=f"val_legal_{config_name}")
        pipeline.index_documents(legal_docs, show_progress=False)
        
        hits = 0
        for query, keywords in VALIDATION_TESTS[LEGAL]:
            result = pipeline.query(query, top_k=5, return_sources=True)
            sources = result.get("sources", [])
            found = any(
                any(kw.lower() in s.get("content", "").lower() for kw in keywords)
                for s in sources
            ) if sources else False
            if found:
                hits += 1
        
        hit_rate = hits / len(VALIDATION_TESTS[LEGAL]) * 100
        print(f"  {config_name}: {hit_rate:.0f}% hit rate ({hits}/{len(VALIDATION_TESTS[LEGAL])})")
        results[f"legal_{config_name}"] = hit_rate
        pipeline.cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Domain':<12} {'Domain-specific':<18} {'General':<12} {'Delta':<10}")
    print("-" * 52)
    
    total_domain = 0
    total_general = 0
    for domain in ["financial", "technical", "legal"]:
        ds = results[f"{domain}_Domain-specific"]
        gen = results[f"{domain}_General"]
        delta = ds - gen
        total_domain += ds
        total_general += gen
        sign = "+" if delta >= 0 else ""
        print(f"{domain:<12} {ds:>6.0f}%            {gen:>6.0f}%       {sign}{delta:.0f}%")
    
    avg_domain = total_domain / 3
    avg_general = total_general / 3
    avg_delta = avg_domain - avg_general
    sign = "+" if avg_delta >= 0 else ""
    print("-" * 52)
    print(f"{'AVERAGE':<12} {avg_domain:>6.1f}%            {avg_general:>6.1f}%       {sign}{avg_delta:.1f}%")
    
    if avg_delta > 0:
        print(f"\n✓ Domain-specific configs validated on unseen data!")
    else:
        print(f"\n✗ Domain-specific configs did not improve on unseen data")

if __name__ == "__main__":
    run_validation()
