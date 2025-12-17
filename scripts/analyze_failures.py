"""
Analyze retrieval failures to understand improvement areas.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load results
results_path = Path("data/eval_datasets/netflix_10k_results.json")
results = json.loads(results_path.read_text())

# Load dataset for context
dataset_path = Path("data/eval_datasets/netflix_10k_comprehensive.json")
dataset = json.loads(dataset_path.read_text())

print("=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)

# Get Dense results (best performer)
dense_metrics = results["Dense Only"]["metrics"]

print(f"\nOverall: MRR={dense_metrics['mrr']:.3f}, Hit Rate={dense_metrics['hit_rate']:.1%}")
print(f"40% of queries found NO relevant results in top 5")

# Analyze by category
print("\n" + "-" * 70)
print("PERFORMANCE BY CATEGORY")
print("-" * 70)

categories = {}
for tc in dataset["test_cases"]:
    cat = tc["metadata"]["category"]
    if cat not in categories:
        categories[cat] = {"queries": [], "keywords": []}
    categories[cat]["queries"].append(tc["query"])
    categories[cat]["keywords"].append(tc["metadata"]["keywords"])

for cat, data in sorted(categories.items()):
    print(f"\n{cat.upper()} ({len(data['queries'])} queries)")
    for q, kw in zip(data["queries"], data["keywords"]):
        print(f"  Q: {q[:50]}...")
        print(f"     Keywords: {kw}")

print("\n" + "-" * 70)
print("IMPROVEMENT HYPOTHESES")
print("-" * 70)

print("""
1. EMBEDDING MODEL LIMITATIONS
   - nomic-embed-text (137M params) may not handle financial jargon well
   - Numbers like "39,000" may not embed meaningfully
   - Consider: financial-domain embeddings or larger models

2. CHUNKING ISSUES
   - 1000-token chunks may split tables/financial data
   - Consider: semantic chunking, table-aware chunking

3. QUERY-DOCUMENT MISMATCH
   - Queries are natural language ("What was revenue?")
   - Documents are formal ("Streaming revenues 39,000,966")
   - Consider: query expansion, HyDE (Hypothetical Document Embeddings)

4. BM25 UNDERPERFORMING
   - Financial docs have unusual term distributions
   - Numbers don't match well with BM25
   - Consider: tuning k1/b, or using dense-only for this domain

5. MISSING RERANKING IN EVAL
   - Cross-encoder reranking not used in evaluation
   - Could significantly improve MRR
""")
