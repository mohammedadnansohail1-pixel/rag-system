"""Medical RAG - Evaluation using the correct pipeline"""

import sys
import json
import time
from datetime import datetime
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig


EVAL_DATASET = [
    {"question": "What are the classic symptoms of myocardial infarction?", 
     "ground_truth": "chest pain pressure squeezing radiating left arm jaw shortness breath diaphoresis sweating nausea fatigue", "category": "cardiology"},
    {"question": "What is the mechanism of action of ACE inhibitors?",
     "ground_truth": "ACE inhibitors block angiotensin-converting enzyme preventing conversion angiotensin I II reducing vasoconstriction aldosterone secretion lowering blood pressure", "category": "pharmacology"},
    {"question": "Describe the pathophysiology of type 2 diabetes mellitus.",
     "ground_truth": "Type 2 diabetes insulin resistance peripheral tissues progressive beta cell dysfunction hyperglycemia obesity genetic factors", "category": "endocrinology"},
    {"question": "What are the phases of wound healing?",
     "ground_truth": "wound healing phases hemostasis clotting inflammation immune response proliferation tissue formation remodeling maturation", "category": "pathology"},
    {"question": "What is the Frank-Starling mechanism?",
     "ground_truth": "Frank-Starling mechanism heart stroke volume increases venous return filling stretches cardiac muscle fibers forceful contraction", "category": "physiology"},
    {"question": "What are the first-line treatments for hypertension?",
     "ground_truth": "first-line treatments thiazide diuretics ACE inhibitors angiotensin receptor blockers ARBs calcium channel blockers lifestyle", "category": "pharmacology"},
    {"question": "What causes iron deficiency anemia?",
     "ground_truth": "iron deficiency anemia chronic blood loss GI bleeding menstruation inadequate dietary intake malabsorption celiac increased demand pregnancy", "category": "hematology"},
    {"question": "Explain the renin-angiotensin-aldosterone system.",
     "ground_truth": "RAAS renin angiotensin aldosterone blood pressure kidneys angiotensinogen ACE vasoconstriction", "category": "physiology"},
]


def calculate_metrics(answer, ground_truth, sources):
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'that', 'this', 'it'}
    
    gt_words = set(ground_truth.lower().replace("-", " ").split()) - stopwords
    answer_words = set(answer.lower().replace("-", " ").split()) - stopwords
    
    matches = gt_words & answer_words
    coverage = len(matches) / len(gt_words) if gt_words else 0
    
    source_text = ""
    for s in sources:
        if hasattr(s, 'content'):
            source_text += s.content + " "
        elif isinstance(s, dict):
            source_text += s.get("content", "") + " "
    
    source_words = set(source_text.lower().split()) - stopwords
    retrieval_matches = gt_words & source_words
    relevance = len(retrieval_matches) / len(gt_words) if gt_words else 0
    
    return {
        "coverage": round(coverage, 3),
        "relevance": round(min(relevance, 1.0), 3),
        "missing": list(gt_words - answer_words)[:5]
    }


def main():
    print("=" * 70)
    print("MEDICAL RAG - CORRECT PIPELINE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Initialize
    print("[1/2] Loading pipeline...")
    
    base_embeddings = OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768)
    embeddings = CachedEmbeddings(base_embeddings, enabled=True)
    
    vectorstore = QdrantHybridStore(host="localhost", port=6333, collection_name="medical_correct", dense_dimensions=768)
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore, sparse_encoder="bm25", top_k=20, reranker=reranker)
    llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
    
    config = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, enable_enrichment=True, enable_parent_child=True, top_k=20)
    
    pipeline = MultiDocumentPipeline(
        embeddings=embeddings, vectorstore=vectorstore, retriever=retriever, llm=llm, config=config,
        registry_path=".cache/medical_registry.json"
    )
    print(f"  ✓ Pipeline loaded ({len(pipeline.registry.all_documents)} docs in registry)")
    
    # Run evaluation
    print("\n[2/2] Running evaluation...")
    print("=" * 70)
    
    results = []
    total_time = 0
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    
    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(EVAL_DATASET)}] {item['category'].upper()}")
        print(f"Q: {item['question']}")
        
        start = time.time()
        response = pipeline.query(item["question"])
        elapsed = time.time() - start
        total_time += elapsed
        
        metrics = calculate_metrics(response.answer, item["ground_truth"], response.sources)
        confidence = response.confidence if hasattr(response, 'confidence') else "N/A"
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        results.append({
            "question": item["question"],
            "category": item["category"],
            "coverage": metrics["coverage"],
            "relevance": metrics["relevance"],
            "confidence": confidence,
            "sources": len(response.sources),
            "time": round(elapsed, 2)
        })
        
        print(f"\nA: {response.answer[:350]}..." if len(response.answer) > 350 else f"\nA: {response.answer}")
        print(f"\n📊 Coverage: {metrics['coverage']:.1%} | Relevance: {metrics['relevance']:.1%} | Confidence: {confidence.upper()} | Time: {elapsed:.2f}s")
        if metrics['missing']:
            print(f"   Missing: {', '.join(metrics['missing'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    
    avg_coverage = sum(r["coverage"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    print(f"\n🎯 OVERALL:")
    print(f"   • Answer Coverage:     {avg_coverage:.1%}")
    print(f"   • Retrieval Relevance: {avg_relevance:.1%}")
    print(f"   • Average Time:        {avg_time:.2f}s")
    print(f"   • Confidence: HIGH={confidence_counts.get('high',0)} MEDIUM={confidence_counts.get('medium',0)} LOW={confidence_counts.get('low',0)}")
    
    print(f"\n📊 BY CATEGORY:")
    categories = set(r['category'] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        cat_cov = sum(r['coverage'] for r in cat_results) / len(cat_results)
        status = "✅" if cat_cov >= 0.5 else "⚠️" if cat_cov >= 0.3 else "❌"
        print(f"   {status} {cat.capitalize():15} {cat_cov:.1%}")
    
    print(f"\n🏆 QUALITY: ", end="")
    if avg_coverage >= 0.6:
        print("✅ PRODUCTION READY")
    elif avg_coverage >= 0.4:
        print("⚠️  ACCEPTABLE")
    else:
        print("❌ NEEDS IMPROVEMENT")
    
    # Save
    import os
    os.makedirs("data/eval", exist_ok=True)
    with open(f"data/eval/medical_correct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump({"summary": {"coverage": avg_coverage, "relevance": avg_relevance}, "results": results}, f, indent=2)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
