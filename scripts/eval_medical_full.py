"""
Medical RAG - FULL FEATURED Evaluation

Uses ALL features:
- HybridRetriever (Dense + BM25)
- CrossEncoderReranker
- RAGPipelineV2
"""

import sys
import json
import time
from datetime import datetime
sys.path.insert(0, ".")

from src.core.config import Config
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline_v2 import RAGPipelineV2


EVAL_DATASET = [
    {
        "question": "What are the classic symptoms of myocardial infarction?",
        "ground_truth": "chest pain pressure squeezing radiating left arm jaw shortness breath diaphoresis sweating nausea fatigue",
        "category": "cardiology"
    },
    {
        "question": "What is the mechanism of action of ACE inhibitors?",
        "ground_truth": "ACE inhibitors block angiotensin-converting enzyme preventing conversion angiotensin I II reducing vasoconstriction aldosterone secretion lowering blood pressure",
        "category": "pharmacology"
    },
    {
        "question": "Describe the pathophysiology of type 2 diabetes mellitus.",
        "ground_truth": "Type 2 diabetes insulin resistance peripheral tissues progressive beta cell dysfunction hyperglycemia obesity genetic factors",
        "category": "endocrinology"
    },
    {
        "question": "What are the phases of wound healing?",
        "ground_truth": "wound healing phases hemostasis clotting inflammation immune response proliferation tissue formation remodeling maturation",
        "category": "pathology"
    },
    {
        "question": "What is the Frank-Starling mechanism?",
        "ground_truth": "Frank-Starling mechanism heart stroke volume increases venous return filling stretches cardiac muscle fibers forceful contraction",
        "category": "physiology"
    },
    {
        "question": "What are the first-line treatments for hypertension?",
        "ground_truth": "first-line treatments thiazide diuretics ACE inhibitors angiotensin receptor blockers ARBs calcium channel blockers lifestyle",
        "category": "pharmacology"
    },
    {
        "question": "What causes iron deficiency anemia?",
        "ground_truth": "iron deficiency anemia chronic blood loss GI bleeding menstruation inadequate dietary intake malabsorption celiac increased demand pregnancy",
        "category": "hematology"
    },
    {
        "question": "Explain the renin-angiotensin-aldosterone system.",
        "ground_truth": "RAAS renin angiotensin aldosterone blood pressure kidneys angiotensinogen ACE vasoconstriction",
        "category": "physiology"
    },
]


def calculate_metrics(answer, ground_truth, sources):
    """Calculate evaluation metrics."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'that', 'this', 'it'}
    
    gt_words = set(ground_truth.lower().replace("-", " ").split()) - stopwords
    answer_words = set(answer.lower().replace("-", " ").split()) - stopwords
    
    matches = gt_words & answer_words
    coverage = len(matches) / len(gt_words) if gt_words else 0
    
    source_text = " ".join([s.get("content", "") for s in sources]).lower()
    source_words = set(source_text.split()) - stopwords
    retrieval_matches = gt_words & source_words
    relevance = len(retrieval_matches) / len(gt_words) if gt_words else 0
    
    return {
        "answer_coverage": round(coverage, 3),
        "retrieval_relevance": round(min(relevance, 1.0), 3),
        "keywords_found": list(matches)[:5],
        "keywords_missing": list(gt_words - answer_words)[:5]
    }


def main():
    print("=" * 70)
    print("MEDICAL RAG - FULL FEATURED EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Load config
    print("[1/3] Loading configuration...")
    config = Config.load("config/rag.yaml")
    retrieval_cfg = config.get_section("retrieval") or {}
    reranking_cfg = retrieval_cfg.get("reranking", {})
    
    print("\n  📋 Features Active:")
    print(f"     • Search:      Hybrid (Dense + BM25)")
    print(f"     • Reranking:   {reranking_cfg.get('cross_encoder', {}).get('model', 'cross-encoder')}")
    print(f"     • Top-K:       {retrieval_cfg.get('retrieval_top_k', 20)} → rerank to {reranking_cfg.get('top_n', 5)}")
    
    # Initialize full pipeline
    print("\n[2/3] Initializing full-featured pipeline...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.1:8b")
    
    vectorstore = QdrantHybridStore(
        host="localhost",
        port=6333,
        collection_name="medical_full",
        dense_dimensions=768
    )
    print("  ✓ VectorStore: medical_full (hybrid)")
    
    reranker = CrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("  ✓ Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
        top_k=20,
        reranker=reranker
    )
    print("  ✓ Retriever: Hybrid + Reranking")
    
    pipeline = RAGPipelineV2(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        reranker=reranker
    )
    print("  ✓ Pipeline: RAGPipelineV2 (full features)")
    
    # Run evaluation
    print("\n[3/3] Running evaluation...")
    print("=" * 70)
    
    results = []
    total_time = 0
    
    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(EVAL_DATASET)}] {item['category'].upper()}")
        print(f"Q: {item['question']}")
        
        start = time.time()
        response = pipeline.query(
            item["question"],
            retrieval_top_k=20,
            rerank_top_n=5
        )
        elapsed = time.time() - start
        total_time += elapsed
        
        answer = response.answer
        sources = response.sources
        
        metrics = calculate_metrics(answer, item["ground_truth"], sources)
        metrics["query_time"] = round(elapsed, 2)
        metrics["category"] = item["category"]
        metrics["question"] = item["question"]
        metrics["answer"] = answer
        metrics["num_sources"] = len(sources)
        metrics["reranked"] = response.rerank_scores is not None
        results.append(metrics)
        
        # Print result
        print(f"\nA: {answer[:400]}..." if len(answer) > 400 else f"\nA: {answer}")
        print(f"\n📊 Metrics:")
        print(f"   • Answer Coverage:     {metrics['answer_coverage']:.1%}")
        print(f"   • Retrieval Relevance: {metrics['retrieval_relevance']:.1%}")
        print(f"   • Sources:             {metrics['num_sources']}")
        print(f"   • Reranked:            {'✓' if metrics['reranked'] else '✗'}")
        print(f"   • Response Time:       {elapsed:.2f}s")
        if metrics['keywords_missing']:
            print(f"   • Missing:             {', '.join(metrics['keywords_missing'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    
    avg_coverage = sum(r["answer_coverage"] for r in results) / len(results)
    avg_relevance = sum(r["retrieval_relevance"] for r in results) / len(results)
    avg_time = total_time / len(results)
    reranked_count = sum(1 for r in results if r["reranked"])
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"   • Average Answer Coverage:     {avg_coverage:.1%}")
    print(f"   • Average Retrieval Relevance: {avg_relevance:.1%}")
    print(f"   • Average Response Time:       {avg_time:.2f}s")
    print(f"   • Questions Reranked:          {reranked_count}/{len(results)}")
    
    # By category
    categories = set(r['category'] for r in results)
    print(f"\n📊 BY CATEGORY:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        cat_cov = sum(r['answer_coverage'] for r in cat_results) / len(cat_results)
        cat_rel = sum(r['retrieval_relevance'] for r in cat_results) / len(cat_results)
        status = "✅" if cat_cov >= 0.5 else "⚠️" if cat_cov >= 0.3 else "❌"
        print(f"   {status} {cat.capitalize():15} Coverage: {cat_cov:.1%} | Relevance: {cat_rel:.1%}")
    
    # Quality assessment
    print(f"\n🏆 QUALITY ASSESSMENT:")
    if avg_coverage >= 0.6 and avg_relevance >= 0.5:
        print("   ✅ PRODUCTION READY - High accuracy and relevance")
    elif avg_coverage >= 0.4 and avg_relevance >= 0.3:
        print("   ⚠️  ACCEPTABLE - Suitable for assisted applications")
    else:
        print("   ❌ NEEDS IMPROVEMENT - Consider tuning or expanding corpus")
    
    # Save results
    import os
    os.makedirs("data/eval", exist_ok=True)
    output_file = f"data/eval/medical_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "collection": "medical_full",
                "llm": "llama3.1:8b",
                "embeddings": "nomic-embed-text",
                "retriever": "hybrid (dense + bm25)",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "retrieval_top_k": 20,
                "rerank_top_n": 5
            },
            "summary": {
                "avg_answer_coverage": round(avg_coverage, 3),
                "avg_retrieval_relevance": round(avg_relevance, 3),
                "avg_response_time": round(avg_time, 2),
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("✓ Full-featured evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
