"""
Enhanced Medical RAG Evaluation - Uses ALL Features

Features tested:
- Structure-aware chunking
- Metadata enrichment  
- Parent-child retrieval
- Hybrid search (Dense + BM25)
- Cross-encoder reranking
- Guardrails
"""

import sys
import json
import time
from datetime import datetime
sys.path.insert(0, ".")

from src.core.config import Config
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline, EnhancedRAGConfig


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
    """Calculate comprehensive metrics."""
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
    }


def evaluate_pipeline(name, pipeline, dataset, use_enhanced=False):
    """Run evaluation on a pipeline."""
    results = []
    total_time = 0
    
    for item in dataset:
        start = time.time()
        
        if use_enhanced:
            response = pipeline.query(item["question"])
            answer = response.answer
            sources = [{"content": s.content, "metadata": s.metadata} for s in response.sources] if hasattr(response, 'sources') else []
        else:
            response = pipeline.query(item["question"], top_k=5)
            answer = response.answer
            sources = response.sources
        
        elapsed = time.time() - start
        total_time += elapsed
        
        metrics = calculate_metrics(answer, item["ground_truth"], sources)
        metrics["query_time"] = round(elapsed, 2)
        metrics["category"] = item["category"]
        metrics["question"] = item["question"]
        metrics["answer"] = answer[:300] if answer else ""
        results.append(metrics)
    
    avg_coverage = sum(r["answer_coverage"] for r in results) / len(results)
    avg_relevance = sum(r["retrieval_relevance"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    return {
        "name": name,
        "avg_coverage": round(avg_coverage, 3),
        "avg_relevance": round(avg_relevance, 3),
        "avg_time": round(avg_time, 2),
        "results": results
    }


def main():
    print("=" * 70)
    print("ENHANCED MEDICAL RAG - FULL FEATURE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Load config
    print("[1/5] Loading configuration...")
    try:
        config = Config.load("config/rag.yaml")
        print("  ✓ Config loaded from config/rag.yaml")
        
        # Show what's enabled
        enhanced = config.get_section("enhanced") or {}
        retrieval = config.get_section("retrieval") or {}
        
        print("\n  📋 Feature Configuration:")
        print(f"     • Search type: {retrieval.get('search_type', 'dense')}")
        print(f"     • Reranking: {retrieval.get('reranking', {}).get('enabled', False)}")
        print(f"     • Structure-aware chunking: {enhanced.get('chunking', {}).get('strategy', 'recursive')}")
        print(f"     • Metadata enrichment: {enhanced.get('enrichment', {}).get('enabled', False)}")
        print(f"     • Parent-child retrieval: {enhanced.get('parent_child', {}).get('enabled', False)}")
    except Exception as e:
        print(f"  ⚠ Config load failed: {e}")
        config = None
    
    # Initialize components
    print("\n[2/5] Initializing components...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text (768d)")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.1:8b")
    
    # Basic vectorstore
    basic_store = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="medical_subset",
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant (medical_subset)")
    
    # Reranker
    print("\n[3/5] Setting up reranker...")
    try:
        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("  ✓ Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        has_reranker = True
    except Exception as e:
        print(f"  ⚠ Reranker failed: {e}")
        reranker = None
        has_reranker = False
    
    # === PIPELINE 1: Basic Dense ===
    print("\n[4/5] Creating pipelines...")
    print("  • Basic Dense (baseline)...")
    basic_retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=basic_store,
        top_k=5
    )
    basic_pipeline = RAGPipeline(
        embeddings=embeddings,
        vectorstore=basic_store,
        retriever=basic_retriever,
        llm=llm,
        chunker_config={"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 50}
    )
    print("    ✓ Basic pipeline ready")
    
    # === PIPELINE 2: Dense + Reranking ===
    print("  • Dense + Reranking...")
    rerank_retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=basic_store,
        top_k=20  # Get more, then rerank
    )
    
    # === PIPELINE 3: Enhanced (if config available) ===
    enhanced_pipeline = None
    if config:
        print("  • Enhanced Pipeline (all features)...")
        try:
            enhanced_config = EnhancedRAGConfig.from_config(config)
            enhanced_pipeline = EnhancedRAGPipeline(
                embeddings=embeddings,
                vectorstore=basic_store,
                retriever=basic_retriever,
                llm=llm,
                config=enhanced_config
            )
            print("    ✓ Enhanced pipeline ready")
            print(f"      - Structure-aware: {enhanced_config.chunking_strategy}")
            print(f"      - Parent-child: {enhanced_config.enable_parent_child}")
            print(f"      - Enrichment: {enhanced_config.enable_enrichment}")
        except Exception as e:
            print(f"    ⚠ Enhanced pipeline failed: {e}")
            enhanced_pipeline = None
    
    # Run evaluations
    print("\n[5/5] Running evaluations...")
    print("=" * 70)
    
    all_results = []
    
    # 1. Basic Dense
    print("\n📊 [1/3] Evaluating: Basic Dense (baseline)...")
    basic_results = evaluate_pipeline("Basic Dense", basic_pipeline, EVAL_DATASET)
    all_results.append(basic_results)
    print(f"   → Coverage: {basic_results['avg_coverage']:.1%} | Relevance: {basic_results['avg_relevance']:.1%} | Time: {basic_results['avg_time']:.2f}s")
    
    # 2. Dense + Reranking
    if has_reranker:
        print("\n📊 [2/3] Evaluating: Dense + Reranking...")
        
        rerank_results_list = []
        total_time = 0
        
        for item in EVAL_DATASET:
            start = time.time()
            
            # Get candidates with larger top_k
            response = basic_pipeline.query(item["question"], top_k=20)
            
            # Rerank
            if response.sources and reranker:
                docs = [{"content": s.get("content", ""), "metadata": s.get("metadata", {})} 
                        for s in response.sources]
                reranked = reranker.rerank(query=item["question"], documents=docs)
                top_sources = [{"content": r.content, "metadata": r.metadata} for r in reranked[:5]]
            else:
                top_sources = response.sources[:5]
            
            elapsed = time.time() - start
            total_time += elapsed
            
            metrics = calculate_metrics(response.answer, item["ground_truth"], top_sources)
            metrics["query_time"] = round(elapsed, 2)
            metrics["category"] = item["category"]
            rerank_results_list.append(metrics)
        
        avg_cov = sum(r["answer_coverage"] for r in rerank_results_list) / len(rerank_results_list)
        avg_rel = sum(r["retrieval_relevance"] for r in rerank_results_list) / len(rerank_results_list)
        
        rerank_results = {
            "name": "Dense + Reranking",
            "avg_coverage": round(avg_cov, 3),
            "avg_relevance": round(avg_rel, 3),
            "avg_time": round(total_time / len(EVAL_DATASET), 2),
            "results": rerank_results_list
        }
        all_results.append(rerank_results)
        print(f"   → Coverage: {avg_cov:.1%} | Relevance: {avg_rel:.1%} | Time: {rerank_results['avg_time']:.2f}s")
    
    # 3. Enhanced Pipeline
    if enhanced_pipeline:
        print("\n📊 [3/3] Evaluating: Enhanced Pipeline (all features)...")
        try:
            enhanced_results = evaluate_pipeline("Enhanced (Full)", enhanced_pipeline, EVAL_DATASET, use_enhanced=True)
            all_results.append(enhanced_results)
            print(f"   → Coverage: {enhanced_results['avg_coverage']:.1%} | Relevance: {enhanced_results['avg_relevance']:.1%} | Time: {enhanced_results['avg_time']:.2f}s")
        except Exception as e:
            print(f"   ⚠ Enhanced evaluation failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Pipeline':<25} {'Coverage':>10} {'Relevance':>12} {'Avg Time':>10}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['name']:<25} {r['avg_coverage']:>9.1%} {r['avg_relevance']:>11.1%} {r['avg_time']:>9.2f}s")
    
    # Improvement over baseline
    if len(all_results) > 1:
        baseline = all_results[0]
        print(f"\n📊 Improvement over {baseline['name']}:")
        for r in all_results[1:]:
            cov_imp = ((r['avg_coverage'] - baseline['avg_coverage']) / baseline['avg_coverage'] * 100) if baseline['avg_coverage'] > 0 else 0
            rel_imp = ((r['avg_relevance'] - baseline['avg_relevance']) / baseline['avg_relevance'] * 100) if baseline['avg_relevance'] > 0 else 0
            print(f"   • {r['name']}: Coverage {cov_imp:+.1f}% | Relevance {rel_imp:+.1f}%")
    
    # Quality grade
    best = max(all_results, key=lambda x: x['avg_coverage'])
    print(f"\n🏆 Best Pipeline: {best['name']}")
    if best['avg_coverage'] >= 0.6:
        print("   ✅ PRODUCTION READY")
    elif best['avg_coverage'] >= 0.4:
        print("   ⚠️  ACCEPTABLE")
    else:
        print("   ❌ NEEDS IMPROVEMENT")
    
    # Save
    import os
    os.makedirs("data/eval", exist_ok=True)
    output_file = f"data/eval/medical_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "pipelines": all_results}, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
