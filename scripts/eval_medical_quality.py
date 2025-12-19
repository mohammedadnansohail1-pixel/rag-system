"""
Comprehensive Medical RAG Quality Evaluation

Tests:
1. Retrieval Quality
2. Answer Accuracy
3. Response Time
4. Citation Coverage
"""

import sys
import json
import time
from datetime import datetime
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline


def create_medical_eval_dataset():
    """Medical QA evaluation dataset with ground truth."""
    return [
        {
            "question": "What are the classic symptoms of myocardial infarction?",
            "ground_truth": "Classic symptoms include chest pain pressure squeezing radiating left arm jaw shortness breath diaphoresis sweating nausea fatigue",
            "category": "cardiology"
        },
        {
            "question": "What is the mechanism of action of ACE inhibitors?",
            "ground_truth": "ACE inhibitors block angiotensin-converting enzyme preventing conversion angiotensin I to II reducing vasoconstriction aldosterone secretion lowering blood pressure",
            "category": "pharmacology"
        },
        {
            "question": "Describe the pathophysiology of type 2 diabetes mellitus.",
            "ground_truth": "Type 2 diabetes involves insulin resistance peripheral tissues progressive beta cell dysfunction hyperglycemia obesity genetic factors contribute",
            "category": "endocrinology"
        },
        {
            "question": "What are the phases of wound healing?",
            "ground_truth": "Wound healing four phases hemostasis clotting inflammation immune response proliferation tissue formation remodeling maturation strengthening",
            "category": "pathology"
        },
        {
            "question": "What is the Frank-Starling mechanism?",
            "ground_truth": "Frank-Starling mechanism heart stroke volume increases venous return greater filling stretches cardiac muscle fibers forceful contraction",
            "category": "physiology"
        },
        {
            "question": "What are the first-line treatments for hypertension?",
            "ground_truth": "First-line treatments thiazide diuretics ACE inhibitors angiotensin receptor blockers ARBs calcium channel blockers lifestyle modifications",
            "category": "pharmacology"
        },
        {
            "question": "What causes iron deficiency anemia?",
            "ground_truth": "Iron deficiency anemia chronic blood loss GI bleeding menstruation inadequate dietary intake malabsorption celiac disease increased demand pregnancy",
            "category": "hematology"
        },
        {
            "question": "Explain the renin-angiotensin-aldosterone system.",
            "ground_truth": "RAAS regulates blood pressure kidneys release renin BP drops angiotensinogen angiotensin I ACE converts angiotensin II vasoconstriction aldosterone",
            "category": "physiology"
        },
    ]


def calculate_keyword_coverage(answer, ground_truth):
    """Calculate what % of ground truth keywords appear in answer."""
    gt_words = set(ground_truth.lower().replace("-", " ").split())
    answer_words = set(answer.lower().replace("-", " ").split())
    
    # Filter out common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'that', 'this', 'it'}
    
    gt_words = gt_words - stopwords
    answer_words = answer_words - stopwords
    
    if not gt_words:
        return 0.0
    
    matches = gt_words & answer_words
    return len(matches) / len(gt_words)


def calculate_retrieval_relevance(sources, ground_truth):
    """Calculate if retrieved sources contain relevant information."""
    gt_words = set(ground_truth.lower().split())
    
    total_matches = 0
    for source in sources:
        content = source.get("content", "").lower()
        source_words = set(content.split())
        matches = len(gt_words & source_words)
        total_matches += matches
    
    return min(total_matches / len(gt_words), 1.0) if gt_words else 0.0


def run_evaluation():
    print("=" * 70)
    print("MEDICAL RAG SYSTEM - COMPREHENSIVE QUALITY EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize pipeline
    print("\n[1/3] Initializing RAG Pipeline...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434", 
        model="nomic-embed-text", 
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text")
    
    vectorstore = QdrantVectorStore(
        host="localhost", 
        port=6333, 
        collection_name="medical_subset", 
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant (medical_subset)")
    
    retriever = DenseRetriever(
        embeddings=embeddings, 
        vectorstore=vectorstore, 
        top_k=5
    )
    print("  ✓ Retriever: Dense (top_k=5)")
    
    llm = OllamaLLM(
        host="http://localhost:11434", 
        model="llama3.1:8b", 
        temperature=0.1
    )
    print("  ✓ LLM: llama3.1:8b")
    
    pipeline = RAGPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        chunker_config={"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 50}
    )
    print("  ✓ Pipeline ready")
    
    # Load evaluation dataset
    print("\n[2/3] Loading evaluation dataset...")
    eval_dataset = create_medical_eval_dataset()
    print(f"  ✓ {len(eval_dataset)} medical questions loaded")
    categories = set(q['category'] for q in eval_dataset)
    print(f"  Categories: {', '.join(categories)}")
    
    # Run evaluation
    print("\n[3/3] Running evaluation...")
    print("=" * 70)
    
    results = []
    total_time = 0
    
    for i, item in enumerate(eval_dataset, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(eval_dataset)}] Category: {item['category'].upper()}")
        print(f"Q: {item['question']}")
        
        # Time the query
        start_time = time.time()
        response = pipeline.query(item["question"], top_k=5)
        query_time = time.time() - start_time
        total_time += query_time
        
        # Calculate metrics
        answer_coverage = calculate_keyword_coverage(response.answer, item["ground_truth"])
        retrieval_relevance = calculate_retrieval_relevance(response.sources, item["ground_truth"])
        
        result = {
            "question": item["question"],
            "category": item["category"],
            "answer": response.answer,
            "num_sources": len(response.sources),
            "query_time": round(query_time, 2),
            "answer_coverage": round(answer_coverage, 3),
            "retrieval_relevance": round(retrieval_relevance, 3)
        }
        results.append(result)
        
        # Print result
        print(f"\nA: {response.answer[:300]}...")
        print(f"\n📊 Metrics:")
        print(f"   • Answer Coverage:     {answer_coverage:.1%}")
        print(f"   • Retrieval Relevance: {retrieval_relevance:.1%}")
        print(f"   • Sources Used:        {len(response.sources)}")
        print(f"   • Response Time:       {query_time:.2f}s")
    
    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    
    avg_coverage = sum(r["answer_coverage"] for r in results) / len(results)
    avg_relevance = sum(r["retrieval_relevance"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"   • Average Answer Coverage:     {avg_coverage:.1%}")
    print(f"   • Average Retrieval Relevance: {avg_relevance:.1%}")
    print(f"   • Average Response Time:       {avg_time:.2f}s")
    print(f"   • Total Questions:             {len(results)}")
    print(f"   • Total Evaluation Time:       {total_time:.1f}s")
    
    print(f"\n📊 BY CATEGORY:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_coverage = sum(r["answer_coverage"] for r in cat_results) / len(cat_results)
        cat_relevance = sum(r["retrieval_relevance"] for r in cat_results) / len(cat_results)
        print(f"   • {cat.capitalize():15} Coverage: {cat_coverage:.1%} | Relevance: {cat_relevance:.1%}")
    
    # Quality assessment
    print(f"\n🏆 QUALITY ASSESSMENT:")
    if avg_coverage >= 0.6 and avg_relevance >= 0.5:
        print("   ✅ PRODUCTION READY - High accuracy and relevance")
    elif avg_coverage >= 0.4 and avg_relevance >= 0.3:
        print("   ⚠️  ACCEPTABLE - Suitable for non-critical applications")
    else:
        print("   ❌ NEEDS IMPROVEMENT - Consider tuning retrieval/model")
    
    # Save results
    import os
    os.makedirs("data/eval", exist_ok=True)
    output_file = f"data/eval/medical_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "collection": "medical_subset",
                "llm": "llama3.1:8b",
                "embeddings": "nomic-embed-text",
                "top_k": 5
            },
            "summary": {
                "avg_answer_coverage": round(avg_coverage, 3),
                "avg_retrieval_relevance": round(avg_relevance, 3),
                "avg_response_time": round(avg_time, 2),
                "total_questions": len(results)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
