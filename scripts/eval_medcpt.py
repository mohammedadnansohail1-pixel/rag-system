"""Evaluate MedCPT vs nomic-embed-text side by side."""

import sys
import time
sys.path.insert(0, ".")

from src.embeddings.medcpt_embeddings import MedCPTEmbeddings
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig

EVAL = [
    {"q": "What are the classic symptoms of myocardial infarction?", "gt": "chest pain pressure squeezing radiating left arm jaw shortness breath diaphoresis sweating nausea fatigue", "cat": "cardiology"},
    {"q": "What is the mechanism of action of ACE inhibitors?", "gt": "ACE inhibitors block angiotensin-converting enzyme preventing conversion angiotensin I II reducing vasoconstriction aldosterone secretion lowering blood pressure", "cat": "pharmacology"},
    {"q": "Describe the pathophysiology of type 2 diabetes mellitus.", "gt": "Type 2 diabetes insulin resistance peripheral tissues progressive beta cell dysfunction hyperglycemia obesity genetic factors", "cat": "endocrinology"},
    {"q": "What are the phases of wound healing?", "gt": "wound healing phases hemostasis clotting inflammation immune response proliferation tissue formation remodeling maturation", "cat": "pathology"},
    {"q": "What is the Frank-Starling mechanism?", "gt": "Frank-Starling mechanism heart stroke volume increases venous return filling stretches cardiac muscle fibers forceful contraction", "cat": "physiology"},
    {"q": "What are the first-line treatments for hypertension?", "gt": "first-line treatments thiazide diuretics ACE inhibitors angiotensin receptor blockers ARBs calcium channel blockers lifestyle", "cat": "pharmacology"},
    {"q": "What causes iron deficiency anemia?", "gt": "iron deficiency anemia chronic blood loss GI bleeding menstruation inadequate dietary intake malabsorption celiac increased demand pregnancy", "cat": "hematology"},
    {"q": "Explain the renin-angiotensin-aldosterone system.", "gt": "RAAS renin angiotensin aldosterone blood pressure kidneys angiotensinogen ACE vasoconstriction", "cat": "physiology"},
]

def calc(answer, gt, sources):
    stop = {'the','a','an','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','must','shall','can','to','of','in','for','on','with','at','by','from','as','into','through','and','or','but','if','then','than','so','that','this','it'}
    gt_w = set(gt.lower().replace("-"," ").split()) - stop
    ans_w = set(answer.lower().replace("-"," ").split()) - stop
    return len(gt_w & ans_w) / len(gt_w) if gt_w else 0

def run_eval(name, embeddings, collection):
    """Run evaluation on a collection."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"Collection: {collection}")
    print(f"{'='*60}")
    
    vs = QdrantHybridStore(host="localhost", port=6333, collection_name=collection, dense_dimensions=768)
    rr = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ret = HybridRetriever(embeddings=embeddings, vectorstore=vs, sparse_encoder="fastembed", top_k=20, reranker=rr)
    llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
    cfg = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, top_k=20)
    
    registry = f".cache/{collection}_registry.json"
    pipe = MultiDocumentPipeline(embeddings=embeddings, vectorstore=vs, retriever=ret, llm=llm, config=cfg, registry_path=registry)
    
    results = []
    total_time = 0
    conf_cnt = {"high": 0, "medium": 0, "low": 0}
    
    for i, e in enumerate(EVAL, 1):
        t0 = time.time()
        r = pipe.query(e["q"])
        t = time.time() - t0
        total_time += t
        
        cov = calc(r.answer, e["gt"], r.sources)
        conf = r.confidence if hasattr(r, 'confidence') else "N/A"
        conf_cnt[conf] = conf_cnt.get(conf, 0) + 1
        results.append({"cat": e["cat"], "cov": cov, "conf": conf})
        
        status = "✓" if cov >= 0.5 else "○" if cov >= 0.3 else "✗"
        print(f"  {status} [{e['cat'][:4]}] {cov:.0%} | {conf} | {t:.1f}s")
    
    avg_cov = sum(r["cov"] for r in results) / len(results)
    print(f"\n  📊 Average Coverage: {avg_cov:.1%}")
    print(f"  📊 Confidence: HIGH={conf_cnt.get('high',0)} MED={conf_cnt.get('medium',0)} LOW={conf_cnt.get('low',0)}")
    print(f"  📊 Avg Time: {total_time/len(EVAL):.2f}s")
    
    return avg_cov, conf_cnt

def main():
    print("="*60)
    print("MEDCPT vs NOMIC-EMBED-TEXT COMPARISON")
    print("="*60)
    
    # Test 1: nomic-embed-text (current)
    print("\nLoading nomic-embed-text...")
    nomic = CachedEmbeddings(
        OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768),
        enabled=True
    )
    nomic_cov, nomic_conf = run_eval("nomic-embed-text", nomic, "medical_final")
    
    # Test 2: MedCPT
    print("\nLoading MedCPT...")
    medcpt = CachedEmbeddings(
        MedCPTEmbeddings(),
        enabled=True,
        cache_dir=".cache/medcpt_embeddings"
    )
    medcpt_cov, medcpt_conf = run_eval("MedCPT (medical-specific)", medcpt, "medical_medcpt")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n  {'Metric':<25} {'nomic-embed-text':<18} {'MedCPT':<18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")
    print(f"  {'Answer Coverage':<25} {nomic_cov:.1%}{'':<14} {medcpt_cov:.1%}")
    print(f"  {'HIGH Confidence':<25} {nomic_conf.get('high',0):<18} {medcpt_conf.get('high',0)}")
    print(f"  {'LOW Confidence':<25} {nomic_conf.get('low',0):<18} {medcpt_conf.get('low',0)}")
    
    diff = medcpt_cov - nomic_cov
    if diff > 0.02:
        print(f"\n  🏆 MedCPT wins by {diff:.1%}")
    elif diff < -0.02:
        print(f"\n  🏆 nomic-embed-text wins by {-diff:.1%}")
    else:
        print(f"\n  🤝 Roughly equal (diff: {diff:+.1%})")
    
    print("="*60)

if __name__ == "__main__":
    main()
