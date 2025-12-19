"""Medical RAG - Final evaluation with fastembed"""
import sys, json, time
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
    cov = len(gt_w & ans_w) / len(gt_w) if gt_w else 0
    src_txt = " ".join([s.content if hasattr(s,'content') else s.get("content","") for s in sources]).lower()
    rel = len(gt_w & set(src_txt.split())) / len(gt_w) if gt_w else 0
    return round(cov,3), round(min(rel,1.0),3), list(gt_w - ans_w)[:5]

print("="*70)
print("MEDICAL RAG - FINAL EVALUATION (FASTEMBED)")
print("="*70)

# Load pipeline
emb = CachedEmbeddings(OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768))
vs = QdrantHybridStore(host="localhost", port=6333, collection_name="medical_final", dense_dimensions=768)
rr = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
ret = HybridRetriever(embeddings=emb, vectorstore=vs, sparse_encoder="fastembed", top_k=20, reranker=rr)
llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
cfg = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, enable_enrichment=True, top_k=20)
pipe = MultiDocumentPipeline(embeddings=emb, vectorstore=vs, retriever=ret, llm=llm, config=cfg, registry_path=".cache/medical_final_registry.json")
print(f"✓ Pipeline loaded ({len(pipe.registry.all_documents)} docs, 37K vectors)\n")

results = []
conf_cnt = {"high":0,"medium":0,"low":0}
t_total = 0

for i, e in enumerate(EVAL, 1):
    print(f"{'─'*70}\n[{i}/8] {e['cat'].upper()}\nQ: {e['q']}")
    t0 = time.time()
    r = pipe.query(e["q"])
    t = time.time() - t0
    t_total += t
    cov, rel, miss = calc(r.answer, e["gt"], r.sources)
    conf = r.confidence if hasattr(r,'confidence') else "N/A"
    conf_cnt[conf] = conf_cnt.get(conf,0) + 1
    results.append({"cat":e["cat"],"cov":cov,"rel":rel,"conf":conf,"t":round(t,2)})
    print(f"\nA: {r.answer[:300]}..." if len(r.answer)>300 else f"\nA: {r.answer}")
    print(f"\n📊 Cov: {cov:.1%} | Rel: {rel:.1%} | Conf: {conf.upper()} | {t:.2f}s")
    if miss: print(f"   Missing: {', '.join(miss)}")

avg_cov = sum(r["cov"] for r in results)/len(results)
avg_rel = sum(r["rel"] for r in results)/len(results)
print(f"\n{'='*70}\n📈 FINAL RESULTS\n{'='*70}")
print(f"\n🎯 OVERALL: Coverage={avg_cov:.1%} | Relevance={avg_rel:.1%} | Time={t_total/len(EVAL):.2f}s")
print(f"   Confidence: HIGH={conf_cnt.get('high',0)} MEDIUM={conf_cnt.get('medium',0)} LOW={conf_cnt.get('low',0)}")

print(f"\n📊 BY CATEGORY:")
for cat in sorted(set(r['cat'] for r in results)):
    cr = [r for r in results if r['cat']==cat]
    c = sum(r['cov'] for r in cr)/len(cr)
    print(f"   {'✅' if c>=0.5 else '⚠️' if c>=0.3 else '❌'} {cat.capitalize():15} {c:.1%}")

print(f"\n🏆 QUALITY: {'✅ PRODUCTION READY' if avg_cov>=0.6 else '⚠️ ACCEPTABLE' if avg_cov>=0.4 else '❌ NEEDS WORK'}")
print("="*70)
