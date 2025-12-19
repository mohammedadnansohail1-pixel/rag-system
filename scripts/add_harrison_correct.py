"""Add Harrison's to medical_correct collection"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig

file_path = Path("data/medical/textbooks/InternalMed_Harrison.txt")
print(f"Adding: {file_path.name} ({file_path.stat().st_size/1024/1024:.1f} MB)")

base_embeddings = OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768)
embeddings = CachedEmbeddings(base_embeddings, enabled=True)
vectorstore = QdrantHybridStore(host="localhost", port=6333, collection_name="medical_correct", dense_dimensions=768)
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore, sparse_encoder="bm25", top_k=20, reranker=reranker)
llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
config = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, enable_enrichment=True, top_k=20)

pipeline = MultiDocumentPipeline(
    embeddings=embeddings, vectorstore=vectorstore, retriever=retriever, llm=llm, config=config,
    registry_path=".cache/medical_registry.json"
)

print(f"Current: {len(pipeline.registry.all_documents)} docs")
stats = pipeline.ingest_file(str(file_path), skip_if_exists=False)
print(f"Added: {stats.get('chunks', 0)} chunks")
print(f"Total docs: {len(pipeline.registry.all_documents)}")
