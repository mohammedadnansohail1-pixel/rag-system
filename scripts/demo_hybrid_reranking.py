"""
End-to-end demo of hybrid retrieval + reranking pipeline.

Demonstrates:
- Hybrid retrieval (dense + sparse with RRF fusion)
- Reranking (cross-encoder)
- Comparison with/without reranking
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking import RerankerFactory
from src.generation.factory import LLMFactory
from src.pipeline import RAGPipelineV2


# Test documents - enterprise knowledge base simulation
DOCUMENTS = [
    # Kafka (0-4)
    {
        "text": "Apache Kafka is a distributed event streaming platform. It handles trillions of events daily for companies like LinkedIn, Netflix, and Uber.",
        "source": "kafka_overview.md",
        "topic": "kafka"
    },
    {
        "text": "Kafka guarantees message ordering within a partition. Each message gets a sequential offset number. Consumers read messages in offset order.",
        "source": "kafka_ordering.md",
        "topic": "kafka"
    },
    {
        "text": "Kafka consumers can join consumer groups for parallel processing. Each partition is assigned to exactly one consumer in the group.",
        "source": "kafka_consumers.md",
        "topic": "kafka"
    },
    {
        "text": "Kafka Connect provides pre-built connectors for databases, S3, Elasticsearch, and more. It simplifies data integration pipelines.",
        "source": "kafka_connect.md",
        "topic": "kafka"
    },
    {
        "text": "Kafka Streams is a client library for building real-time applications. It supports stateful operations like joins and aggregations.",
        "source": "kafka_streams.md",
        "topic": "kafka"
    },
    
    # RAG Systems (5-9)
    {
        "text": "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation. It reduces hallucinations by grounding answers in retrieved documents.",
        "source": "rag_overview.md",
        "topic": "rag"
    },
    {
        "text": "Hybrid retrieval combines dense (semantic) and sparse (keyword) search. RRF fusion merges results from both methods for better recall.",
        "source": "rag_hybrid.md",
        "topic": "rag"
    },
    {
        "text": "Reranking uses cross-encoders to re-score retrieved documents. It improves precision by examining query-document pairs jointly.",
        "source": "rag_reranking.md",
        "topic": "rag"
    },
    {
        "text": "Chunking strategies include fixed-size, recursive, and semantic chunking. Chunk size affects retrieval quality and context window usage.",
        "source": "rag_chunking.md",
        "topic": "rag"
    },
    {
        "text": "RAGAS evaluates RAG systems on faithfulness, relevancy, precision, and recall. It provides metrics for continuous improvement.",
        "source": "rag_evaluation.md",
        "topic": "rag"
    },
    
    # Vector Databases (10-14)
    {
        "text": "Vector databases store embeddings for similarity search. Popular options include Qdrant, Pinecone, Milvus, and Chroma.",
        "source": "vectordb_overview.md",
        "topic": "vectordb"
    },
    {
        "text": "HNSW (Hierarchical Navigable Small World) is an ANN algorithm. It provides fast approximate nearest neighbor search with high recall.",
        "source": "vectordb_hnsw.md",
        "topic": "vectordb"
    },
    {
        "text": "Qdrant supports both dense and sparse vectors in the same collection. This enables hybrid search with a single database.",
        "source": "vectordb_qdrant.md",
        "topic": "vectordb"
    },
    {
        "text": "Cosine similarity measures angle between vectors. Euclidean distance measures straight-line distance. Dot product is fastest but requires normalized vectors.",
        "source": "vectordb_similarity.md",
        "topic": "vectordb"
    },
    {
        "text": "Vector database scaling strategies include sharding, replication, and quantization. Product quantization reduces memory while maintaining accuracy.",
        "source": "vectordb_scaling.md",
        "topic": "vectordb"
    },
    
    # LLMs (15-19)
    {
        "text": "Large Language Models like GPT-4, Claude, and Llama use transformer architecture. They predict the next token based on context.",
        "source": "llm_overview.md",
        "topic": "llm"
    },
    {
        "text": "Prompt engineering techniques include few-shot examples, chain-of-thought, and system prompts. Good prompts significantly improve output quality.",
        "source": "llm_prompting.md",
        "topic": "llm"
    },
    {
        "text": "Fine-tuning adapts pre-trained models to specific tasks. LoRA reduces memory requirements by training low-rank adapter matrices.",
        "source": "llm_finetuning.md",
        "topic": "llm"
    },
    {
        "text": "Ollama runs LLMs locally with simple CLI. It supports Llama, Mistral, and other open models. Great for development and privacy-sensitive use cases.",
        "source": "llm_ollama.md",
        "topic": "llm"
    },
    {
        "text": "Token limits constrain LLM context windows. Strategies include chunking, summarization, and hierarchical retrieval to fit within limits.",
        "source": "llm_context.md",
        "topic": "llm"
    },
]


def run_demo():
    print("=" * 70)
    print("HYBRID RETRIEVAL + RERANKING DEMO")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/5] Initializing embeddings (Ollama)...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    print("[2/5] Initializing hybrid vector store (Qdrant)...")
    vectorstore = QdrantHybridStore(
        collection_name="demo_hybrid_rerank",
        dense_dimensions=768,
        recreate_collection=True,
    )
    
    print("[3/5] Initializing hybrid retriever (dense + SPLADE)...")
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
    
    print("[4/5] Initializing reranker (cross-encoder)...")
    reranker = RerankerFactory.create("cross_encoder")
    
    print("[5/5] Initializing LLM (Ollama)...")
    llm = LLMFactory.create("ollama", model="llama3.2")
    
    # Create pipeline
    print("\nCreating RAGPipelineV2...")
    pipeline = RAGPipelineV2(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    
    # Index documents
    print(f"\nIndexing {len(DOCUMENTS)} documents...")
    texts = [d["text"] for d in DOCUMENTS]
    metadatas = [{"source": d["source"], "topic": d["topic"]} for d in DOCUMENTS]
    
    start = time.perf_counter()
    retriever.add_documents(texts, metadatas)
    index_time = time.perf_counter() - start
    print(f"Indexing completed in {index_time:.2f}s")
    
    # Test queries
    queries = [
        "How does Kafka guarantee message ordering?",
        "What is hybrid retrieval and how does RRF work?",
        "How do I run LLMs locally?",
        "What are the best practices for chunking documents?",
    ]
    
    print("\n" + "=" * 70)
    print("QUERY COMPARISON: With vs Without Reranking")
    print("=" * 70)
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"QUERY: {query}")
        print("─" * 70)
        
        # Compare with/without reranking
        comparison = pipeline.query_compare(query, top_k=3)
        
        print("\n WITHOUT RERANKING:")
        without = comparison["without_reranking"]
        for i, src in enumerate(without.sources, 1):
            topic = src["metadata"].get("topic", "?")
            source = src["metadata"].get("source", "?")
            print(f"  {i}. [{topic}] {source} (score: {src['score']:.3f})")
            print(f"     {src['content'][:80]}...")
        
        print("\n WITH RERANKING:")
        with_rerank = comparison["with_reranking"]
        for i, src in enumerate(with_rerank.sources, 1):
            topic = src["metadata"].get("topic", "?")
            source = src["metadata"].get("source", "?")
            print(f"  {i}. [{topic}] {source} (score: {src['score']:.3f})")
            print(f"     {src['content'][:80]}...")
        
        print(f"\n ANSWER (with reranking):")
        print(f"  {with_rerank.answer[:300]}...")
    
    # Full query demo
    print("\n" + "=" * 70)
    print("FULL PIPELINE DEMO")
    print("=" * 70)
    
    query = "Explain how to build a production RAG system with hybrid retrieval"
    print(f"\nQuery: {query}")
    
    start = time.perf_counter()
    response = pipeline.query(
        query,
        retrieval_top_k=10,
        rerank_top_n=3,
    )
    query_time = time.perf_counter() - start
    
    print(f"\nLatency: {query_time*1000:.0f}ms")
    print(f"Sources used: {len(response.sources)}")
    print(f"Reranked: {response.metadata.get('reranked', False)}")
    
    print(f"\nAnswer:\n{response.answer}")
    
    print(f"\nSources:")
    for i, src in enumerate(response.sources, 1):
        print(f"  {i}. {src['metadata'].get('source', '?')} (score: {src['score']:.3f})")
    
    # Health check
    print("\n" + "=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    health = pipeline.health_check()
    for component, status in health.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {component}: {'healthy' if status else 'unhealthy'}")
    
    # Cleanup
    print(f"\nCleaning up...")
    vectorstore._client.delete_collection("demo_hybrid_rerank")
    print("Done!")


if __name__ == "__main__":
    run_demo()
