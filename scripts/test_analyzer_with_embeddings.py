"""Test embedding analyzer with real Ollama embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_analyzer import EmbeddingAnalyzer
from src.embedding_analyzer.config_loader import load_config
from src.embedding_analyzer.analyzers.semantic import SemanticAnalyzer
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.chunkers.recursive_chunker import RecursiveChunker
from src.loaders.text_loader import TextLoader


def load_and_chunk(file_path: str, chunk_size: int = 512) -> list:
    """Load a file and chunk it."""
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=50)
    docs = loader.load(Path(file_path))
    
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    return all_chunks


def main():
    print("=" * 60)
    print("EMBEDDING ANALYZER WITH REAL EMBEDDINGS")
    print("=" * 60)
    
    # Initialize Ollama embeddings
    print("\n1. Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768,
    )
    
    # Test embedding function works
    print("   Testing embedding function...")
    test_vec = embeddings.embed_text("Hello world test")
    print(f"   ✓ Embedding works: {len(test_vec)} dimensions")
    
    # Create analyzer with embedding function
    print("\n2. Creating analyzer with semantic analysis...")
    config = load_config("financial")
    
    # Update semantic analyzer config for nomic-embed-text
    config["analyzers"]["semantic"]["thresholds"]["embedding_dim"] = 768
    
    analyzer = EmbeddingAnalyzer.from_config(config)
    analyzer.set_embedding_function(embeddings.embed_text)
    print(f"   ✓ Analyzer ready: {analyzer}")
    
    # Load test data - use a smaller sample first
    print("\n3. Loading test data...")
    file_path = "data/test_docs/netflix_10k.txt"
    
    if not Path(file_path).exists():
        # Fallback to sample
        file_path = "data/sample/rag_intro.txt"
    
    chunks = load_and_chunk(file_path, chunk_size=512)
    print(f"   Loaded {len(chunks)} chunks from {file_path}")
    
    # Analyze with real embeddings
    print("\n4. Analyzing chunks with semantic analysis...")
    print("   (This calls Ollama for each chunk)")
    
    texts = [c.content for c in chunks]
    
    # Analyze in batches to show progress
    all_reports = []
    batch_size = 10
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"   Processing {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}...")
        reports = analyzer.analyze_many(batch)
        all_reports.extend(reports)
    
    # Get stats
    stats = analyzer.get_summary_stats(all_reports)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTotal chunks: {stats['total_analyzed']}")
    print(f"Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
    print(f"Failed: {stats['failed']}")
    print(f"Avg score: {stats['avg_score']:.3f}")
    print(f"Min score: {stats['min_score']:.3f}")
    
    print(f"\nIssues: {stats['total_critical']}C / {stats['total_warning']}W / {stats['total_info']}I")
    
    if stats['top_issues']:
        print(f"\nTop Issues:")
        for code, count in list(stats['top_issues'].items())[:7]:
            print(f"   {code}: {count}")
    
    # Show failed chunks with embedding issues
    failed = [r for r in all_reports if not r.overall_passed]
    embedding_issues = [
        r for r in all_reports 
        if any(i.code in ['EMBEDDING_FAILED', 'EMBEDDING_NAN', 'EMBEDDING_ZERO', 'SEMANTIC_OUTLIER'] 
               for i in r.all_issues)
    ]
    
    if embedding_issues:
        print(f"\n❌ Chunks with EMBEDDING issues: {len(embedding_issues)}")
        for report in embedding_issues[:5]:
            print(f"\n   Text: {report.text_preview[:60]}...")
            for issue in report.all_issues:
                if issue.category.value == "semantic":
                    print(f"   └─ [{issue.code}] {issue.message[:70]}")
    else:
        print(f"\n✅ No embedding failures detected!")
    
    # Show low-magnitude embeddings (potential quality issues)
    low_quality = [
        r for r in all_reports
        for ar in r.analyzer_results
        if ar.analyzer_name == "semantic" and ar.metrics.get("embedding_magnitude", 1) < 0.5
    ]
    
    if low_quality:
        print(f"\n⚠️  Low embedding magnitude (potential quality issues): {len(low_quality)}")
    
    # Compute corpus stats for outlier detection
    print("\n" + "=" * 60)
    print("COMPUTING CORPUS STATS FOR OUTLIER DETECTION")
    print("=" * 60)
    
    # Get semantic analyzer
    semantic_analyzer = None
    for a in analyzer.analyzers:
        if isinstance(a, SemanticAnalyzer):
            semantic_analyzer = a
            break
    
    if semantic_analyzer and len(texts) > 10:
        print("\nComputing corpus centroid from sample...")
        sample_texts = texts[:min(50, len(texts))]  # Use up to 50 for stats
        try:
            corpus_stats = semantic_analyzer.compute_corpus_stats(sample_texts)
            print(f"   ✓ Corpus stats computed:")
            print(f"     Samples: {corpus_stats['num_samples']}")
            print(f"     Mean distance: {corpus_stats['mean_distance']:.4f}")
            print(f"     Std distance: {corpus_stats['std_distance']:.4f}")
            
            # Re-analyze to find outliers
            print("\n   Re-analyzing for outliers...")
            outlier_reports = analyzer.analyze_many(texts)
            outliers = [
                r for r in outlier_reports
                if any(i.code == 'SEMANTIC_OUTLIER' for i in r.all_issues)
            ]
            
            if outliers:
                print(f"\n   🎯 Found {len(outliers)} semantic outliers:")
                for report in outliers[:3]:
                    print(f"      {report.text_preview[:50]}...")
                    for issue in report.all_issues:
                        if issue.code == 'SEMANTIC_OUTLIER':
                            print(f"      └─ z-score: {issue.metadata.get('z_score', 'N/A')}")
            else:
                print(f"\n   ✓ No semantic outliers found")
                
        except Exception as e:
            print(f"   ⚠️  Could not compute corpus stats: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
