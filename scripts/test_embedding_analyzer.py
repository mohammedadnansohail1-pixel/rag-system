"""End-to-end test for embedding analyzer across different document types."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_analyzer import EmbeddingAnalyzer
from src.embedding_analyzer.config_loader import load_config
from src.chunkers.recursive_chunker import RecursiveChunker
from src.loaders.text_loader import TextLoader


def load_and_chunk(file_path: str, chunk_size: int = 512) -> list:
    """Load a file and chunk it."""
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=50)
    
    # Loader returns List[Document]
    docs = loader.load(Path(file_path))
    
    # Chunk all documents
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    
    return all_chunks


def analyze_file(
    file_path: str,
    analyzer: EmbeddingAnalyzer,
    label: str,
    chunk_size: int = 512,
) -> dict:
    """Analyze all chunks from a file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label}")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    try:
        chunks = load_and_chunk(file_path, chunk_size)
        print(f"Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Failed to load/chunk: {e}")
        return {"error": str(e)}
    
    # Analyze each chunk
    texts = [c.content for c in chunks]
    reports = analyzer.analyze_many(texts)
    
    # Get stats
    stats = analyzer.get_summary_stats(reports)
    
    # Print summary
    print(f"\n📊 Results:")
    print(f"   Total chunks: {stats['total_analyzed']}")
    print(f"   Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
    print(f"   Failed: {stats['failed']}")
    print(f"   Avg score: {stats['avg_score']:.3f}")
    print(f"   Min score: {stats['min_score']:.3f}")
    print(f"   Max score: {stats['max_score']:.3f}")
    print(f"\n   Issues: {stats['total_critical']}C / {stats['total_warning']}W / {stats['total_info']}I")
    
    if stats['top_issues']:
        print(f"\n   Top Issues:")
        for code, count in list(stats['top_issues'].items())[:5]:
            print(f"      {code}: {count}")
    
    # Show worst chunks
    failed_reports = [r for r in reports if not r.overall_passed]
    if failed_reports:
        print(f"\n   ❌ Sample failed chunks:")
        for report in failed_reports[:3]:
            print(f"      Score: {report.overall_score:.2f} | {report.text_preview[:50]}...")
            for issue in report.all_issues[:2]:
                print(f"         └─ [{issue.code}] {issue.message[:60]}")
    
    return stats


def main():
    print("=" * 60)
    print("EMBEDDING ANALYZER - END TO END TEST")
    print("=" * 60)
    
    # Test files
    test_files = [
        ("data/sample/rag_intro.txt", "RAG Introduction (simple text)", "default"),
        ("data/sample/chunking.txt", "Chunking Guide (simple text)", "default"),
        ("data/test_docs/netflix_10k.txt", "Netflix 10-K (financial)", "financial"),
        ("data/validation/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt", "Apple 10-K (financial)", "financial"),
    ]
    
    all_stats = {}
    
    for file_path, label, config_name in test_files:
        if not Path(file_path).exists():
            print(f"\n⚠️  Skipping {label} - file not found: {file_path}")
            continue
        
        # Load appropriate config
        config = load_config(config_name)
        analyzer = EmbeddingAnalyzer.from_config(config)
        
        stats = analyze_file(file_path, analyzer, label)
        all_stats[label] = stats
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    total_chunks = 0
    total_passed = 0
    total_failed = 0
    
    for label, stats in all_stats.items():
        if "error" not in stats:
            total_chunks += stats["total_analyzed"]
            total_passed += stats["passed"]
            total_failed += stats["failed"]
            print(f"{label}:")
            print(f"   {stats['pass_rate']:.1%} pass rate ({stats['passed']}/{stats['total_analyzed']})")
    
    if total_chunks > 0:
        overall_pass_rate = total_passed / total_chunks
        print(f"\n🎯 Overall: {overall_pass_rate:.1%} pass rate ({total_passed}/{total_chunks} chunks)")
        print(f"   {total_failed} chunks need attention")


if __name__ == "__main__":
    main()
