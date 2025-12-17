"""
Cross-domain evaluation: Does domain-specific config actually help?

Tests:
1. Financial data with FINANCIAL vs GENERAL config
2. Technical data with TECHNICAL vs GENERAL config
3. Legal data with LEGAL vs GENERAL config

Measures real retrieval metrics to prove business value.
"""
import sys
import hashlib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_domain_config, FINANCIAL, TECHNICAL, LEGAL, GENERAL
from src.loaders import SECLoader, WebLoader, CrawlConfig
from src.loaders.base import Document
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.base import RetrievalResult
from src.evaluation import EvaluationDataset, TestCase, RetrievalEvaluator


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


class ConfigurableRetriever:
    """Retriever that uses domain config settings."""
    
    def __init__(self, embeddings, vectorstore, config):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.config = config
    
    def retrieve(self, query, top_k=5):
        query_emb = self.embeddings.embed_text(query)
        results = self.vectorstore.search(query_emb, top_k=top_k)
        return [
            RetrievalResult(content=r.content, metadata=r.metadata, score=r.score)
            for r in results
        ]
    
    def health_check(self):
        return True


def evaluate_domain(domain_name, documents, test_cases, embeddings):
    """Evaluate domain-specific vs general config."""
    
    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'=' * 70}")
    
    # Get configs
    domain_config = get_domain_config(domain_name)
    general_config = get_domain_config(GENERAL)
    
    results = {}
    
    for config_name, config in [("Domain-specific", domain_config), ("General", general_config)]:
        print(f"\n  Testing: {config_name} config")
        print(f"    Chunk size: {config.chunking.chunk_size}")
        print(f"    Sparse weight: {config.retrieval.sparse_weight}")
        
        # Chunk with config
        chunker = config.create_chunker()
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        # Clean chunks
        texts = []
        for chunk in all_chunks:
            content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
            if len(content) > 50:
                texts.append(content)
        
        print(f"    Chunks created: {len(texts)}")
        
        # Create vector store
        collection = f"eval_{domain_name}_{config_name.lower().replace('-', '_')}"
        vectorstore = QdrantVectorStore(collection_name=collection, dimensions=768)
        
        # Index
        emb_list = embeddings.embed_batch(texts)
        vectorstore.add(texts=texts, embeddings=emb_list)
        
        # Build test dataset with chunk IDs
        chunk_ids = {stable_hash(t): t for t in texts}
        
        dataset_cases = []
        for tc in test_cases:
            # Find relevant chunks by keyword matching
            relevant = []
            for chunk_id, content in chunk_ids.items():
                content_lower = content.lower()
                if all(kw.lower() in content_lower for kw in tc["keywords"]):
                    relevant.append(chunk_id)
                    if len(relevant) >= 3:
                        break
            
            if relevant:
                dataset_cases.append(TestCase(
                    query=tc["query"],
                    relevant_ids=set(relevant),
                ))
        
        if not dataset_cases:
            print(f"    WARNING: No test cases with matching chunks")
            vectorstore._client.delete_collection(collection)
            continue
        
        dataset = EvaluationDataset(
            name=f"{domain_name}_{config_name}",
            test_cases=dataset_cases,
        )
        
        print(f"    Test cases: {len(dataset_cases)}")
        
        # Evaluate
        retriever = ConfigurableRetriever(embeddings, vectorstore, config)
        evaluator = RetrievalEvaluator(
            retriever,
            id_extractor=lambda r: stable_hash(r.content),
        )
        
        result = evaluator.evaluate(dataset, k=config.retrieval.top_k, verbose=False)
        results[config_name] = result
        
        print(f"    MRR: {result.mean_mrr:.3f}, Hit Rate: {result.hit_rate:.1%}, Recall: {result.mean_recall:.3f}")
        
        # Cleanup
        vectorstore._client.delete_collection(collection)
    
    return results


def main():
    print("=" * 70)
    print("CROSS-DOMAIN EVALUATION")
    print("Does domain-specific configuration actually help?")
    print("=" * 70)
    
    # Initialize embeddings once
    print("\nInitializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    all_results = {}
    
    # ==========================================================================
    # FINANCIAL DOMAIN
    # ==========================================================================
    print("\n[1/3] Loading financial data (Netflix 10-K)...")
    loader = SECLoader(download_dir='data/test_docs')
    financial_docs = loader.load(
        'data/test_docs/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt'
    )
    
    financial_tests = [
        {"query": "What was Netflix revenue in 2024?", "keywords": ["streaming revenues", "39,000"]},
        {"query": "What is the operating margin?", "keywords": ["operating margin", "percent"]},
        {"query": "How many employees does Netflix have?", "keywords": ["employees", "full-time"]},
        {"query": "What are the risk factors?", "keywords": ["risk factors"]},
        {"query": "Who are the executive officers?", "keywords": ["executive officers", "chief"]},
    ]
    
    all_results[FINANCIAL] = evaluate_domain(FINANCIAL, financial_docs, financial_tests, embeddings)
    
    # ==========================================================================
    # TECHNICAL DOMAIN
    # ==========================================================================
    print("\n[2/3] Loading technical data (Python docs)...")
    web_loader = WebLoader()
    config = CrawlConfig(max_pages=5, max_depth=1, delay_seconds=0.5)
    technical_docs = web_loader.crawl('https://docs.python.org/3/tutorial/index.html', config=config)
    
    technical_tests = [
        {"query": "How do I handle exceptions?", "keywords": ["try", "except"]},
        {"query": "What are list comprehensions?", "keywords": ["list", "comprehension"]},
        {"query": "How do I define a function?", "keywords": ["def", "function"]},
        {"query": "How do classes work?", "keywords": ["class", "object"]},
        {"query": "How do I work with files?", "keywords": ["open", "file"]},
    ]
    
    all_results[TECHNICAL] = evaluate_domain(TECHNICAL, technical_docs, technical_tests, embeddings)
    
    # ==========================================================================
    # LEGAL DOMAIN
    # ==========================================================================
    print("\n[3/3] Loading legal data (Terms of Service)...")
    legal_docs = [
        Document(content='''
TERMS OF SERVICE - ACME CORPORATION

ARTICLE 1: ACCEPTANCE OF TERMS
1.1 By accessing or using the ACME Service ("Service"), you agree to be bound by these Terms of Service ("Terms"). If you do not agree to all terms and conditions, you must not use the Service.
1.2 We reserve the right to modify these Terms at any time. Continued use of the Service after modifications constitutes acceptance of the new Terms.
1.3 These Terms constitute a legally binding agreement between you ("User") and ACME Corporation ("Company").

ARTICLE 2: USER ELIGIBILITY AND ACCOUNTS
2.1 Minimum Age: You must be at least 18 years of age to create an account and use the Service. Users between 13-17 may use the Service only with parental consent.
2.2 Account Security: You are responsible for maintaining the confidentiality of your account credentials. You must notify us immediately of any unauthorized access.
2.3 Account Termination: We reserve the right to suspend or terminate accounts that violate these Terms without prior notice.

ARTICLE 3: PAYMENT AND BILLING
3.1 Subscription Fees: Access to premium features requires payment of subscription fees as listed on our pricing page.
3.2 Refund Policy: All fees are non-refundable except as required by applicable law. Pro-rated refunds may be issued at Company's sole discretion.
3.3 Price Changes: We may modify pricing with 30 days advance notice. Existing subscriptions will be honored until renewal.
3.4 Payment Failure: Failure to pay may result in suspension or termination of service.

ARTICLE 4: INTELLECTUAL PROPERTY RIGHTS
4.1 Company Property: The Service, including all content, features, and functionality, is owned by ACME Corporation and protected by copyright, trademark, and other intellectual property laws.
4.2 User License: We grant you a limited, non-exclusive, non-transferable license to use the Service for personal, non-commercial purposes.
4.3 Restrictions: You may not copy, modify, distribute, sell, or lease any part of our Service without explicit written permission.
4.4 User Content: You retain ownership of content you submit, but grant us a worldwide license to use, display, and distribute such content.

ARTICLE 5: LIMITATION OF LIABILITY
5.1 Disclaimer: THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED.
5.2 Liability Cap: IN NO EVENT SHALL COMPANY'S TOTAL LIABILITY EXCEED THE AMOUNT PAID BY YOU IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.
5.3 Exclusions: COMPANY SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES.
5.4 Essential Purpose: These limitations apply even if any remedy fails of its essential purpose.

ARTICLE 6: INDEMNIFICATION
6.1 You agree to indemnify, defend, and hold harmless ACME Corporation from any claims, damages, or expenses arising from your use of the Service or violation of these Terms.

ARTICLE 7: DISPUTE RESOLUTION
7.1 Arbitration: Any disputes shall be resolved through binding arbitration in accordance with AAA rules.
7.2 Class Action Waiver: You waive the right to participate in class action lawsuits against the Company.
7.3 Jurisdiction: These Terms are governed by the laws of the State of Delaware.

ARTICLE 8: PRIVACY AND DATA
8.1 Data Collection: Our collection and use of personal data is governed by our Privacy Policy.
8.2 Data Security: We implement reasonable security measures but cannot guarantee absolute security.
8.3 Third Parties: We may share data with third-party service providers as described in our Privacy Policy.

ARTICLE 9: TERMINATION
9.1 User Termination: You may terminate your account at any time through account settings.
9.2 Company Termination: We may terminate or suspend access immediately for Terms violations.
9.3 Effect of Termination: Upon termination, your right to use the Service ceases immediately.

ARTICLE 10: MISCELLANEOUS
10.1 Entire Agreement: These Terms constitute the entire agreement between you and Company.
10.2 Severability: If any provision is found unenforceable, remaining provisions remain in effect.
10.3 Waiver: Failure to enforce any right does not constitute waiver of that right.

Last Updated: January 1, 2025
''', metadata={'source': 'terms_of_service.txt'}),
    ]
    
    legal_tests = [
        {"query": "What is the minimum age requirement?", "keywords": ["age", "18"]},
        {"query": "Can I get a refund?", "keywords": ["refund", "non-refundable"]},
        {"query": "What is the liability limit?", "keywords": ["liability", "twelve", "months"]},
        {"query": "How are disputes resolved?", "keywords": ["arbitration", "dispute"]},
        {"query": "What happens if I violate terms?", "keywords": ["terminate", "suspend"]},
    ]
    
    all_results[LEGAL] = evaluate_domain(LEGAL, legal_docs, legal_tests, embeddings)
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Domain-Specific vs General Configuration")
    print("=" * 70)
    
    print(f"\n{'Domain':<12} {'Config':<18} {'MRR':<8} {'Hit Rate':<10} {'Recall':<8}")
    print("-" * 60)
    
    improvements = []
    
    for domain in [FINANCIAL, TECHNICAL, LEGAL]:
        if domain not in all_results or not all_results[domain]:
            continue
            
        results = all_results[domain]
        
        for config_name in ["Domain-specific", "General"]:
            if config_name not in results:
                continue
            r = results[config_name]
            print(f"{domain:<12} {config_name:<18} {r.mean_mrr:<8.3f} {r.hit_rate:<10.1%} {r.mean_recall:<8.3f}")
        
        # Calculate improvement
        if "Domain-specific" in results and "General" in results:
            domain_mrr = results["Domain-specific"].mean_mrr
            general_mrr = results["General"].mean_mrr
            if general_mrr > 0:
                improvement = (domain_mrr - general_mrr) / general_mrr * 100
                improvements.append((domain, improvement))
        
        print()
    
    if improvements:
        print("-" * 60)
        print("IMPROVEMENT FROM DOMAIN-SPECIFIC CONFIG:")
        for domain, imp in improvements:
            symbol = "+" if imp >= 0 else ""
            print(f"  {domain}: {symbol}{imp:.1f}% MRR")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    avg_improvement = sum(imp for _, imp in improvements) / len(improvements) if improvements else 0
    
    if avg_improvement > 5:
        print(f"\n✓ Domain-specific configuration improves MRR by {avg_improvement:.1f}% on average")
        print("  RECOMMENDATION: Use domain-specific configs in production")
    elif avg_improvement > 0:
        print(f"\n~ Domain-specific configuration shows modest {avg_improvement:.1f}% improvement")
        print("  RECOMMENDATION: Consider domain configs for specialized use cases")
    else:
        print(f"\n✗ Domain-specific configuration shows no clear benefit ({avg_improvement:.1f}%)")
        print("  RECOMMENDATION: General config may be sufficient")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
