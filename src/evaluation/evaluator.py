"""RAG system evaluator."""

import logging
from typing import List, Optional, Dict, Any

from src.evaluation.metrics import (
    EvaluationResult,
    BatchEvaluationResult,
    calculate_context_relevance,
    calculate_answer_relevance_simple,
    calculate_faithfulness_simple,
    calculate_overall_score,
)
from src.pipeline.rag_pipeline_production import ProductionRAGPipeline, ProductionRAGResponse

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator for RAG system responses.
    
    Measures:
    - Context relevance: Are retrieved chunks relevant to query?
    - Answer relevance: Does answer address the query?
    - Faithfulness: Is answer grounded in retrieved context?
    
    Usage:
        evaluator = RAGEvaluator(pipeline)
        
        # Evaluate single query
        result = evaluator.evaluate_query("What is ML?")
        
        # Evaluate batch
        results = evaluator.evaluate_batch([
            {"query": "What is ML?"},
            {"query": "What is gradient descent?"},
        ])
    """
    
    def __init__(
        self,
        pipeline: ProductionRAGPipeline,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pipeline: RAG pipeline to evaluate
            weights: Custom weights for overall score
        """
        self.pipeline = pipeline
        self.weights = weights or {
            "context_relevance": 0.3,
            "answer_relevance": 0.3,
            "faithfulness": 0.4,
        }
        
        logger.info("Initialized RAGEvaluator")
    
    def evaluate_response(
        self,
        query: str,
        response: ProductionRAGResponse,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.
        
        Args:
            query: Original query
            response: RAG pipeline response
            
        Returns:
            EvaluationResult with scores
        """
        # Extract data from response
        answer = response.answer
        contexts = [s["content"] for s in response.sources]
        scores = [s["score"] for s in response.sources]
        
        # Calculate metrics
        context_rel = calculate_context_relevance(query, contexts, scores)
        answer_rel = calculate_answer_relevance_simple(query, answer)
        faithfulness = calculate_faithfulness_simple(answer, contexts)
        
        overall = calculate_overall_score(
            context_rel, answer_rel, faithfulness, self.weights
        )
        
        return EvaluationResult(
            query=query,
            answer=answer,
            context_relevance=context_rel,
            answer_relevance=answer_rel,
            faithfulness=faithfulness,
            overall_score=overall,
            details={
                "num_sources": len(contexts),
                "avg_retrieval_score": response.avg_score,
                "confidence": response.confidence,
                "validation_passed": response.validation_passed,
            }
        )
    
    def evaluate_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> EvaluationResult:
        """
        Run query through pipeline and evaluate.
        
        Args:
            query: Question to evaluate
            top_k: Number of chunks to retrieve
            
        Returns:
            EvaluationResult with scores
        """
        # Get response from pipeline
        response = self.pipeline.query(query, top_k=top_k)
        
        # Evaluate
        return self.evaluate_response(query, response)
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple queries.
        
        Args:
            test_cases: List of {"query": "..."} dicts
            top_k: Number of chunks to retrieve per query
            
        Returns:
            BatchEvaluationResult with aggregate scores
        """
        results = []
        
        for i, case in enumerate(test_cases):
            query = case["query"]
            logger.info(f"Evaluating {i+1}/{len(test_cases)}: {query[:50]}...")
            
            try:
                result = self.evaluate_query(query, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate query: {e}")
                # Add failed result
                results.append(EvaluationResult(
                    query=query,
                    answer="ERROR",
                    context_relevance=0.0,
                    answer_relevance=0.0,
                    faithfulness=0.0,
                    overall_score=0.0,
                    details={"error": str(e)}
                ))
        
        # Calculate averages
        n = len(results)
        if n == 0:
            return BatchEvaluationResult(
                results=[],
                avg_context_relevance=0.0,
                avg_answer_relevance=0.0,
                avg_faithfulness=0.0,
                avg_overall=0.0,
                num_samples=0,
            )
        
        return BatchEvaluationResult(
            results=results,
            avg_context_relevance=sum(r.context_relevance for r in results) / n,
            avg_answer_relevance=sum(r.answer_relevance for r in results) / n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n,
            avg_overall=sum(r.overall_score for r in results) / n,
            num_samples=n,
        )
    
    def evaluate_with_ground_truth(
        self,
        test_cases: List[Dict[str, str]],
        top_k: int = 10,
    ) -> BatchEvaluationResult:
        """
        Evaluate with ground truth answers.
        
        Args:
            test_cases: List of {"query": "...", "expected": "..."} dicts
            top_k: Number of chunks to retrieve
            
        Returns:
            BatchEvaluationResult with scores including ground truth comparison
        """
        results = []
        
        for i, case in enumerate(test_cases):
            query = case["query"]
            expected = case.get("expected", "")
            
            logger.info(f"Evaluating {i+1}/{len(test_cases)}: {query[:50]}...")
            
            try:
                response = self.pipeline.query(query, top_k=top_k)
                result = self.evaluate_response(query, response)
                
                # Add ground truth comparison if available
                if expected:
                    result.details["expected"] = expected
                    result.details["answer_vs_expected"] = self._compare_answers(
                        response.answer, expected
                    )
                
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate: {e}")
                results.append(EvaluationResult(
                    query=query,
                    answer="ERROR",
                    context_relevance=0.0,
                    answer_relevance=0.0,
                    faithfulness=0.0,
                    overall_score=0.0,
                    details={"error": str(e)}
                ))
        
        n = len(results)
        return BatchEvaluationResult(
            results=results,
            avg_context_relevance=sum(r.context_relevance for r in results) / n if n else 0,
            avg_answer_relevance=sum(r.answer_relevance for r in results) / n if n else 0,
            avg_faithfulness=sum(r.faithfulness for r in results) / n if n else 0,
            avg_overall=sum(r.overall_score for r in results) / n if n else 0,
            num_samples=n,
        )
    
    def _compare_answers(self, generated: str, expected: str) -> float:
        """Simple comparison between generated and expected answers."""
        if not generated or not expected:
            return 0.0
        
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        gen_words -= stop_words
        exp_words -= stop_words
        
        if not exp_words:
            return 0.5
        
        overlap = len(gen_words & exp_words)
        return min(1.0, overlap / len(exp_words))
