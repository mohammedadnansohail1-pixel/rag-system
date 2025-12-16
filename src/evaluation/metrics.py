"""RAG evaluation metrics (RAGAS-style)."""

import logging
from dataclasses import dataclass
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single RAG response.
    
    Attributes:
        faithfulness: How well answer is grounded in context (0-1)
        relevance: How relevant retrieved context is to query (0-1)
        answer_relevance: How well answer addresses the query (0-1)
        context_precision: Proportion of relevant chunks (0-1)
        overall_score: Weighted average of all metrics
    """
    faithfulness: float
    relevance: float
    answer_relevance: float
    context_precision: float
    overall_score: float
    
    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_precision": round(self.context_precision, 3),
            "overall_score": round(self.overall_score, 3),
        }


@dataclass 
class BatchEvaluationResult:
    """Results from evaluating multiple queries."""
    results: List[EvaluationResult]
    avg_faithfulness: float
    avg_relevance: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_overall_score: float
    
    def to_dict(self) -> dict:
        return {
            "num_queries": len(self.results),
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_relevance": round(self.avg_relevance, 3),
            "avg_answer_relevance": round(self.avg_answer_relevance, 3),
            "avg_context_precision": round(self.avg_context_precision, 3),
            "avg_overall_score": round(self.avg_overall_score, 3),
        }


class RAGEvaluator:
    """
    Evaluates RAG system responses using LLM-as-judge.
    
    Metrics:
    - Faithfulness: Is answer grounded in retrieved context?
    - Relevance: Is retrieved context relevant to query?
    - Answer Relevance: Does answer address the query?
    - Context Precision: What proportion of context is useful?
    
    Usage:
        evaluator = RAGEvaluator(llm)
        result = evaluator.evaluate(
            query="What is ML?",
            answer="Machine learning is...",
            contexts=["ML is a subset of AI...", "..."]
        )
    """
    
    def __init__(self, llm, weights: Optional[dict] = None):
        """
        Args:
            llm: LLM for evaluation (must have generate method)
            weights: Custom weights for overall score
        """
        self.llm = llm
        self.weights = weights or {
            "faithfulness": 0.3,
            "relevance": 0.25,
            "answer_relevance": 0.25,
            "context_precision": 0.2,
        }
        
        logger.info("Initialized RAGEvaluator")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            query: Original question
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationResult with all metrics
        """
        # Calculate each metric
        faithfulness = self._evaluate_faithfulness(answer, contexts)
        relevance = self._evaluate_relevance(query, contexts)
        answer_relevance = self._evaluate_answer_relevance(query, answer)
        context_precision = self._evaluate_context_precision(query, contexts)
        
        # Calculate weighted overall score
        overall_score = (
            self.weights["faithfulness"] * faithfulness +
            self.weights["relevance"] * relevance +
            self.weights["answer_relevance"] * answer_relevance +
            self.weights["context_precision"] * context_precision
        )
        
        return EvaluationResult(
            faithfulness=faithfulness,
            relevance=relevance,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            overall_score=overall_score,
        )
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple RAG responses.
        
        Args:
            queries: List of questions
            answers: List of generated answers
            contexts_list: List of context lists
            
        Returns:
            BatchEvaluationResult with averages
        """
        results = []
        
        for query, answer, contexts in zip(queries, answers, contexts_list):
            result = self.evaluate(query, answer, contexts)
            results.append(result)
        
        # Calculate averages
        n = len(results)
        avg_faithfulness = sum(r.faithfulness for r in results) / n
        avg_relevance = sum(r.relevance for r in results) / n
        avg_answer_relevance = sum(r.answer_relevance for r in results) / n
        avg_context_precision = sum(r.context_precision for r in results) / n
        avg_overall = sum(r.overall_score for r in results) / n
        
        return BatchEvaluationResult(
            results=results,
            avg_faithfulness=avg_faithfulness,
            avg_relevance=avg_relevance,
            avg_answer_relevance=avg_answer_relevance,
            avg_context_precision=avg_context_precision,
            avg_overall_score=avg_overall,
        )
    
    def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Check if answer is grounded in context."""
        if not contexts:
            return 0.0
        
        context_combined = "\n\n".join(contexts)
        
        prompt = f"""You are evaluating whether an answer is faithful to the given context.
        
CONTEXT:
{context_combined}

ANSWER:
{answer}

Rate how well the answer is supported by the context on a scale of 0-10:
- 0: Answer contains claims not in context (hallucination)
- 5: Answer is partially supported
- 10: Answer is fully grounded in context

Respond with ONLY a number between 0 and 10."""

        try:
            response = self.llm.generate(prompt, system_prompt="You are a precise evaluator. Respond only with a number.")
            score = self._extract_score(response)
            return score / 10.0
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_relevance(self, query: str, contexts: List[str]) -> float:
        """Check if retrieved context is relevant to query."""
        if not contexts:
            return 0.0
        
        context_combined = "\n\n".join(contexts)
        
        prompt = f"""You are evaluating whether retrieved context is relevant to a query.

QUERY:
{query}

RETRIEVED CONTEXT:
{context_combined}

Rate how relevant the context is to answering the query on a scale of 0-10:
- 0: Context is completely irrelevant
- 5: Context is somewhat relevant
- 10: Context is highly relevant and useful

Respond with ONLY a number between 0 and 10."""

        try:
            response = self.llm.generate(prompt, system_prompt="You are a precise evaluator. Respond only with a number.")
            score = self._extract_score(response)
            return score / 10.0
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Check if answer addresses the query."""
        prompt = f"""You are evaluating whether an answer addresses the question asked.

QUESTION:
{query}

ANSWER:
{answer}

Rate how well the answer addresses the question on a scale of 0-10:
- 0: Answer does not address the question at all
- 5: Answer partially addresses the question
- 10: Answer directly and completely addresses the question

Respond with ONLY a number between 0 and 10."""

        try:
            response = self.llm.generate(prompt, system_prompt="You are a precise evaluator. Respond only with a number.")
            score = self._extract_score(response)
            return score / 10.0
        except Exception as e:
            logger.warning(f"Answer relevance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        """Evaluate what proportion of contexts are useful."""
        if not contexts:
            return 0.0
        
        relevant_count = 0
        
        for context in contexts:
            prompt = f"""Is this context relevant to the query?

QUERY: {query}

CONTEXT: {context[:500]}

Respond with only YES or NO."""

            try:
                response = self.llm.generate(prompt, system_prompt="Respond only YES or NO.")
                if "YES" in response.upper():
                    relevant_count += 1
            except:
                relevant_count += 0.5
        
        return relevant_count / len(contexts)
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response."""
        # Find numbers in response
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0), 10)  # Clamp to 0-10
        
        return 5.0  # Default to middle score
