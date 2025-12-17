"""Evaluation dataset management."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """
    A single test case for retrieval evaluation.
    
    Attributes:
        query: The search query
        relevant_ids: Set of document IDs that are relevant
        relevance_scores: Optional graded relevance (for NDCG)
        metadata: Optional additional info
    """
    query: str
    relevant_ids: Set[str]
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure relevant_ids is a set
        if isinstance(self.relevant_ids, list):
            self.relevant_ids = set(self.relevant_ids)
        
        # Set default relevance scores for binary relevance
        if not self.relevance_scores:
            self.relevance_scores = {doc_id: 1.0 for doc_id in self.relevant_ids}


@dataclass
class EvaluationDataset:
    """
    Collection of test cases for evaluation.
    
    Attributes:
        name: Dataset name
        test_cases: List of test cases
        description: Optional description
    """
    name: str
    test_cases: List[TestCase]
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __iter__(self):
        return iter(self.test_cases)
    
    def __getitem__(self, idx: int) -> TestCase:
        return self.test_cases[idx]
    
    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "test_cases": [
                {
                    "query": tc.query,
                    "relevant_ids": list(tc.relevant_ids),
                    "relevance_scores": tc.relevance_scores,
                    "metadata": tc.metadata,
                }
                for tc in self.test_cases
            ]
        }
        
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Saved dataset '{self.name}' to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text())
        
        test_cases = [
            TestCase(
                query=tc["query"],
                relevant_ids=set(tc["relevant_ids"]),
                relevance_scores=tc.get("relevance_scores", {}),
                metadata=tc.get("metadata", {}),
            )
            for tc in data["test_cases"]
        ]
        
        return cls(
            name=data["name"],
            test_cases=test_cases,
            description=data.get("description", ""),
        )
    
    def summary(self) -> str:
        """Return summary statistics."""
        if not self.test_cases:
            return f"Dataset '{self.name}': empty"
        
        avg_relevant = sum(len(tc.relevant_ids) for tc in self.test_cases) / len(self.test_cases)
        
        return (
            f"Dataset '{self.name}':\n"
            f"  Test cases: {len(self.test_cases)}\n"
            f"  Avg relevant docs per query: {avg_relevant:.1f}\n"
            f"  Description: {self.description or 'N/A'}"
        )


def create_dataset_from_qa_pairs(
    qa_pairs: List[Dict[str, Any]],
    name: str = "custom",
    chunk_matcher: Optional[callable] = None,
) -> EvaluationDataset:
    """
    Create evaluation dataset from Q&A pairs.
    
    This is useful when you have questions and their expected answers,
    and need to find which chunks contain those answers.
    
    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys
        name: Dataset name
        chunk_matcher: Optional function(answer, chunks) -> relevant_ids
        
    Returns:
        EvaluationDataset
        
    Example:
        qa_pairs = [
            {"question": "What is Netflix revenue?", "answer": "$39 billion"},
            {"question": "Who is CEO?", "answer": "Ted Sarandos"},
        ]
        dataset = create_dataset_from_qa_pairs(qa_pairs)
    """
    test_cases = []
    
    for pair in qa_pairs:
        query = pair.get("question") or pair.get("query")
        answer = pair.get("answer") or pair.get("expected")
        relevant = pair.get("relevant_ids", set())
        
        if isinstance(relevant, list):
            relevant = set(relevant)
        
        test_cases.append(TestCase(
            query=query,
            relevant_ids=relevant,
            metadata={"expected_answer": answer},
        ))
    
    return EvaluationDataset(
        name=name,
        test_cases=test_cases,
        description=f"Created from {len(qa_pairs)} Q&A pairs",
    )
