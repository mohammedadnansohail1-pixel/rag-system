"""
NLI-based Faithfulness Evaluation.

Uses DeBERTa NLI model to check if claims are entailed by context.
Based on research showing NLI entailment correlates with faithfulness.
"""

import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    import torch
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    score: float
    num_claims: int
    supported_claims: int
    unsupported_claims: List[str]
    details: dict


class NLIFaithfulnessEvaluator:
    """Evaluate faithfulness using NLI."""
    
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = None,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.nli_pipeline = None
        self.device = device
        self.model_name = model_name
        
        if NLI_AVAILABLE:
            self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        try:
            if self.device is None:
                self.device = 0 if torch.cuda.is_available() else -1
            
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.nli_pipeline = None
    
    def decompose_into_claims(self, text: str) -> List[str]:
        """Decompose text into atomic claims."""
        claims = []
        
        # Split by newlines first
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for bullet points: -, *, •, 1., 1)
            bullet_pattern = r'^[\-\*\•]|\d+[\.\)]'
            if re.match(bullet_pattern, line):
                # Remove bullet marker
                claim = re.sub(r'^[\-\*\•\d\.\)\s]+', '', line).strip()
                if len(claim) > 15:
                    claims.append(claim)
            else:
                # Split by sentence endings
                sentences = re.split(r'(?<=[.!?])\s+', line)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 15:
                        continue
                    
                    # Remove meta-prefixes but keep content
                    sent = re.sub(
                        r'^(based on|according to)(\s+the)?(\s+documents?|\s+sources?|\s+context)?,?\s*',
                        '', sent, flags=re.IGNORECASE
                    )
                    sent = sent.strip()
                    
                    # Skip pure meta sentences
                    if re.match(r'^(in summary|overall|in conclusion)[\s:,.]*$', sent, re.IGNORECASE):
                        continue
                    
                    if len(sent) > 15:
                        claims.append(sent)
        
        # Deduplicate
        seen = set()
        unique = []
        for c in claims:
            key = c.lower()[:40]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique[:15]
    
    def check_entailment(self, claim: str, context: str) -> Tuple[float, str]:
        """Check if context entails the claim."""
        if self.nli_pipeline is None:
            return self._fallback_entailment(claim, context)
        
        try:
            # Truncate context to fit model
            ctx = context[:1500]
            result = self.nli_pipeline(
                f"{ctx}</s></s>{claim}",
                truncation=True,
                max_length=512,
            )
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if label == 'entailment':
                return score, 'entailment'
            elif label == 'contradiction':
                return 1 - score, 'contradiction'
            else:
                return 0.5, 'neutral'
            
        except Exception as e:
            logger.warning(f"NLI failed: {e}")
            return self._fallback_entailment(claim, context)
    
    def _fallback_entailment(self, claim: str, context: str) -> Tuple[float, str]:
        """Fallback using keyword overlap."""
        # Get significant words (4+ chars)
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
        context_lower = context.lower()
        
        stopwords = {'this', 'that', 'these', 'those', 'which', 'what',
                     'where', 'when', 'with', 'from', 'have', 'been',
                     'would', 'could', 'should', 'about', 'their', 'there'}
        claim_words -= stopwords
        
        if not claim_words:
            return 0.5, "neutral"
        
        matches = sum(1 for w in claim_words if w in context_lower)
        score = min(1.0, (matches / len(claim_words)) * 1.2)
        
        if score > 0.6:
            return score, "entailment"
        elif score < 0.25:
            return score, "contradiction"
        return score, "neutral"
    
    def evaluate(self, answer: str, contexts: List[str]) -> FaithfulnessResult:
        """Evaluate faithfulness of answer against contexts."""
        if not answer or not contexts:
            return FaithfulnessResult(0.0, 0, 0, [], {"error": "Empty input"})
        
        combined_context = " ".join(contexts)
        claims = self.decompose_into_claims(answer)
        
        if not claims:
            return FaithfulnessResult(0.5, 0, 0, [], {"note": "No claims extracted"})
        
        supported = 0
        unsupported = []
        claim_scores = []
        
        for claim in claims:
            score, label = self.check_entailment(claim, combined_context)
            claim_scores.append({
                "claim": claim[:80],
                "score": round(score, 3),
                "label": label,
            })
            
            if score >= self.threshold:
                supported += 1
            else:
                unsupported.append(claim[:80])
        
        final_score = supported / len(claims)
        
        return FaithfulnessResult(
            score=final_score,
            num_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            details={
                "claim_scores": claim_scores,
                "model": self.model_name if self.nli_pipeline else "fallback",
                "threshold": self.threshold,
            },
        )


def calculate_faithfulness_nli(
    answer: str,
    contexts: List[str],
    evaluator: Optional[NLIFaithfulnessEvaluator] = None,
) -> float:
    """Convenience function for NLI-based faithfulness."""
    if evaluator is None:
        evaluator = NLIFaithfulnessEvaluator()
    return evaluator.evaluate(answer, contexts).score


__all__ = [
    "NLIFaithfulnessEvaluator",
    "FaithfulnessResult", 
    "calculate_faithfulness_nli",
    "NLI_AVAILABLE",
]
