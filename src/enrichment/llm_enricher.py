"""LLM-based enrichment for summaries and questions."""

import logging
from typing import List, Dict, Optional
from src.enrichment.base import BaseEnricher, EnrichmentResult

logger = logging.getLogger(__name__)


class LLMEnricher(BaseEnricher):
    """
    LLM-based metadata enrichment.
    
    Uses Ollama to generate:
    - Concise summaries
    - Potential questions the content can answer
    
    Note: Slower than rule-based extraction. Use selectively.
    
    Usage:
        from src.generation.ollama_llm import OllamaLLM
        
        llm = OllamaLLM(model="llama3.2")
        enricher = LLMEnricher(llm=llm)
        result = enricher.enrich("Meta's revenue grew 15% to $134 billion...")
        print(result.summary)
        print(result.potential_questions)
    """
    
    SUMMARY_PROMPT = """Summarize the following text in 1-2 sentences. Be concise and factual.

Text:
{content}

Summary:"""

    QUESTIONS_PROMPT = """What questions can be answered by the following text? List 2-3 specific questions.

Text:
{content}

Questions (one per line):"""

    def __init__(
        self,
        llm,
        generate_summary: bool = True,
        generate_questions: bool = True,
        max_content_length: int = 2000,
        timeout: float = 30.0,
    ):
        """
        Args:
            llm: LLM instance (OllamaLLM or compatible)
            generate_summary: Generate summaries
            generate_questions: Generate potential questions
            max_content_length: Truncate content beyond this length
            timeout: Timeout for LLM calls
        """
        self.llm = llm
        self.generate_summary = generate_summary
        self.generate_questions = generate_questions
        self.max_content_length = max_content_length
        self.timeout = timeout
        
        logger.info(f"Initialized LLMEnricher: summary={generate_summary}, questions={generate_questions}")
    
    @property
    def name(self) -> str:
        return "llm_enricher"
    
    def enrich(self, content: str, metadata: Optional[Dict] = None) -> EnrichmentResult:
        """Enrich content using LLM."""
        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        summary = None
        questions = []
        
        if self.generate_summary:
            summary = self._generate_summary(content)
        
        if self.generate_questions:
            questions = self._generate_questions(content)
        
        return EnrichmentResult(
            summary=summary,
            potential_questions=questions,
        )
    
    def _generate_summary(self, content: str) -> Optional[str]:
        """Generate a concise summary."""
        try:
            prompt = self.SUMMARY_PROMPT.format(content=content)
            response = self.llm.generate(
                prompt, 
                system_prompt="You are a concise summarizer. Output only the summary, nothing else."
            )
            
            # Clean up response
            summary = response.strip()
            
            # Remove common prefixes
            prefixes = ["Summary:", "Here is", "The text", "This text"]
            for prefix in prefixes:
                if summary.lower().startswith(prefix.lower()):
                    summary = summary[len(prefix):].strip()
            
            return summary if summary else None
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return None
    
    def _generate_questions(self, content: str) -> List[str]:
        """Generate potential questions."""
        try:
            prompt = self.QUESTIONS_PROMPT.format(content=content)
            response = self.llm.generate(
                prompt,
                system_prompt="You are a question generator. Output only questions, one per line."
            )
            
            # Parse questions
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering
                if line and line[0].isdigit():
                    line = line.lstrip('0123456789.-) ').strip()
                # Remove bullet points
                line = line.lstrip('•*- ').strip()
                
                if line and '?' in line:
                    questions.append(line)
            
            return questions[:3]  # Max 3 questions
            
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
            return []
    
    def enrich_batch(
        self, 
        contents: List[str], 
        metadatas: Optional[List[Dict]] = None
    ) -> List[EnrichmentResult]:
        """
        Enrich multiple contents.
        
        Note: This is sequential. For large batches, consider async.
        """
        results = []
        total = len(contents)
        
        for i, content in enumerate(contents):
            meta = metadatas[i] if metadatas else None
            
            if (i + 1) % 10 == 0:
                logger.info(f"LLM enrichment progress: {i + 1}/{total}")
            
            result = self.enrich(content, meta)
            results.append(result)
        
        return results
