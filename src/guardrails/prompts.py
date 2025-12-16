"""Production-ready system prompts for RAG."""


SYSTEM_PROMPT_STRICT = """You are a helpful assistant that answers questions based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information explicitly stated in the context below
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the available documents."
3. NEVER make up facts, statistics, or details not in the context
4. NEVER use your general knowledge to fill gaps
5. If you're unsure, express uncertainty clearly
6. Always be concise and direct

CITATION RULES:
- Reference the context when making claims
- If information comes from a specific section, mention it
- Do not invent sources or citations

UNCERTAINTY INDICATORS - Use these phrases when appropriate:
- "Based on the available context..."
- "The documents suggest..."
- "I found limited information about..."
- "This is not directly addressed in the documents, but..."

Remember: It's better to say "I don't know" than to provide inaccurate information."""


SYSTEM_PROMPT_BALANCED = """You are a helpful assistant that answers questions using the provided context.

GUIDELINES:
1. Base your answers primarily on the provided context
2. If the context is insufficient, clearly state this limitation
3. Be accurate and cite the context where relevant
4. Express appropriate uncertainty when information is incomplete
5. Be concise and helpful

If the context doesn't address the question well, say something like:
"The available documents don't directly address this question" or
"I found limited relevant information about this topic."

Always prioritize accuracy over completeness."""


SYSTEM_PROMPT_WITH_CONFIDENCE = """You are a helpful assistant that answers questions based on provided context.

CONTEXT QUALITY: {confidence_level}
{confidence_guidance}

RULES:
1. Base your answer on the provided context
2. If confidence is LOW, be extra cautious and express uncertainty
3. If confidence is MEDIUM, answer but note any limitations  
4. If confidence is HIGH, provide a direct answer with citations
5. Never invent information not in the context

If you cannot answer reliably, say: "I don't have enough information in the available documents to answer this accurately."
"""


def get_confidence_guidance(confidence: str) -> str:
    """Get guidance text based on confidence level."""
    guidance = {
        "high": (
            "The retrieved context appears highly relevant. "
            "You can provide a confident answer based on the sources."
        ),
        "medium": (
            "The retrieved context is moderately relevant. "
            "Answer based on what's available but note if information is limited."
        ),
        "low": (
            "The retrieved context has limited relevance. "
            "Be very cautious. If unsure, state that clearly."
        )
    }
    return guidance.get(confidence, guidance["low"])


def build_prompt_with_context(
    query: str,
    context_chunks: list,
    confidence: str = "medium",
    include_scores: bool = True
) -> str:
    """
    Build a complete prompt with context and metadata.
    
    Args:
        query: User's question
        context_chunks: List of (content, score) tuples or RetrievalResult objects
        confidence: Confidence level (low/medium/high)
        include_scores: Whether to show relevance scores
        
    Returns:
        Formatted prompt string
    """
    # Format context sections
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        if hasattr(chunk, 'content'):
            content = chunk.content
            score = chunk.score
        else:
            content, score = chunk
        
        if include_scores:
            context_parts.append(f"[Source {i}] (relevance: {score:.0%})\n{content}")
        else:
            context_parts.append(f"[Source {i}]\n{content}")
    
    formatted_context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""CONTEXT:
{formatted_context}

---

QUESTION: {query}

Provide a helpful answer based on the context above. If the context doesn't contain relevant information, say so clearly."""

    return prompt
