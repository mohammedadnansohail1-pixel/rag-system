"""Text utility functions."""
import re


def is_garbage_text(text: str) -> bool:
    """
    Detect if text is likely binary/encoded garbage or non-useful metadata.
    
    Catches:
    - Base64 encoded data
    - Binary data rendered as text
    - Embedded images/graphics from PDFs/SEC filings
    - XBRL/XML taxonomy metadata
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be garbage
    """
    if len(text) < 50:
        return False
    
    sample = text[:500]
    sample_lower = sample.lower()
    
    # 1. XBRL/XML metadata (SEC filings) - check first as it's common
    xbrl_markers = ["xbrl", "taxonomy", "linkbase", "ex-101", "_pre.xml", "_lab.xml", "_def.xml"]
    if any(marker in sample_lower for marker in xbrl_markers):
        return True
    
    # 2. Few spaces (encoded data is dense)
    space_ratio = sample.count(" ") / len(sample)
    
    # 3. High special char ratio
    special_chars = sum(1 for c in sample if c in "*+/=\\[]{}|@$^&")
    special_ratio = special_chars / len(sample)
    
    # 4. Repeating patterns (HHHH, BBB@, ****, etc)
    repeats = len(re.findall(r"(.)\1{2,}", sample))
    
    # 5. Very low word count (real text has words)
    words = len(sample.split())
    word_ratio = words / (len(sample) / 5)  # Expect ~5 chars per word
    
    # Decision rules
    if space_ratio < 0.05 and special_ratio > 0.05:
        return True
    
    if repeats > 10:
        return True
    
    if word_ratio < 0.1 and special_ratio > 0.1:
        return True
        
    if space_ratio < 0.1 and special_ratio > 0.15:
        return True
    
    return False


def clean_text(text: str) -> str:
    """
    Clean text for embedding.
    
    - Removes non-printable characters
    - Normalizes whitespace
    - Strips leading/trailing whitespace
    """
    # Keep printable chars + common whitespace
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
