"""
Text preprocessing for STKI.
Lightweight preprocessing since BERT handles semantics internally.
"""
import re
from config import STOPWORDS


def preprocess(text):
    """
    Preprocess text for SBERT encoding.
    
    Steps:
        1. Lowercase
        2. Remove punctuation (keep alphanumeric and spaces)
        3. Simple stopword removal
        4. Strip extra whitespace
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Step 3: Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    # Step 4: Rejoin
    return " ".join(words)


def preprocess_for_embedding(title, abstract):
    """
    Combine title and abstract for SBERT embedding.
    Uses light preprocessing — BERT handles semantics.
    
    Args:
        title: Paper title string.
        abstract: Paper abstract string.
        
    Returns:
        Combined text suitable for SBERT encoding.
    """
    title = title or ""
    abstract = abstract or ""
    
    # For SBERT, we keep more natural text (minimal preprocessing)
    # Just combine title and abstract with separator
    combined = f"{title.strip()}. {abstract.strip()}"
    return combined


def truncate_text(text, max_len=250):
    """
    Truncate text for display purposes.
    
    Args:
        text: Text to truncate.
        max_len: Maximum character length.
        
    Returns:
        Truncated text with ellipsis if needed.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    # Cut at the last space before max_len
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.7:
        truncated = truncated[:last_space]
    return truncated + "..."


def format_authors(authors_list):
    """
    Format a list of author dicts into a readable string.
    
    Args:
        authors_list: List of dicts with 'name' keys from Semantic Scholar API.
        
    Returns:
        Formatted author string like "A. Author, B. Writer, et al."
    """
    if not authors_list:
        return "Unknown Authors"
    
    names = [a.get('name', 'Unknown') for a in authors_list if a.get('name')]
    
    if len(names) == 0:
        return "Unknown Authors"
    elif len(names) <= 3:
        return ", ".join(names)
    else:
        return ", ".join(names[:3]) + " et al."
