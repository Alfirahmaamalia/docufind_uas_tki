"""
Semantic Scholar API client with caching.
Fetches research papers based on search queries.
"""
import os
import json
import hashlib
import time
import requests
from config import SEMANTIC_SCHOLAR_API_URL, SEMANTIC_SCHOLAR_FIELDS, CACHE_DIR, DEFAULT_API_LIMIT

# Circuit breaker flag to avoid repeatedly waiting for rate-limited API calls
_api_circuit_broken = False


def _cache_key(query, limit):
    """Generate a cache filename based on query hash."""
    raw = f"{query.lower().strip()}_{limit}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"papers_{h}.json")


def _load_cache(cache_path):
    """Load cached papers from JSON file if it exists and is fresh (< 24h)."""
    if not os.path.exists(cache_path):
        return None
    
    try:
        # Check if cache is less than 24 hours old
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > 86400:  # 24 hours
            return None
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return None


def _save_cache(cache_path, papers):
    """Save papers to JSON cache."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[Cache] Warning: Could not save cache: {e}")


def fetch_papers(query, limit=DEFAULT_API_LIMIT, max_retries=3):
    """
    Fetch papers from Semantic Scholar API.
    
    Args:
        query: Search query string.
        limit: Max number of papers to retrieve (default 50).
        max_retries: Number of retry attempts on failure.
        
    Returns:
        List of paper dicts with keys:
        - paperId, title, abstract, authors, year, citationCount, url, externalIds
        
    Papers without abstracts are filtered out.
    """
    # Check cache first
    cache_path = _cache_key(query, limit)
    cached = _load_cache(cache_path)
    if cached is not None:
        print(f"[API] Using cached results for: '{query}' ({len(cached)} papers)")
        return cached
    
    # Check if API circuit is broken
    global _api_circuit_broken
    if _api_circuit_broken:
        print(f"[API] Circuit breaker active. Loading local fallback dataset for: '{query}'")
        from services.fallback_data import FALLBACK_PAPERS
        _save_cache(cache_path, FALLBACK_PAPERS)
        return FALLBACK_PAPERS
    
    # Fetch from API
    papers = []
    offset = 0
    batch_size = min(limit, 100)  # API max per request is 100
    
    while len(papers) < limit:
        data = None
        params = {
            'query': query,
            'limit': batch_size,
            'offset': offset,
            'fields': SEMANTIC_SCHOLAR_FIELDS,
        }
        
        for attempt in range(max_retries):
            try:
                print(f"[API] Fetching papers (offset={offset}, limit={batch_size})...")
                response = requests.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    timeout=30,
                    headers={'Accept': 'application/json'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    batch = data.get('data', [])
                    
                    if not batch:
                        # No more results
                        break
                    
                    for paper in batch:
                        # Filter out papers without abstract
                        if paper.get('abstract') and paper.get('title'):
                            papers.append({
                                'paperId': paper.get('paperId', ''),
                                'title': paper.get('title', ''),
                                'abstract': paper.get('abstract', ''),
                                'authors': paper.get('authors', []),
                                'year': paper.get('year'),
                                'citationCount': paper.get('citationCount', 0),
                                'url': paper.get('url', ''),
                                'externalIds': paper.get('externalIds', {}),
                            })
                    
                    offset += batch_size
                    break  # Success, move to next batch
                    
                elif response.status_code == 429:
                    print(f"[API] Rate limited (429). Tripping circuit breaker to use fallback dataset.")
                    _api_circuit_broken = True
                    break
                    
                else:
                    print(f"[API] Error {response.status_code}: {response.text[:200]}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(1)
                    
            except requests.exceptions.Timeout:
                print(f"[API] Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"[API] Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        else:
            # All retries exhausted for this batch
            break
        
        if len(papers) >= limit or data is None or not data.get('data'):
            break
        
        # Small delay between batches to avoid rate limiting
        time.sleep(0.5)
    
    # Trim to requested limit
    papers = papers[:limit]
    
    print(f"[API] Fetched {len(papers)} papers for: '{query}'")
    
    # If API failed or rate-limited, use the local fallback dataset
    if not papers:
        print(f"[API] Warning: Semantic Scholar API failed/rate-limited. Loading local fallback dataset.")
        from services.fallback_data import FALLBACK_PAPERS
        papers = FALLBACK_PAPERS
        # We can also cache it under the query hash to avoid repeated failed hits
        _save_cache(cache_path, papers)
    else:
        # Cache successful API results
        _save_cache(cache_path, papers)
    
    return papers


def get_paper_url(paper):
    """
    Get the best URL for a paper (preferring DOI link).
    
    Args:
        paper: Paper dict from fetch_papers.
        
    Returns:
        URL string to the paper.
    """
    # Try DOI first
    external_ids = paper.get('externalIds', {})
    if external_ids and external_ids.get('DOI'):
        return f"https://doi.org/{external_ids['DOI']}"
    
    # Fallback to Semantic Scholar URL
    if paper.get('url'):
        return paper['url']
    
    # Last resort: construct from paperId
    paper_id = paper.get('paperId', '')
    if paper_id:
        return f"https://www.semanticscholar.org/paper/{paper_id}"
    
    return "#"
