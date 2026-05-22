"""
SBERT-based document retrieval with cosine similarity.
Core retrieval engine for the STKI system.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.preprocessing import preprocess, preprocess_for_embedding, truncate_text, format_authors
from services.semantic_scholar import get_paper_url


def retrieve(query, papers, model, top_k=10):
    """
    Perform semantic retrieval using Sentence-BERT + Cosine Similarity.
    
    Algorithm:
        1. Preprocess query (lightweight, BERT handles semantics)
        2. Combine each paper's title + abstract as document text
        3. Encode query and all documents with SBERT
        4. Compute cosine similarity between query embedding and doc embeddings
        5. Rank documents by similarity score (descending)
        6. Return top-k results
    
    Args:
        query: User search query string.
        papers: List of paper dicts from Semantic Scholar API.
        model: SBERTModel instance (must be loaded).
        top_k: Number of top results to return.
        
    Returns:
        List of result dicts with keys:
        - rank, title, abstract, authors, year, citationCount, 
          similarityScore, url, paperId
    """
    if not papers or not query.strip():
        return []
    
    # Step 1: Preprocess query
    query_preprocessed = preprocess(query)
    if not query_preprocessed:
        query_preprocessed = query.strip().lower()
    
    # Step 2: Prepare document texts (title + abstract for rich embedding)
    doc_texts = []
    for paper in papers:
        combined = preprocess_for_embedding(paper.get('title', ''), paper.get('abstract', ''))
        doc_texts.append(combined)
    
    # Step 3: Encode query and documents with SBERT
    # For query, use the original (lightly cleaned) text for better BERT understanding
    query_for_encoding = f"{query.strip()}"
    query_embedding = model.encode_single(query_for_encoding)
    doc_embeddings = model.encode(doc_texts, batch_size=32)
    
    # Step 4: Compute cosine similarity
    # Reshape query to 2D for sklearn
    query_embedding_2d = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_2d, doc_embeddings).flatten()
    
    # Step 5: Rank by similarity (descending)
    ranked_indices = np.argsort(similarities)[::-1]
    
    # Step 6: Build top-k results
    results = []
    for rank_num, idx in enumerate(ranked_indices[:top_k], start=1):
        paper = papers[idx]
        score = float(similarities[idx])
        
        results.append({
            'rank': rank_num,
            'title': paper.get('title', 'Untitled'),
            'abstract': truncate_text(paper.get('abstract', ''), max_len=300),
            'abstractFull': paper.get('abstract', ''),
            'authors': format_authors(paper.get('authors', [])),
            'year': paper.get('year', 'N/A'),
            'citationCount': paper.get('citationCount', 0) or 0,
            'similarityScore': round(score, 4),
            'url': get_paper_url(paper),
            'paperId': paper.get('paperId', ''),
        })
    
    return results


def retrieve_with_details(query, papers, model, top_k=10):
    """
    Extended retrieval that also returns preprocessing and embedding info
    for display/debugging purposes.
    
    Returns:
        Tuple of (results_list, details_dict)
    """
    query_preprocessed = preprocess(query)
    
    results = retrieve(query, papers, model, top_k)
    
    details = {
        'original_query': query,
        'preprocessed_query': query_preprocessed,
        'total_papers_fetched': len(papers),
        'top_k': top_k,
        'model_name': model.model_name,
        'method': 'Cosine Similarity',
        'num_results': len(results),
    }
    
    return results, details
