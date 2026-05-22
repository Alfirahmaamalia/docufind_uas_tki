"""
Evaluation module for STKI retrieval system.
Computes Precision@K and Mean Average Precision (MAP).
"""
from evaluation.ground_truth import get_ground_truth, determine_relevance
from services.semantic_scholar import fetch_papers
from services.retrieval import retrieve


def precision_at_k(retrieved_papers, relevant_keywords, k):
    """
    Compute Precision@K.
    
    Precision@K = (# relevant documents in top-K) / K
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Cutoff rank.
        
    Returns:
        float — Precision@K value between 0 and 1.
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_papers[:k]
    relevant_count = sum(
        1 for paper in top_k 
        if determine_relevance(paper, relevant_keywords)
    )
    return relevant_count / k


def average_precision(retrieved_papers, relevant_keywords, k=None):
    """
    Compute Average Precision (AP) for a single query.
    
    AP = (1/R) * Σ (Precision@i * rel(i))
    where R = total relevant docs, rel(i) = 1 if doc i is relevant.
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Optional cutoff. If None, uses all retrieved papers.
        
    Returns:
        float — Average Precision value between 0 and 1.
    """
    if not retrieved_papers:
        return 0.0
    
    if k:
        retrieved_papers = retrieved_papers[:k]
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, paper in enumerate(retrieved_papers, start=1):
        if determine_relevance(paper, relevant_keywords):
            relevant_count += 1
            precision_sum += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / relevant_count


def mean_average_precision(all_results):
    """
    Compute Mean Average Precision (MAP) across all queries.
    
    MAP = (1/Q) * Σ AP(q) for each query q.
    
    Args:
        all_results: List of dicts with 'ap' keys.
        
    Returns:
        float — MAP value between 0 and 1.
    """
    if not all_results:
        return 0.0
    
    ap_values = [r['ap'] for r in all_results]
    return sum(ap_values) / len(ap_values)


def run_full_evaluation(model, top_k=10, api_limit=50):
    """
    Run complete evaluation across all ground truth queries.
    
    For each test query:
        1. Fetch papers from Semantic Scholar API
        2. Run SBERT retrieval
        3. Determine relevance using keyword matching
        4. Compute Precision@5, Precision@10, and Average Precision
    
    Args:
        model: Loaded SBERTModel instance.
        top_k: Top-K for retrieval.
        api_limit: Number of papers to fetch per query.
        
    Returns:
        dict with keys:
        - 'per_query': List of per-query results
        - 'map': Overall MAP score
        - 'avg_precision_at_5': Average P@5
        - 'avg_precision_at_10': Average P@10
        - 'num_queries': Number of test queries
    """
    ground_truth = get_ground_truth()
    per_query_results = []
    
    for gt_entry in ground_truth:
        query = gt_entry['query']
        relevant_keywords = gt_entry['relevant_keywords']
        description = gt_entry['description']
        
        print(f"[Eval] Evaluating query: '{query}'")
        
        # Fetch papers
        papers = fetch_papers(query, limit=api_limit)
        
        if not papers:
            per_query_results.append({
                'query': query,
                'description': description,
                'precision_at_5': 0.0,
                'precision_at_10': 0.0,
                'ap': 0.0,
                'num_retrieved': 0,
                'num_relevant': 0,
                'results_preview': [],
            })
            continue
        
        # Run retrieval
        results = retrieve(query, papers, model, top_k=max(top_k, 10))
        
        # Build paper-like dicts from results for evaluation
        result_papers = []
        for r in results:
            result_papers.append({
                'title': r['title'],
                'abstract': r.get('abstractFull', r.get('abstract', '')),
                'paperId': r.get('paperId', ''),
            })
        
        # Compute metrics
        p_at_5 = precision_at_k(result_papers, relevant_keywords, 5)
        p_at_10 = precision_at_k(result_papers, relevant_keywords, 10)
        ap = average_precision(result_papers, relevant_keywords)
        
        # Count relevant in results
        num_relevant = sum(
            1 for p in result_papers 
            if determine_relevance(p, relevant_keywords)
        )
        
        # Preview of results with relevance labels
        results_preview = []
        for i, (r, rp) in enumerate(zip(results[:10], result_papers[:10])):
            is_rel = determine_relevance(rp, relevant_keywords)
            results_preview.append({
                'rank': i + 1,
                'title': r['title'],
                'score': r['similarityScore'],
                'relevant': is_rel,
            })
        
        per_query_results.append({
            'query': query,
            'description': description,
            'precision_at_5': round(p_at_5, 4),
            'precision_at_10': round(p_at_10, 4),
            'ap': round(ap, 4),
            'num_retrieved': len(results),
            'num_relevant': num_relevant,
            'results_preview': results_preview,
        })
    
    # Compute overall metrics
    map_score = mean_average_precision(per_query_results)
    avg_p5 = sum(r['precision_at_5'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    avg_p10 = sum(r['precision_at_10'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    
    return {
        'per_query': per_query_results,
        'map': round(map_score, 4),
        'avg_precision_at_5': round(avg_p5, 4),
        'avg_precision_at_10': round(avg_p10, 4),
        'num_queries': len(per_query_results),
    }
