"""
Evaluation module for STKI retrieval system.
Comprehensive metrics: Precision@K, Recall@K, MAP, NDCG, F1-score, Confusion Matrix.
"""
import numpy as np
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


def recall_at_k(retrieved_papers, relevant_keywords, k, total_relevant=None):
    """
    Compute Recall@K.
    
    Recall@K = (# relevant documents in top-K) / (total relevant documents)
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Cutoff rank.
        total_relevant: Total number of relevant documents (if known).
        
    Returns:
        float — Recall@K value between 0 and 1.
    """
    if not retrieved_papers:
        return 0.0
    
    if total_relevant is None:
        # Count relevant among all retrieved papers
        total_relevant = sum(
            1 for paper in retrieved_papers 
            if determine_relevance(paper, relevant_keywords)
        )
    
    if total_relevant == 0:
        return 0.0
    
    top_k = retrieved_papers[:k]
    relevant_count = sum(
        1 for paper in top_k 
        if determine_relevance(paper, relevant_keywords)
    )
    return relevant_count / total_relevant


def f1_at_k(retrieved_papers, relevant_keywords, k, total_relevant=None):
    """
    Compute F1-score@K.
    
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Cutoff rank.
        total_relevant: Total number of relevant documents.
        
    Returns:
        float — F1-score@K value between 0 and 1.
    """
    p_at_k = precision_at_k(retrieved_papers, relevant_keywords, k)
    r_at_k = recall_at_k(retrieved_papers, relevant_keywords, k, total_relevant)
    
    if p_at_k + r_at_k == 0:
        return 0.0
    
    return 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)


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


def ndcg_at_k(retrieved_papers, relevant_keywords, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG@K = DCG@K / IDCG@K
    where DCG = Σ (rel(i) / log2(i+1)) for i in 1..K
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Cutoff rank.
        
    Returns:
        float — NDCG@K value between 0 and 1.
    """
    if not retrieved_papers or k <= 0:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i, paper in enumerate(retrieved_papers[:k], start=1):
        rel = 1.0 if determine_relevance(paper, relevant_keywords) else 0.0
        dcg += rel / np.log2(i + 1)
    
    # Compute IDCG (ideal ranking: all relevant docs at top)
    num_relevant = sum(
        1 for paper in retrieved_papers 
        if determine_relevance(paper, relevant_keywords)
    )
    ideal_dcg = 0.0
    for i in range(min(k, num_relevant)):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    if ideal_dcg == 0.0:
        return 0.0
    
    return dcg / ideal_dcg


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


def confusion_matrix_metrics(retrieved_papers, relevant_keywords, k):
    """
    Compute confusion matrix metrics.
    
    Args:
        retrieved_papers: List of paper result dicts (ranked).
        relevant_keywords: List of keywords for determining relevance.
        k: Cutoff rank.
        
    Returns:
        dict with TP, FP, TN, FN counts and derived metrics.
    """
    top_k = retrieved_papers[:k]
    
    # Count true positives, false positives
    tp = sum(1 for p in top_k if determine_relevance(p, relevant_keywords))
    fp = len(top_k) - tp
    
    # For TN and FN, we consider the remaining documents (those not in top-k)
    # TN = not retrieved and not relevant
    # FN = relevant but not retrieved
    remaining = retrieved_papers[k:]
    fn = sum(1 for p in remaining if determine_relevance(p, relevant_keywords))
    tn = len(remaining) - fn
    
    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
    }


def run_full_evaluation(model, top_k=10, api_limit=50):
    """
    Run complete evaluation across all ground truth queries.
    
    For each test query:
        1. Fetch papers from Semantic Scholar API
        2. Run SBERT retrieval
        3. Determine relevance using keyword matching
        4. Compute comprehensive metrics:
           - Precision@K, Recall@K, F1@K
           - Average Precision (AP)
           - NDCG@K
           - Confusion Matrix metrics
    
    Args:
        model: Loaded SBERTModel instance.
        top_k: Top-K for retrieval.
        api_limit: Number of papers to fetch per query.
        
    Returns:
        dict with comprehensive evaluation results.
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
                'recall_at_10': 0.0,
                'f1_at_10': 0.0,
                'ap': 0.0,
                'ndcg_at_10': 0.0,
                'confusion_matrix': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
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
        
        # Compute all metrics
        p_at_5 = precision_at_k(result_papers, relevant_keywords, 5)
        p_at_10 = precision_at_k(result_papers, relevant_keywords, 10)
        
        # Count total relevant for recall computation
        total_rel = sum(
            1 for p in result_papers 
            if determine_relevance(p, relevant_keywords)
        )
        
        r_at_10 = recall_at_k(result_papers, relevant_keywords, 10, total_rel)
        f1_at_10 = f1_at_k(result_papers, relevant_keywords, 10, total_rel)
        ap = average_precision(result_papers, relevant_keywords)
        ndcg = ndcg_at_k(result_papers, relevant_keywords, 10)
        cm = confusion_matrix_metrics(result_papers, relevant_keywords, 10)
        
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
            'recall_at_10': round(r_at_10, 4),
            'f1_at_10': round(f1_at_10, 4),
            'ap': round(ap, 4),
            'ndcg_at_10': round(ndcg, 4),
            'confusion_matrix': cm,
            'num_retrieved': len(results),
            'num_relevant': num_relevant,
            'results_preview': results_preview,
        })
    
    # Compute overall metrics
    map_score = mean_average_precision(per_query_results)
    avg_p5 = sum(r['precision_at_5'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    avg_p10 = sum(r['precision_at_10'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    avg_r10 = sum(r['recall_at_10'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    avg_f1 = sum(r['f1_at_10'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    avg_ndcg = sum(r['ndcg_at_10'] for r in per_query_results) / len(per_query_results) if per_query_results else 0
    
    return {
        'per_query': per_query_results,
        'map': round(map_score, 4),
        'avg_precision_at_5': round(avg_p5, 4),
        'avg_precision_at_10': round(avg_p10, 4),
        'avg_recall_at_10': round(avg_r10, 4),
        'avg_f1_at_10': round(avg_f1, 4),
        'avg_ndcg_at_10': round(avg_ndcg, 4),
        'num_queries': len(per_query_results),
    }
