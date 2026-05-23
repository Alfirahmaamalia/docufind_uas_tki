"""
STKI - Sistem Temu Kembali Informasi
Integrated IR System: Preprocessing + Models (SBERT/TF-IDF/BM25) + Evaluation
All in one file for simplicity

This script can be run standalone: python models/integrated_ir_system.py
It will train all models and save status to SQLite database
"""

import re
import os
import sys
import json
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOPWORDS, get_db_connection

def preprocess(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Step 3: Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    # Step 4: Rejoin
    return " ".join(words)


def preprocess_for_embedding(title, abstract):
    """
    Combine title and abstract for embedding
    
    Args:
        title: Paper title
        abstract: Paper abstract
        
    Returns:
        Combined text for embedding
    """
    title = title or ""
    abstract = abstract or ""
    combined = f"{title.strip()}. {abstract.strip()}"
    return combined


# ==================== MODEL 1: SBERT ====================

class SBERTRetriever:
    """SBERT (Sentence-BERT) based retriever with cosine similarity"""
    
    def __init__(self):
        """Initialize SBERT model (lazy loading)"""
        self._model = None
        self.model_name = 'all-MiniLM-L6-v2'
    
    def load(self):
        """Load SBERT model from sentence-transformers"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[SBERT] Loading model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                print("[SBERT] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load SBERT: {e}")
                raise
        return self
    
    def encode(self, texts, batch_size=32):
        """Encode multiple texts to embeddings"""
        if self._model is None:
            self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text):
        """Encode single text"""
        if self._model is None:
            self.load()
        return self._model.encode(text, convert_to_numpy=True)
    
    def retrieve(self, query, documents, top_k=10):
        """
        Retrieve top-k documents using SBERT + Cosine Similarity with enhanced scoring
        
        Args:
            query: Search query string
            documents: List of document dicts (title, abstract, etc)
            top_k: Number of results to return
            
        Returns:
            List of ranked results with scores
        """
        if not documents or not query.strip():
            return []
        
        # Prepare document texts
        doc_texts = []
        for doc in documents:
            combined = preprocess_for_embedding(
                doc.get('title', ''),
                doc.get('abstract', '')
            )
            doc_texts.append(combined)
        
        # Encode
        query_embedding = self.encode_single(query.strip())
        doc_embeddings = self.encode(doc_texts, batch_size=32)
        
        # Compute similarity
        query_embedding_2d = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding_2d, doc_embeddings).flatten()
        
        # Enhance scores with citation count and keyword matching
        enhanced_scores = self._enhance_scores(similarities, query, documents)
        
        # Rank
        ranked_indices = np.argsort(enhanced_scores)[::-1]
        
        # Build results
        results = []
        for rank_num, idx in enumerate(ranked_indices[:top_k], start=1):
            doc = documents[idx]
            score = float(enhanced_scores[idx])
            
            results.append({
                'rank': rank_num,
                'title': doc.get('title', 'Untitled'),
                'abstract': doc.get('abstract', '')[:300],
                'abstractFull': doc.get('abstract', ''),
                'authors': doc.get('authors', []),
                'year': doc.get('year', 'N/A'),
                'citationCount': doc.get('citationCount', 0),
                'similarityScore': round(score, 4),
                'url': doc.get('url', ''),
                'paperId': doc.get('paperId', ''),
                'model': 'SBERT'
            })
        
        return results
    
    def _enhance_scores(self, base_scores, query, documents):
        """
        Enhance SBERT similarity scores with multiple factors for better relevance
        
        Scoring formula:
        - Base SBERT similarity: 70% weight
        - Citation boost: 15% weight 
        - Keyword matching in title: 10% weight
        - Year recency bonus: 5% weight
        
        Args:
            base_scores: Cosine similarity scores from SBERT
            query: Search query
            documents: List of documents
            
        Returns:
            Enhanced scores (0-1 range)
        """
        enhanced = base_scores.copy()
        
        # Normalize for safe boosting
        max_score = max(enhanced) if len(enhanced) > 0 else 1.0
        if max_score > 0:
            base_scores_normalized = enhanced / max(max_score, 1e-6)
        
        # Citation normalization
        max_citations = max([doc.get('citationCount', 0) for doc in documents] + [1])
        
        # Query keywords for title matching
        query_terms = set(w.lower() for w in query.lower().split() if len(w) > 2)
        
        # Year range for recency
        years = [doc.get('year', 2020) for doc in documents if doc.get('year')]
        max_year = max(years) if years else 2024
        min_year = max_year - 10
        
        for i, doc in enumerate(documents):
            # Base score (70%)
            score = base_scores_normalized[i] * 0.70 if max_score > 0 else 0
            
            # Citation boost (15%) - logarithmic to avoid extreme values
            citation_ratio = doc.get('citationCount', 0) / max(max_citations, 1)
            citation_boost = (np.log1p(citation_ratio * 100) / np.log1p(100)) * 0.15
            
            # Title keyword matching (10%) - strong weight for title matches
            title = doc.get('title', '').lower()
            title_matches = sum(1 for term in query_terms if term in title)
            keyword_boost = min(0.10, (title_matches / max(len(query_terms), 1)) * 0.12)
            
            # Year recency bonus (5%) - boost for recent papers
            year = doc.get('year', 2020)
            year_score = max(0, (year - min_year) / (max_year - min_year))
            recency_bonus = year_score * 0.05
            
            # Combine all factors
            enhanced[i] = min(1.0, score + citation_boost + keyword_boost + recency_bonus)
        
        return enhanced


# ==================== MODEL 2: TF-IDF ====================

class TFIDFRetriever:
    """TF-IDF based retriever"""
    
    def __init__(self):
        """Initialize TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=1,
            max_df=0.9,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def fit(self, documents):
        """Fit vectorizer on documents"""
        if documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.is_fitted = True
    
    def retrieve(self, query, documents, top_k=10):
        """
        Retrieve top-k documents using TF-IDF + Cosine Similarity with enhanced scoring
        
        Args:
            query: Search query string
            documents: List of document dicts
            top_k: Number of results to return
            
        Returns:
            List of ranked results with scores
        """
        if not self.is_fitted or not documents:
            return []
        
        # Prepare document texts
        doc_texts = [
            f"{doc.get('title', '')}. {doc.get('abstract', '')}"
            for doc in documents
        ]
        
        # Transform
        doc_vectors = self.vectorizer.transform(doc_texts)
        query_vector = self.vectorizer.transform([query])
        
        # Similarity
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Enhance scores with citation count
        enhanced_scores = self._enhance_tfidf_scores(similarities, documents)
        
        # Rank
        ranked_indices = np.argsort(enhanced_scores)[::-1]
        
        # Build results
        results = []
        for rank_num, idx in enumerate(ranked_indices[:top_k], start=1):
            doc = documents[idx]
            score = float(enhanced_scores[idx])
            
            results.append({
                'rank': rank_num,
                'title': doc.get('title', 'Untitled'),
                'abstract': doc.get('abstract', '')[:300],
                'abstractFull': doc.get('abstract', ''),
                'authors': doc.get('authors', []),
                'year': doc.get('year', 'N/A'),
                'citationCount': doc.get('citationCount', 0),
                'similarityScore': round(score, 4),
                'url': doc.get('url', ''),
                'paperId': doc.get('paperId', ''),
                'model': 'TF-IDF'
            })
        
        return results
    
    def _enhance_tfidf_scores(self, base_scores, documents):
        """
        Enhance TF-IDF scores with multiple relevance factors
        
        Scoring formula:
        - Base TF-IDF: 65% weight
        - Citation boost: 20% weight
        - Title matching bonus: 15% weight
        """
        enhanced = base_scores.copy()
        
        # Normalize base scores
        max_score = max(enhanced) if len(enhanced) > 0 else 1.0
        if max_score > 0:
            enhanced = enhanced / max_score
        
        # Citation normalization
        max_citations = max([doc.get('citationCount', 0) for doc in documents] + [1])
        
        for i, doc in enumerate(documents):
            base = enhanced[i] * 0.65
            
            # Citation boost (20%)
            citation_ratio = doc.get('citationCount', 0) / max(max_citations, 1)
            citation_boost = (np.log1p(citation_ratio * 50) / np.log1p(50)) * 0.20
            
            # Title boost (15%) - if doc title is short and good
            title_quality = min(0.15, len(doc.get('title', '')) / 100 * 0.15)
            
            enhanced[i] = min(1.0, base + citation_boost + title_quality)
        
        return enhanced


# ==================== MODEL 3: BM25 ====================

class BM25Retriever:
    """BM25 based retriever (probabilistic model)"""
    
    def __init__(self, k1=1.5, b=0.75):
        """Initialize BM25 parameters"""
        self.k1 = k1
        self.b = b
        self.model = None
        self.tokenized_docs = None
        self.is_fitted = False
    
    def fit(self, documents):
        """Fit BM25 model on documents"""
        if documents:
            self.tokenized_docs = [
                doc.lower().split()
                for doc in documents
            ]
            self.model = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
            self.is_fitted = True
    
    def retrieve(self, query, documents, top_k=10):
        """
        Retrieve top-k documents using BM25 with enhanced scoring
        
        Args:
            query: Search query string
            documents: List of document dicts
            top_k: Number of results to return
            
        Returns:
            List of ranked results with scores
        """
        if not self.is_fitted or not documents:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get scores
        scores = self.model.get_scores(query_tokens)
        
        # Enhance scores with citation count
        enhanced_scores = self._enhance_bm25_scores(scores, documents)
        
        # Rank
        ranked_indices = np.argsort(enhanced_scores)[::-1]
        
        # Build results
        results = []
        for rank_num, idx in enumerate(ranked_indices[:top_k], start=1):
            doc = documents[idx]
            score = float(enhanced_scores[idx])
            
            results.append({
                'rank': rank_num,
                'title': doc.get('title', 'Untitled'),
                'abstract': doc.get('abstract', '')[:300],
                'abstractFull': doc.get('abstract', ''),
                'authors': doc.get('authors', []),
                'year': doc.get('year', 'N/A'),
                'citationCount': doc.get('citationCount', 0),
                'similarityScore': round(score, 4),
                'url': doc.get('url', ''),
                'paperId': doc.get('paperId', ''),
                'model': 'BM25'
            })
        
        return results
    
    def _enhance_bm25_scores(self, base_scores, documents):
        """
        Enhance BM25 scores with citation and recency factors
        
        Scoring formula:
        - Base BM25: 60% weight
        - Citation boost: 25% weight
        - Recency bonus: 15% weight
        """
        enhanced = base_scores.copy()
        
        # Normalize BM25 scores to 0-1 range
        max_score = max(enhanced) if len(enhanced) > 0 else 1.0
        if max_score > 0:
            enhanced = enhanced / max(max_score, 1e-6)
        
        # Citation normalization
        max_citations = max([doc.get('citationCount', 0) for doc in documents] + [1])
        
        # Year range
        years = [doc.get('year', 2020) for doc in documents if doc.get('year')]
        max_year = max(years) if years else 2024
        min_year = max_year - 10
        
        for i, doc in enumerate(documents):
            # Base score (60%)
            base = enhanced[i] * 0.60
            
            # Citation boost (25%) - strong weight for BM25
            citation_ratio = doc.get('citationCount', 0) / max(max_citations, 1)
            citation_boost = (np.log1p(citation_ratio * 100) / np.log1p(100)) * 0.25
            
            # Recency bonus (15%)
            year = doc.get('year', 2020)
            year_score = max(0, (year - min_year) / (max(max_year - min_year, 1)))
            recency_bonus = year_score * 0.15
            
            enhanced[i] = min(1.0, base + citation_boost + recency_bonus)
        
        return enhanced


# ==================== UNIFIED RETRIEVAL INTERFACE ====================

class IRSystem:
    """Unified Information Retrieval System with 3 models"""
    
    def __init__(self):
        """Initialize all retrievers"""
        self.sbert = SBERTRetriever()
        self.tfidf = TFIDFRetriever()
        self.bm25 = BM25Retriever()
        self.is_trained = False
    
    def train(self, documents):
        """
        Train all models on the given documents
        
        This is the explicit training step required for TF-IDF and BM25 models.
        SBERT is pre-trained and doesn't require training, but models are loaded here.
        
        Args:
            documents: List of document dicts (must have 'title' and 'abstract')
        """
        print("\n[IR System] Starting model training...")
        
        # Load SBERT model (pre-trained, just load it)
        print("[1/3] Loading SBERT (Sentence-BERT) model...")
        self.sbert.load()
        print("  ✓ SBERT loaded (pre-trained on 215M sentence pairs)")
        
        # Prepare document texts for TF-IDF and BM25
        doc_texts = [
            f"{doc.get('title', '')}. {doc.get('abstract', '')}"
            for doc in documents
        ]
        
        # Train TF-IDF
        print("[2/3] Training TF-IDF vectorizer...")
        self.tfidf.fit(doc_texts)
        print(f"  ✓ TF-IDF trained ({len(doc_texts)} documents indexed)")
        
        # Train BM25
        print("[3/3] Training BM25 probabilistic ranker...")
        self.bm25.fit(doc_texts)
        print(f"  ✓ BM25 trained ({len(doc_texts)} documents indexed)")
        
        self.is_trained = True
        print("\n[IR System] ✓ All models trained successfully!")
    
    def load_models(self):
        """Backward compatibility: load just SBERT"""
        print("[IR System] Loading SBERT model...")
        self.sbert.load()
        print("[IR System] SBERT loaded.")
    
    def retrieve(self, query, documents, model_name='sbert', top_k=10):
        """
        Retrieve documents using specified model
        
        Args:
            query: Search query
            documents: List of documents
            model_name: 'sbert', 'tfidf', or 'bm25'
            top_k: Top-K results
            
        Returns:
            List of ranked results
        """
        if model_name == 'sbert':
            return self.sbert.retrieve(query, documents, top_k)
        
        elif model_name == 'tfidf':
            # Prepare texts
            doc_texts = [
                f"{d.get('title', '')}. {d.get('abstract', '')}"
                for d in documents
            ]
            self.tfidf.fit(doc_texts)
            return self.tfidf.retrieve(query, documents, top_k)
        
        elif model_name == 'bm25':
            # Prepare texts
            doc_texts = [
                f"{d.get('title', '')}. {d.get('abstract', '')}"
                for d in documents
            ]
            self.bm25.fit(doc_texts)
            return self.bm25.retrieve(query, documents, top_k)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")


# ==================== GROUND TRUTH & EVALUATION ====================

GROUND_TRUTH_QUERIES = [
    {
        "query": "transformer architecture for natural language processing",
        "relevant_keywords": ["transformer", "attention", "nlp", "bert", "gpt", "language model", "self-attention"],
        "description": "Papers about Transformer architecture and NLP applications"
    },
    {
        "query": "reinforcement learning for robotics control",
        "relevant_keywords": ["reinforcement learning", "robotics", "robot", "control", "policy", "agent"],
        "description": "Papers about applying RL to robotic control"
    },
    {
        "query": "convolutional neural network for image classification",
        "relevant_keywords": ["cnn", "image classification", "computer vision", "convolution"],
        "description": "Papers about CNNs for image classification"
    },
    {
        "query": "generative adversarial network image synthesis",
        "relevant_keywords": ["gan", "image generation", "generator", "discriminator"],
        "description": "Papers about GANs for image generation"
    },
    {
        "query": "graph neural network node classification",
        "relevant_keywords": ["gnn", "graph", "node classification", "graph convolution"],
        "description": "Papers about GNNs"
    },
    {
        "query": "federated learning privacy preserving machine learning",
        "relevant_keywords": ["federated", "privacy", "distributed", "differential privacy"],
        "description": "Papers about federated learning"
    },
    {
        "query": "attention mechanism deep learning sequence modeling",
        "relevant_keywords": ["attention", "self-attention", "sequence", "encoder", "decoder"],
        "description": "Papers about attention mechanisms"
    },
    {
        "query": "neural machine translation language pairs",
        "relevant_keywords": ["machine translation", "nmt", "seq2seq", "translation"],
        "description": "Papers about neural machine translation"
    },
    {
        "query": "object detection autonomous driving perception",
        "relevant_keywords": ["object detection", "autonomous driving", "perception"],
        "description": "Papers about object detection for autonomous vehicles"
    },
    {
        "query": "recommendation system collaborative filtering",
        "relevant_keywords": ["recommendation", "collaborative filtering", "matrix factorization"],
        "description": "Papers about recommendation systems"
    },
    {
        "query": "transfer learning domain adaptation pretrained models",
        "relevant_keywords": ["transfer learning", "domain adaptation", "fine-tuning"],
        "description": "Papers about transfer learning"
    },
    {
        "query": "recurrent neural network sequence modeling",
        "relevant_keywords": ["rnn", "lstm", "gru", "sequence", "recurrent"],
        "description": "Papers about RNN and sequence modeling"
    },
]


def determine_relevance(paper, relevant_keywords):
    """
    Determine if paper is relevant based on keywords with improved accuracy scoring
    
    Uses multi-level keyword matching:
    - Exact phrase matching (highest weight)
    - Title keyword matching (high weight)
    - Abstract keyword matching (medium weight)
    
    Returns True if paper has sufficient keyword matches
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    
    # Count keyword matches with weighted scoring
    score = 0
    for keyword in relevant_keywords:
        kw_lower = keyword.lower()
        
        # Exact phrase match in title (weight: 3)
        if kw_lower in title:
            score += 3
        # Keyword in abstract (weight: 1)
        elif kw_lower in abstract:
            score += 1
    
    # Require score >= 3 for relevance (e.g., 1 title match + 0 abstract, or 3+ abstract matches)
    return score >= 3


def precision_at_k(papers, keywords, k):
    """Precision@K metric"""
    if k <= 0:
        return 0.0
    top_k = papers[:k]
    relevant = sum(1 for p in top_k if determine_relevance(p, keywords))
    return relevant / k


def recall_at_k(papers, keywords, k, total_relevant=None):
    """Recall@K metric"""
    if total_relevant is None:
        total_relevant = sum(1 for p in papers if determine_relevance(p, keywords))
    if total_relevant == 0:
        return 0.0
    top_k = papers[:k]
    relevant = sum(1 for p in top_k if determine_relevance(p, keywords))
    return relevant / total_relevant


def f1_at_k(papers, keywords, k, total_relevant=None):
    """F1-Score@K metric"""
    p = precision_at_k(papers, keywords, k)
    r = recall_at_k(papers, keywords, k, total_relevant)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def average_precision(papers, keywords, k=None):
    """Average Precision metric"""
    if not papers:
        return 0.0
    if k:
        papers = papers[:k]
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, paper in enumerate(papers, start=1):
        if determine_relevance(paper, keywords):
            relevant_count += 1
            precision_sum += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count


def ndcg_at_k(papers, keywords, k):
    """NDCG@K metric"""
    if not papers or k <= 0:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, paper in enumerate(papers[:k], start=1):
        rel = 1.0 if determine_relevance(paper, keywords) else 0.0
        dcg += rel / np.log2(i + 1)
    
    # IDCG
    num_relevant = sum(1 for p in papers if determine_relevance(p, keywords))
    ideal_dcg = 0.0
    for i in range(min(k, num_relevant)):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def confusion_matrix(papers, keywords, k):
    """Confusion Matrix metrics"""
    top_k = papers[:k]
    tp = sum(1 for p in top_k if determine_relevance(p, keywords))
    fp = len(top_k) - tp
    
    remaining = papers[k:]
    fn = sum(1 for p in remaining if determine_relevance(p, keywords))
    tn = len(remaining) - fn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4)
    }


def run_full_evaluation(ir_system, top_k=10):
    """
    Run full evaluation on all models
    
    Args:
        ir_system: IRSystem instance
        top_k: Top-K for retrieval
        
    Returns:
        Evaluation results dict
    """
    from services.semantic_scholar import fetch_papers
    
    results_by_model = {}
    
    for model_name in ['sbert', 'tfidf', 'bm25']:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'='*60}")
        
        per_query_results = []
        
        for i, query_data in enumerate(GROUND_TRUTH_QUERIES, 1):
            query = query_data['query']
            keywords = query_data['relevant_keywords']
            
            print(f"[{i}/12] Query: {query[:50]}...")
            
            try:
                # Fetch papers
                papers = fetch_papers(query, limit=50)
                if not papers:
                    print("  ⚠ No papers fetched")
                    continue
                
                # Retrieve
                results = ir_system.retrieve(query, papers, model_name, top_k)
                
                # Build paper dicts
                result_papers = [
                    {'title': r['title'], 'abstract': r.get('abstractFull', '')}
                    for r in results
                ]
                
                # Compute metrics
                p5 = precision_at_k(result_papers, keywords, 5)
                p10 = precision_at_k(result_papers, keywords, 10)
                total_rel = sum(1 for p in result_papers if determine_relevance(p, keywords))
                r10 = recall_at_k(result_papers, keywords, 10, total_rel)
                f1 = f1_at_k(result_papers, keywords, 10, total_rel)
                ap = average_precision(result_papers, keywords)
                ndcg = ndcg_at_k(result_papers, keywords, 10)
                cm = confusion_matrix(result_papers, keywords, 10)
                
                print(f"  ✓ P@10: {p10:.3f} | R@10: {r10:.3f} | F1@10: {f1:.3f} | NDCG: {ndcg:.3f}")
                
                per_query_results.append({
                    'query': query,
                    'p@5': round(p5, 4),
                    'p@10': round(p10, 4),
                    'r@10': round(r10, 4),
                    'f1@10': round(f1, 4),
                    'ap': round(ap, 4),
                    'ndcg@10': round(ndcg, 4),
                    'cm': cm,
                    'num_relevant': total_rel,
                    'num_retrieved': len(results)
                })
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:50]}")
        
        if per_query_results:
            map_score = sum(r['ap'] for r in per_query_results) / len(per_query_results)
            results_by_model[model_name] = {
                'per_query': per_query_results,
                'map': round(map_score, 4),
                'avg_p@10': round(sum(r['p@10'] for r in per_query_results) / len(per_query_results), 4),
                'avg_r@10': round(sum(r['r@10'] for r in per_query_results) / len(per_query_results), 4),
                'avg_f1@10': round(sum(r['f1@10'] for r in per_query_results) / len(per_query_results), 4),
                'avg_ndcg@10': round(sum(r['ndcg@10'] for r in per_query_results) / len(per_query_results), 4),
                'num_queries': len(per_query_results)
            }
    
    return results_by_model


# Singleton instance
_ir_system = IRSystem()

def get_ir_system():
    """Get singleton IR system instance"""
    return _ir_system


# ==================== DATABASE MANAGEMENT ====================

def save_model_status(model_name, is_loaded=True, status="ready"):
    """Save model status to SQLite database"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO model_status (model_name, is_loaded, loaded_at, status)
            VALUES (?, ?, ?, ?)
        ''', (model_name, 1 if is_loaded else 0, datetime.now().isoformat(), status))
        conn.commit()
        conn.close()
        print(f"  ✓ Saved {model_name} status to database")
    except Exception as e:
        print(f"  ! Warning: Could not save model status: {e}")


def get_model_status(model_name):
    """Get model status from SQLite database"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM model_status WHERE model_name = ?', (model_name,))
        result = c.fetchone()
        conn.close()
        return dict(result) if result else None
    except Exception:
        return None


# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """Main training script - can be run standalone: python models/integrated_ir_system.py"""
    print("\n" + "="*70)
    print("  🚀 STKI - Model Training & Initialization")
    print("  Sistem Temu Kembali Informasi - Training Models")
    print("="*70 + "\n")
    
    # Step 1: Initialize database
    print("[1/4] Initializing SQLite database...")
    try:
        from config import init_db
        init_db()
        print("  ✓ Database initialized\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return
    
    # Step 2: Load or create sample documents
    print("[2/4] Preparing sample documents...")
    SAMPLE_DOCS = [
        {
            "paperId": "sample_01",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism.",
            "authors": [{"name": "Ashish Vaswani"}],
            "year": 2017,
            "citationCount": 98450,
            "url": "https://arxiv.org/abs/1706.03762"
        },
        {
            "paperId": "sample_02",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
            "authors": [{"name": "Jacob Devlin"}],
            "year": 2018,
            "citationCount": 42100,
            "url": "https://arxiv.org/abs/1810.04805"
        },
        {
            "paperId": "sample_03",
            "title": "Language Models are Unsupervised Multitask Learners",
            "abstract": "Natural language processing tasks typically phrase prediction problems using specially engineered task-specific architectures and publicly demonstrate strong performance of the previously proposed task-agnostic, model-agnostic algorithm for solving NLP tasks in a zero-shot setting.",
            "authors": [{"name": "Alec Radford"}],
            "year": 2019,
            "citationCount": 21500,
            "url": "https://arxiv.org/abs/1902.10673"
        }
    ]
    print(f"  ✓ Prepared {len(SAMPLE_DOCS)} sample documents\n")
    
    # Step 3: Train IR System
    print("[3/4] Training IR System models...")
    try:
        ir_system = get_ir_system()
        ir_system.train(SAMPLE_DOCS)
        print("  ✓ All models trained successfully\n")
        
        # Save status to database
        for model_name in ['sbert', 'tfidf', 'bm25']:
            save_model_status(model_name, is_loaded=True, status="trained")
        
    except Exception as e:
        print(f"  ✗ Error during training: {e}\n")
        return
    
    # Step 4: Test retrieval
    print("[4/4] Testing retrieval system...")
    try:
        test_query = "transformer attention mechanism"
        results_sbert = ir_system.retrieve(test_query, SAMPLE_DOCS, 'sbert', top_k=3)
        results_tfidf = ir_system.retrieve(test_query, SAMPLE_DOCS, 'tfidf', top_k=3)
        results_bm25 = ir_system.retrieve(test_query, SAMPLE_DOCS, 'bm25', top_k=3)
        
        print(f"  ✓ SBERT retrieved {len(results_sbert)} documents")
        print(f"  ✓ TF-IDF retrieved {len(results_tfidf)} documents")
        print(f"  ✓ BM25 retrieved {len(results_bm25)} documents\n")
        
    except Exception as e:
        print(f"  ! Warning: Test retrieval failed: {e}\n")
    
    # Final message
    print("="*70)
    print("  ✅ Training Complete!")
    print("="*70)
    print("\n📝 Next steps:")
    print("  1. Run: python app.py")
    print("  2. Open: http://127.0.0.1:5000")
    print("  3. Search for articles!\n")


if __name__ == '__main__':
    main()
