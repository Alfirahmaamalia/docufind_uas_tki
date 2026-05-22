"""
Ground truth data for evaluation.
Contains 10+ test queries with relevance judgments for measuring 
Precision@K and MAP of the SBERT retrieval system.

Each entry has:
    - query: The test search query
    - relevant_keywords: Keywords that should appear in relevant papers' titles/abstracts.
      Used to automatically determine relevance when we don't have fixed paper IDs
      (since Semantic Scholar results can change over time).
    - description: Human-readable description of what the query is about.
"""

GROUND_TRUTH_QUERIES = [
    {
        "query": "transformer architecture for natural language processing",
        "relevant_keywords": [
            "transformer", "attention mechanism", "nlp", "natural language",
            "bert", "gpt", "language model", "self-attention"
        ],
        "description": "Papers about Transformer architecture and its applications in NLP"
    },
    {
        "query": "reinforcement learning for robotics control",
        "relevant_keywords": [
            "reinforcement learning", "robotics", "robot", "control",
            "policy", "reward", "agent", "manipulation"
        ],
        "description": "Papers about applying RL to robotic control systems"
    },
    {
        "query": "convolutional neural network for image classification",
        "relevant_keywords": [
            "convolutional", "cnn", "image classification", "image recognition",
            "computer vision", "convolution", "visual", "object recognition"
        ],
        "description": "Papers about CNNs for image classification tasks"
    },
    {
        "query": "generative adversarial network image synthesis",
        "relevant_keywords": [
            "generative adversarial", "gan", "image generation", "image synthesis",
            "generator", "discriminator", "deep generative", "style"
        ],
        "description": "Papers about GANs for image generation and synthesis"
    },
    {
        "query": "graph neural network node classification",
        "relevant_keywords": [
            "graph neural", "gnn", "node classification", "graph convolution",
            "message passing", "graph learning", "network embedding", "node embedding"
        ],
        "description": "Papers about GNNs and node classification on graphs"
    },
    {
        "query": "federated learning privacy preserving machine learning",
        "relevant_keywords": [
            "federated learning", "privacy", "distributed learning",
            "differential privacy", "secure", "decentralized", "data privacy"
        ],
        "description": "Papers about federated learning and privacy-preserving ML"
    },
    {
        "query": "attention mechanism deep learning sequence modeling",
        "relevant_keywords": [
            "attention", "self-attention", "sequence", "encoder",
            "decoder", "transformer", "recurrent", "seq2seq"
        ],
        "description": "Papers about attention mechanisms in deep learning"
    },
    {
        "query": "neural machine translation language pairs",
        "relevant_keywords": [
            "machine translation", "neural translation", "nmt",
            "sequence to sequence", "bilingual", "translation quality", "language pair"
        ],
        "description": "Papers about neural machine translation systems"
    },
    {
        "query": "object detection autonomous driving perception",
        "relevant_keywords": [
            "object detection", "autonomous driving", "self-driving",
            "perception", "lidar", "vehicle", "pedestrian detection", "autonomous vehicle"
        ],
        "description": "Papers about perception systems for autonomous vehicles"
    },
    {
        "query": "recommendation system collaborative filtering",
        "relevant_keywords": [
            "recommendation", "collaborative filtering", "user preference",
            "matrix factorization", "recommender", "rating prediction", "item recommendation"
        ],
        "description": "Papers about recommendation systems using collaborative filtering"
    },
    {
        "query": "transfer learning domain adaptation pretrained models",
        "relevant_keywords": [
            "transfer learning", "domain adaptation", "pretrained",
            "fine-tuning", "pre-training", "feature extraction", "cross-domain"
        ],
        "description": "Papers about transfer learning and domain adaptation techniques"
    },
    {
        "query": "recurrent neural network time series forecasting",
        "relevant_keywords": [
            "recurrent neural", "rnn", "lstm", "time series",
            "forecasting", "prediction", "sequential", "gru", "temporal"
        ],
        "description": "Papers about RNNs for time series prediction"
    },
]


def determine_relevance(paper, relevant_keywords):
    """
    Determine if a paper is relevant based on keyword matching.
    A paper is considered relevant if its title or abstract contains
    at least 2 of the relevant keywords.
    
    Args:
        paper: Paper dict with 'title' and 'abstract' keys.
        relevant_keywords: List of keyword strings.
        
    Returns:
        bool — True if paper is relevant.
    """
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    matches = sum(1 for kw in relevant_keywords if kw.lower() in text)
    return matches >= 2


def get_ground_truth():
    """Return the list of ground truth query entries."""
    return GROUND_TRUTH_QUERIES
