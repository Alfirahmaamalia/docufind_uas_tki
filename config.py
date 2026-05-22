"""
Configuration constants for STKI SBERT application.
"""
import os

# --- SBERT Model ---
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Semantic Scholar API ---
SEMANTIC_SCHOLAR_API_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
SEMANTIC_SCHOLAR_FIELDS = 'title,abstract,authors,year,citationCount,url,externalIds,paperId'
DEFAULT_API_LIMIT = 50  # Number of papers to fetch per query

# --- Retrieval ---
DEFAULT_TOP_K = 10
TOP_K_OPTIONS = [5, 10, 20]

# --- Caching ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Redirect HF and Torch cache directories to D: drive to preserve space on C:
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'hf_home')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch_home')

# --- Preprocessing ---
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
    'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'about', 'up',
}
