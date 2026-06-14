"""
Configuration constants for STKI - Sistem Temu Kembali Informasi
Uses SQLite for model caching and evaluation results
"""
import os
import sqlite3

# --- SBERT Model ---
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Semantic Scholar API ---
SEMANTIC_SCHOLAR_API_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
SEMANTIC_SCHOLAR_FIELDS = 'title,abstract,authors,year,citationCount,url,externalIds,paperId'
DEFAULT_API_LIMIT = 50  # Number of papers to fetch per query

# --- Retrieval ---
DEFAULT_TOP_K = 10
TOP_K_OPTIONS = [5, 10, 20]

# --- File & Database Structure ---
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

DB_PATH = os.path.join(CACHE_DIR, 'stki.db')

# Redirect HF and Torch cache directories
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'hf_home')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch_home')

# --- SQLite Database Helper ---
def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database with required tables"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Model status table
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_status (
            model_name TEXT PRIMARY KEY,
            is_loaded INTEGER DEFAULT 0,
            loaded_at TEXT,
            status TEXT
        )
    ''')
    
    # Papers cache table
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            papers TEXT,
            cached_at TEXT,
            ttl_hours INTEGER DEFAULT 24
        )
    ''')
    
    # Evaluation results table
    c.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            query TEXT,
            metric_name TEXT,
            metric_value REAL,
            evaluated_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

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

# Initialize database on import
init_db()
