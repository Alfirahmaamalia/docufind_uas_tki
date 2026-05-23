# -*- coding: utf-8 -*-
"""
STKI - Sistem Temu Kembali Informasi
Flask Web Application - User Interface & API
Model training and evaluation are in models/integrated_ir_system.py

Two-step process:
1. python models/integrated_ir_system.py  → Train models
2. python app.py                          → Run web server
"""
import os
import sys
import json
import re
import hashlib
import time
import requests

# Fix Windows encoding for emoji/unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, render_template, request, jsonify
from models.integrated_ir_system import get_ir_system, run_full_evaluation
from config import DEFAULT_TOP_K, TOP_K_OPTIONS, CACHE_DIR, STOPWORDS, SEMANTIC_SCHOLAR_API_URL, SEMANTIC_SCHOLAR_FIELDS, DEFAULT_API_LIMIT, DB_PATH, get_db_connection

print("\n" + "="*70)
print("  [STKI] Sistem Temu Kembali Informasi")
print("  Aplikasi Web Pencarian Artikel Ilmiah")
print("  Database: SQLite (cache/stki.db)")
print("="*70)

# Initialize Flask app
app = Flask(__name__)

# ========== DATABASE STATUS CHECK ==========
def check_models_trained():
    """Check if models have been trained by checking SQLite"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM model_status WHERE is_loaded = 1')
        result = c.fetchone()
        conn.close()
        trained_count = result[0] if result else 0
        return trained_count >= 3  # All 3 models should be trained
    except Exception as e:
        print(f"[DB] Warning: {e}")
        return False

# Check if models are trained
models_trained = check_models_trained()
if not models_trained:
    print("\n[WARNING] Models not yet trained!")
    print("   Run: python models/integrated_ir_system.py")
    print("   Then: python app.py\n")
else:
    print("\n[OK] Models already trained and ready!\n")

# ========== CIRCUIT BREAKER FOR API RATE LIMITING ==========
_api_circuit_broken = False

# ========== FALLBACK DATASET (Backup Papers) ==========
FALLBACK_PAPERS = [
    {"paperId": "tf_nlp_01", "title": "Attention Is All You Need for Natural Language Processing", "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two translation tasks show these models to be superior in quality while being more parallelizable. This self-attention based model achieves state-of-the-art results in NLP tasks and forms the foundation of modern language models like BERT and GPT.", "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}, {"name": "Niki Parmar"}], "year": 2017, "citationCount": 98450, "url": "https://arxiv.org/abs/1706.03762", "externalIds": {"DOI": "10.48550/arXiv.1706.03762"}},
    {"paperId": "tf_nlp_02", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of natural language processing tasks.", "authors": [{"name": "Jacob Devlin"}, {"name": "Ming-Wei Chang"}, {"name": "Kenton Lee"}], "year": 2018, "citationCount": 42100, "url": "https://arxiv.org/abs/1810.04805", "externalIds": {"DOI": "10.48550/arXiv.1810.04805"}},
    {"paperId": "tf_nlp_03", "title": "Language Models are Few-Shot Learners: The GPT-3 Architecture", "abstract": "We present GPT-3, an autoregressive language model with 175 billion parameters. GPT-3 is trained on a massive natural language corpus and demonstrates that scaling up a transformer-based language model significantly improves few-shot performance. We evaluate GPT-3 on a broad range of NLP datasets, finding it performs well on translation, question-answering, and cloze tasks using self-attention layers.", "authors": [{"name": "Tom B. Brown"}, {"name": "Benjamin Mann"}, {"name": "Nick Ryder"}], "year": 2020, "citationCount": 21500, "url": "https://arxiv.org/abs/2005.14165", "externalIds": {"DOI": "10.48550/arXiv.2005.14165"}},
    {"paperId": "rl_rob_01", "title": "Deep Reinforcement Learning for Robotic Manipulation and Control", "abstract": "We present a deep reinforcement learning framework for training a robotic arm to perform complex manipulation tasks. By designing a continuous reward function, the agent learns an optimal policy for joint control. Our experiments demonstrate that the robot can successfully grasp and move arbitrary objects, bridging the gap between simulation and real-world robotics control.", "authors": [{"name": "Sergey Levine"}, {"name": "Chelsea Finn"}, {"name": "Trevor Darrell"}], "year": 2016, "citationCount": 5420, "url": "https://arxiv.org/abs/1504.00822", "externalIds": {"DOI": "10.48550/arXiv.1504.00822"}},
]

# ========== INITIALIZE IR SYSTEM ==========
print("[App] Loading IR System...")
ir_system = get_ir_system()
ir_system.load_models()
print("[App] OK - IR System ready!\n")

# ========== UTILITY FUNCTIONS ==========

def _cache_key(query, limit):
    """Generate cache filename from query hash"""
    raw = f"{query.lower().strip()}_{limit}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"papers_{h}.json")


def _load_cache(cache_path):
    """Load cached papers if fresh (<24h)"""
    if not os.path.exists(cache_path):
        return None
    try:
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > 86400:  # 24 hours
            return None
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_cache(cache_path, papers):
    """Save papers to cache"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[Cache] Warning: Could not save cache: {e}")


def fetch_papers(query, limit=DEFAULT_API_LIMIT):
    """Fetch papers from Semantic Scholar API with caching and fallback
    
    Fetches more papers than requested to ensure enough results for top-k retrieval.
    For top_k=10 request, should return at least 20-50 papers to ensure diversity.
    """
    global _api_circuit_broken
    
    # Determine actual fetch limit (fetch more to ensure top-k can be achieved)
    fetch_limit = max(limit, 100)  # Always fetch at least 100 papers for better diversity
    
    # Check cache first
    cache_path = _cache_key(query, fetch_limit)
    cached = _load_cache(cache_path)
    if cached is not None:
        print(f"[API] Using cached results for: '{query}' ({len(cached)} papers)")
        return cached[:limit]  # Return up to requested limit
    
    # Check circuit breaker
    if _api_circuit_broken:
        print(f"[API] Circuit breaker active. Using fallback for: '{query}'")
        _save_cache(cache_path, FALLBACK_PAPERS)
        return FALLBACK_PAPERS
    
    # Fetch from API
    papers = []
    offset = 0
    batch_size = 100  # Always use 100 per batch for efficiency
    
    while len(papers) < fetch_limit:
        params = {
            'query': query,
            'limit': batch_size,
            'offset': offset,
            'fields': SEMANTIC_SCHOLAR_FIELDS,
        }
        
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
                    break
                for paper in batch:
                    if paper.get('abstract') and paper.get('title'):
                        papers.append({
                            'paperId': paper.get('paperId', ''),
                            'title': paper.get('title', ''),
                            'abstract': paper.get('abstract', ''),
                            'authors': paper.get('authors', []),
                            'year': paper.get('year', 0),
                            'citationCount': paper.get('citationCount', 0),
                            'url': paper.get('url', ''),
                            'externalIds': paper.get('externalIds', {}),
                        })
                offset += batch_size
            elif response.status_code == 429:
                print("[API] Rate limited (429). Activating circuit breaker.")
                _api_circuit_broken = True
                _save_cache(cache_path, FALLBACK_PAPERS)
                return FALLBACK_PAPERS
            else:
                print(f"[API] Error {response.status_code}")
                break
        except Exception as e:
            print(f"[API] Error: {str(e)}")
            _api_circuit_broken = True
            _save_cache(cache_path, FALLBACK_PAPERS)
            return FALLBACK_PAPERS
    
    if papers:
        _save_cache(cache_path, papers)
    return papers


def format_authors(authors_list):
    """Format author list for display: 'A. Author, B. Writer, et al.'"""
    if not authors_list:
        return "Unknown Authors"
    names = [a.get('name', 'Unknown') for a in authors_list if a.get('name')]
    if not names:
        return "Unknown Authors"
    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]}, {names[1]}"
    else:
        return f"{names[0]}, {names[1]}, et al."


# ========== FLASK ROUTES ==========


# ========== EVALUATION CACHING ==========
EVAL_CACHE_FILE = os.path.join(CACHE_DIR, 'evaluation_results.json')

def load_eval_cache():
    """Load cached evaluation results from JSON file"""
    if os.path.exists(EVAL_CACHE_FILE):
        try:
            with open(EVAL_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert @ keys to safe names for Jinja2 templates
                for model_name in data:
                    if 'per_query' in data[model_name]:
                        for query_result in data[model_name]['per_query']:
                            # Convert p@5 -> p__5, etc. for Jinja2 compatibility
                            if 'p@5' in query_result:
                                query_result['p__5'] = query_result.pop('p@5')
                            if 'p@10' in query_result:
                                query_result['p__10'] = query_result.pop('p@10')
                            if 'r@10' in query_result:
                                query_result['r__10'] = query_result.pop('r@10')
                            if 'f1@10' in query_result:
                                query_result['f1__10'] = query_result.pop('f1@10')
                            if 'ndcg@10' in query_result:
                                query_result['ndcg__10'] = query_result.pop('ndcg@10')
                return data
        except:
            return None
    return None

def save_eval_cache(data):
    """Save evaluation results to cache"""
    try:
        with open(EVAL_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Eval Cache] Warning: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main search interface"""
    query = ""
    model = "sbert"
    top_k = DEFAULT_TOP_K
    results = None
    details = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        model = request.form.get('model', 'sbert')
        top_k = int(request.form.get('top_k', DEFAULT_TOP_K))

        # Validate inputs
        if model not in ['sbert', 'tfidf', 'bm25']:
            model = 'sbert'
        if top_k not in TOP_K_OPTIONS:
            top_k = DEFAULT_TOP_K

        if query:
            try:
                print(f"\n[Search] Query: '{query}' | Model: {model} | Top-K: {top_k}")
                papers = fetch_papers(query, limit=50)

                if papers:
                    print(f"[API] Fetched {len(papers)} papers")
                    results = ir_system.retrieve(query, papers, model_name=model, top_k=top_k)
                    
                    details = {
                        'original_query': query,
                        'model': model,
                        'top_k': top_k,
                        'total_papers_fetched': len(papers),
                        'num_results': len(results),
                    }
                    print(f"[Results] Retrieved {len(results)} results")
                else:
                    print(f"[API] Failed to fetch papers")
                    results = []
                    details = {
                        'error': 'Failed to fetch papers from Semantic Scholar API',
                        'query': query,
                        'model': model
                    }

            except Exception as e:
                print(f"[Error] {str(e)}")
                results = []
                details = {
                    'error': f'Error during search: {str(e)}',
                    'query': query,
                    'model': model
                }

    return render_template('index.html',
                         query=query,
                         model=model,
                         top_k=top_k,
                         results=results,
                         details=details,
                         top_k_options=TOP_K_OPTIONS,
                         models=['sbert', 'tfidf', 'bm25'])


@app.route('/evaluation', methods=['GET'])
def evaluation():
    """Evaluation page - show metrics for all models"""
    cached_results = load_eval_cache()
    return render_template('evaluation.html', results=cached_results, from_cache=cached_results is not None)


@app.route('/api/run-evaluation', methods=['POST'])
def api_run_evaluation():
    """API endpoint to run/generate evaluation results"""
    try:
        print("\n[Evaluation] Generating evaluation results...")
        
        # Call run_full_evaluation from integrated_ir_system
        results = run_full_evaluation(ir_system, fetch_papers_func=fetch_papers, top_k=10)
        
        # Save to cache
        save_eval_cache(results)
        print("[Evaluation] OK - Evaluation results generated and saved!")
        return jsonify({'status': 'success', 'data': results})
    except Exception as e:
        print(f"[Evaluation Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def api_models():
    """API endpoint to get available models"""
    return jsonify({
        'models': [
            {
                'name': 'sbert',
                'display': 'SBERT (Semantic)',
                'description': 'Sentence-BERT dengan cosine similarity - Terbaik untuk pencarian semantik'
            },
            {
                'name': 'tfidf',
                'display': 'TF-IDF',
                'description': 'Term Frequency-Inverse Document Frequency - Metode klasik yang terbukti'
            },
            {
                'name': 'bm25',
                'display': 'BM25',
                'description': 'Probabilistic ranking - Standar industri untuk IR'
            }
        ]
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'STKI IR System', 'version': '2.0'})


# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Halaman tidak ditemukan'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Kesalahan server internal'), 500


# ========== MAIN ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🌐 Starting STKI Web Application")
    print("  🔗 Open browser: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)

