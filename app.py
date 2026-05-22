"""
STKI - Sistem Temu Kembali Informasi
Flask application entry point.

Uses Sentence-BERT (all-MiniLM-L6-v2) with Cosine Similarity
for semantic document retrieval from Semantic Scholar API.
"""
import os
import json
from flask import Flask, render_template, request, jsonify
from models.sbert_model import SBERTModel
from services.semantic_scholar import fetch_papers
from services.retrieval import retrieve_with_details
from evaluation.evaluator import run_full_evaluation
from evaluation.ground_truth import get_ground_truth
from config import DEFAULT_TOP_K, TOP_K_OPTIONS, CACHE_DIR

# --- Flask App ---
app = Flask(__name__)

# --- Load SBERT Model at startup ---
print("=" * 60)
print("  STKI - Sistem Temu Kembali Informasi")
print("  Model: Sentence-BERT (all-MiniLM-L6-v2)")
print("  Similarity: Cosine Similarity")
print("=" * 60)

sbert_model = SBERTModel()
sbert_model.load()

# --- Store evaluation results in memory ---
eval_cache = {'data': None}

# --- Evaluation cache file ---
EVAL_CACHE_FILE = os.path.join(CACHE_DIR, 'evaluation_results.json')


def load_eval_cache():
    """Load cached evaluation results from disk."""
    if os.path.exists(EVAL_CACHE_FILE):
        try:
            with open(EVAL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_eval_cache(data):
    """Save evaluation results to disk."""
    try:
        with open(EVAL_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Warning: Could not save evaluation cache: {e}")


# ==========================================
# ROUTES
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main search page.
    GET: Display search form.
    POST: Perform search and display results.
    """
    query = ""
    top_k = DEFAULT_TOP_K
    results = None
    details = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        top_k = int(request.form.get('top_k', DEFAULT_TOP_K))

        # Clamp top_k to valid options
        if top_k not in TOP_K_OPTIONS:
            top_k = DEFAULT_TOP_K

        if query:
            # Step 1: Fetch papers from Semantic Scholar API
            papers = fetch_papers(query, limit=50)

            if papers:
                # Step 2: Perform SBERT retrieval with cosine similarity
                results, details = retrieve_with_details(
                    query, papers, sbert_model, top_k=top_k
                )
            else:
                results = []
                details = {
                    'original_query': query,
                    'preprocessed_query': query,
                    'total_papers_fetched': 0,
                    'top_k': top_k,
                    'model_name': sbert_model.model_name,
                    'method': 'Cosine Similarity',
                    'num_results': 0,
                }

    return render_template(
        'index.html',
        active_page='search',
        query=query,
        top_k=top_k,
        results=results,
        details=details,
    )


@app.route('/search', methods=['POST'])
def search_api():
    """
    AJAX search endpoint.
    Accepts JSON body: { "query": "...", "top_k": 10 }
    Returns JSON results.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    query = data.get('query', '').strip()
    top_k = int(data.get('top_k', DEFAULT_TOP_K))

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    if top_k not in TOP_K_OPTIONS:
        top_k = DEFAULT_TOP_K

    # Fetch and retrieve
    papers = fetch_papers(query, limit=50)

    if not papers:
        return jsonify({
            'results': [],
            'details': {
                'original_query': query,
                'total_papers_fetched': 0,
                'num_results': 0,
            }
        })

    results, details = retrieve_with_details(query, papers, sbert_model, top_k=top_k)

    return jsonify({
        'results': results,
        'details': details,
    })


@app.route('/evaluation', methods=['GET'])
def evaluation_page():
    """
    Evaluation page.
    Displays cached evaluation results or a button to run evaluation.
    """
    # Try to load from memory cache first, then disk
    eval_data = eval_cache.get('data')
    if eval_data is None:
        eval_data = load_eval_cache()
        if eval_data:
            eval_cache['data'] = eval_data

    num_queries = len(get_ground_truth())

    return render_template(
        'evaluation.html',
        active_page='evaluation',
        eval_data=eval_data,
        num_queries=num_queries,
    )


@app.route('/api/evaluate', methods=['POST'])
def run_evaluation_api():
    """
    AJAX endpoint to run full evaluation.
    This can take several minutes as it processes 12 queries.
    Returns JSON with evaluation metrics.
    """
    try:
        print("\n" + "=" * 60)
        print("  Running Full Evaluation...")
        print("=" * 60)

        eval_data = run_full_evaluation(sbert_model, top_k=10, api_limit=50)

        # Cache results
        eval_cache['data'] = eval_data
        save_eval_cache(eval_data)

        print(f"\n  Evaluation Complete!")
        print(f"  MAP: {eval_data['map']}")
        print(f"  Avg P@5: {eval_data['avg_precision_at_5']}")
        print(f"  Avg P@10: {eval_data['avg_precision_at_10']}")
        print("=" * 60 + "\n")

        return jsonify({'success': True, 'data': eval_data})

    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
