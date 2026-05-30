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
    sys.stdout.reconfigure(encoding='utf-8')

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
    {"paperId": "cv_cnn_01", "title": "ImageNet Classification with Deep Convolutional Neural Networks", "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test set, we achieved top-1 and top-5 error rates of 37.5% and 17.0%. The network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers. Our results show that deep convolutional neural networks are highly effective at image classification tasks.", "authors": [{"name": "Alex Krizhevsky"}, {"name": "Ilya Sutskever"}, {"name": "Geoffrey E. Hinton"}], "year": 2012, "citationCount": 88000, "url": "https://arxiv.org/abs/1202.5660", "externalIds": {"DOI": "10.1145/1869790.1869829"}},
    {"paperId": "cv_resnet_01", "title": "Deep Residual Learning for Image Recognition", "abstract": "Deep neural networks are difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. Residual networks make it possible to train networks with over 1000 layers. We evaluate residual networks on ImageNet and CIFAR-10 datasets. Our results indicate that residual networks can significantly improve the training of very deep networks and avoid the degradation problem.", "authors": [{"name": "Kaiming He"}, {"name": "Xiangyu Zhang"}, {"name": "Shaoqing Ren"}], "year": 2015, "citationCount": 88000, "url": "https://arxiv.org/abs/1512.03385", "externalIds": {"DOI": "10.1109/ICCV.2015.123"}},
    {"paperId": "nlp_word2vec_01", "title": "Efficient Estimation of Word Representations in Vector Space", "abstract": "We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously published results on similar tasks. We demonstrate that our word embeddings significantly outperform previous methods and can be trained efficiently on large corpora.", "authors": [{"name": "Tomas Mikolov"}, {"name": "Kai Chen"}, {"name": "Greg Corrado"}], "year": 2013, "citationCount": 65000, "url": "https://arxiv.org/abs/1301.3781", "externalIds": {"DOI": "10.48550/arXiv.1301.3781"}},
    {"paperId": "gan_01", "title": "Generative Adversarial Networks", "abstract": "We propose a new framework for estimating generative models via an adversarial process. The framework comprises two players - a generator network and a discriminator network. We explore the framework through theoretical analysis and supported by experiments and applications to demonstrate the potential of the adversarial framework for generative modeling.", "authors": [{"name": "Ian Goodfellow"}, {"name": "Jean Pouget-Abadie"}, {"name": "Mehdi Mirza"}], "year": 2014, "citationCount": 64000, "url": "https://arxiv.org/abs/1406.2661", "externalIds": {"DOI": "10.48550/arXiv.1406.2661"}},
    {"paperId": "rnn_lstm_01", "title": "Long Short-Term Memory", "abstract": "Learning to store information over extended time intervals using recurrent neural networks is difficult due to the vanishing gradient problem. We propose a method for solving this problem by learning when to forget information. We introduce cells with self-connections and learn when to read, write, and erase content in these cells. This enables networks to store accurate information about the problem over an extended period of time.", "authors": [{"name": "Sepp Hochreiter"}, {"name": "Jurgen Schmidhuber"}], "year": 1997, "citationCount": 42000, "url": "https://arxiv.org/abs/1402.1128", "externalIds": {"DOI": "10.1162/neco.1997.9.8.1735"}},
    {"paperId": "opt_adam_01", "title": "Adam: A Method for Stochastic Optimization", "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions. The method combines the advantages of AdaGrad and RMSProp. Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. We provide empirical results showing that Adam works well in practice compared with other stochastic optimization methods.", "authors": [{"name": "Diederik P. Kingma"}, {"name": "Jimmy Ba"}], "year": 2014, "citationCount": 61000, "url": "https://arxiv.org/abs/1412.6980", "externalIds": {"DOI": "10.48550/arXiv.1412.6980"}},
    {"paperId": "nlp_seq2seq_01", "title": "Sequence to Sequence Learning with Neural Networks", "abstract": "Deep neural networks have recently been very successful on supervised sequence labeling tasks such as speech recognition and machine translation. However, these networks are restricted to learning mappings between fixed-sized input and output vectors. We propose an end-to-end approach using an LSTM to map the input sequence to a vector of fixed dimensionality, and then another LSTM to decode the target sequence from the vector.", "authors": [{"name": "Ilya Sutskever"}, {"name": "Oriol Vinyals"}, {"name": "Quoc V. Le"}], "year": 2014, "citationCount": 36000, "url": "https://arxiv.org/abs/1409.3215", "externalIds": {"DOI": "10.48550/arXiv.1409.3215"}},
    {"paperId": "dropout_01", "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "abstract": "Deep neural networks contain multiple levels of representation which give them a lot of flexibility and power. However, this flexibility makes them prone to overfitting. We introduce dropout, a technique for addressing this problem. The key idea is to randomly drop units from the network during training to prevent complex co-adaptations on training data. We apply dropout to various neural network architectures and show that the method significantly reduces overfitting.", "authors": [{"name": "Nitish Srivastava"}, {"name": "Geoffrey Hinton"}, {"name": "Alex Krizhevsky"}], "year": 2014, "citationCount": 45000, "url": "https://arxiv.org/abs/1207.0580", "externalIds": {"DOI": "10.48550/arXiv.1207.0580"}},
    {"paperId": "bn_01", "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "abstract": "Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization. We address this problem by introducing Batch Normalization, a technique to normalize the inputs of each layer, which allows us to use much higher learning rates and be less careful about initialization.", "authors": [{"name": "Sergey Ioffe"}, {"name": "Christian Szegedy"}], "year": 2015, "citationCount": 55000, "url": "https://arxiv.org/abs/1502.03167", "externalIds": {"DOI": "10.48550/arXiv.1502.03167"}},
    {"paperId": "xception_01", "title": "Xception: Deep Learning with Depthwise Separable Convolutions", "abstract": "We present an interpretation of Inception modules in convolutional neural networks as being equivalent to depthwise separable convolutions. We use this observation to propose a new deep convolutional neural network architecture inspired by the Inception modules, called Xception. The Xception architecture has 36 convolutional layers forming a deep network. We evaluate the proposed architecture on ImageNet and show improvements in model accuracy.", "authors": [{"name": "Francois Chollet"}], "year": 2016, "citationCount": 19000, "url": "https://arxiv.org/abs/1610.02357", "externalIds": {"DOI": "10.1109/ICCV.2017.195"}},
    {"paperId": "attention_01", "title": "Neural Machine Translation by Jointly Learning to Align and Translate", "abstract": "In this paper, we conjecture that the difficulty in achieving a high performance neural machine translation system comes from the difficulty of learning to translate and aligning in a single neural network with a fixed-length context window. We propose to use a mechanism to search for a set of source words, the representation of which is expected to contain most of the information about the foreign word to be predicted. We show the proposed method significantly outperforms the baseline model.", "authors": [{"name": "Dzmitry Bahdanau"}, {"name": "Kyungyoon Cho"}, {"name": "Yoshua Bengio"}], "year": 2014, "citationCount": 45000, "url": "https://arxiv.org/abs/1409.0473", "externalIds": {"DOI": "10.48550/arXiv.1409.0473"}},
    {"paperId": "vgg_01", "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "abstract": "In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. We make the main contribution by rigorously evaluating networks of increasing depth using an architecture with very small 3x3 convolution filters. We find that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of a very successful submission to the ILSVRC-2014 competition.", "authors": [{"name": "Karen Simonyan"}, {"name": "Andrew Zisserman"}], "year": 2014, "citationCount": 68000, "url": "https://arxiv.org/abs/1409.1556", "externalIds": {"DOI": "10.48550/arXiv.1409.1556"}},
    {"paperId": "inception_01", "title": "Going Deeper with Convolutions", "abstract": "We propose a deep convolutional neural network architecture codenamed Inception. This architecture is designed to be computationally efficient while achieving excellent accuracy. The proposed network contains 22 layers and consists of carefully designed modules that use parallel convolutions with multiple kernel sizes. We demonstrate the effectiveness of the proposed architecture on the ImageNet dataset.", "authors": [{"name": "Christian Szegedy"}, {"name": "Wei Liu"}, {"name": "Yangqing Jia"}], "year": 2014, "citationCount": 32000, "url": "https://arxiv.org/abs/1409.4842", "externalIds": {"DOI": "10.1109/CVPR.2015.7298594"}},
    {"paperId": "glove_01", "title": "GloVe: Global Vectors for Word Representation", "abstract": "Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using relatively simple model architectures, while attaining good performance on word analogy tasks. However, the vectors learned by these methods relatedly poorly on word similarity tasks and other downstream applications. We propose GloVe, an unsupervised learning algorithm for obtaining vector representations for words.", "authors": [{"name": "Jeffrey Pennington"}, {"name": "Richard Socher"}, {"name": "Christopher D. Manning"}], "year": 2014, "citationCount": 23000, "url": "https://arxiv.org/abs/1504.06654", "externalIds": {"DOI": "10.3115/v1/D14-1162"}},
    {"paperId": "mobilenet_01", "title": "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", "abstract": "We present MobileNets, efficient convolutional neural networks optimized for mobile vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. We benchmark MobileNets across different width multipliers and resolutions and compare them with popular single-unit and multi-unit deep learning models.", "authors": [{"name": "Andrew G. Howard"}, {"name": "Mengqi Zhang"}, {"name": "Boyang Li"}], "year": 2017, "citationCount": 19000, "url": "https://arxiv.org/abs/1704.04861", "externalIds": {"DOI": "10.48550/arXiv.1704.04861"}},
    {"paperId": "fasttext_01", "title": "Enriching Word Vectors with Subword Information", "abstract": "We present a method for computing word representations in a fast and simple way, using subword information. The key advantage is that word representations can be computed for out-of-vocabulary (OOV) words at test time by summing the representations of subwords. We evaluate FastText on word similarity and analogy tasks, and show it achieves state-of-the-art results on morphologically rich languages.", "authors": [{"name": "Piotr Bojanowski"}, {"name": "Edouard Grave"}, {"name": "Armand Joulin"}], "year": 2016, "citationCount": 16000, "url": "https://arxiv.org/abs/1607.04606", "externalIds": {"DOI": "10.48550/arXiv.1607.04606"}},
]

# ========== INITIALIZE IR SYSTEM ==========
print("[App] Loading IR System...")
ir_system = get_ir_system()
try:
    ir_system.load_models()
    print("[App] ✓ IR System ready!\n")
except Exception as e:
    print(f"[App] ⚠️ WARNING: Could not load SBERT: {e}")
    print("[App] System will use TF-IDF and BM25 models. SBERT will fallback to TF-IDF.\n")

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
# Evaluation cache file (will include top_k in filename)
def get_eval_cache_file(top_k=10):
    """Get evaluation cache file path for specific top_k value"""
    return os.path.join(CACHE_DIR, f'evaluation_results_top{top_k}.json')

def load_eval_cache(top_k=10):
    """Load cached evaluation results from JSON file and normalize keys for Jinja2"""
    cache_file = get_eval_cache_file(top_k)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Standardize keys dynamically for Jinja2 template compatibility
                for model_name in data:
                    model_data = data[model_name]
                    
                    # Convert root level keys like avg_p@10 -> avg_p_at_10, avg_p10
                    for key in list(model_data.keys()):
                        if '@' in key:
                            # Convert e.g. avg_p@10 -> avg_p_at_10
                            new_key_at = key.replace('@', '_at_')
                            model_data[new_key_at] = model_data[key]
                            
                            # Convert e.g. avg_p@10 -> avg_p10
                            new_key_flat = key.replace('@', '')
                            model_data[new_key_flat] = model_data[key]
                            
                        elif '_at_' in key:
                            # If already in _at_ format, add the flat version for templates
                            new_key_flat = key.replace('_at_', '')
                            model_data[new_key_flat] = model_data[key]
                            
                    # Convert query level keys
                    if 'per_query' in model_data:
                        for query_result in model_data['per_query']:
                            for key in list(query_result.keys()):
                                if '@' in key:
                                    # Convert p@5 -> p_at_5
                                    new_key_at = key.replace('@', '_at_')
                                    query_result[new_key_at] = query_result[key]
                                    
                                    # Convert p@5 -> p__5
                                    new_key_double = key.replace('@', '__')
                                    query_result[new_key_double] = query_result[key]
                                    
                                elif '_at_' in key:
                                    # Convert p_at_5 -> p__5 and p@5
                                    new_key_double = key.replace('_at_', '__')
                                    query_result[new_key_double] = query_result[key]
                                    new_key_old = key.replace('_at_', '@')
                                    query_result[new_key_old] = query_result[key]
                                    
                                elif '__' in key:
                                    # Convert p__5 -> p_at_5 and p@5
                                    new_key_at = key.replace('__', '_at_')
                                    query_result[new_key_at] = query_result[key]
                                    new_key_old = key.replace('__', '@')
                                    query_result[new_key_old] = query_result[key]
                return data
        except Exception as e:
            print(f"[Eval Cache] Error: {e}")
            return None
    return None

def save_eval_cache(data, top_k=10):
    """Save evaluation results to cache with top_k specific filename"""
    try:
        cache_file = get_eval_cache_file(top_k)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Eval Cache] Saved evaluation results (top_k={top_k}) to {cache_file}")
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
                    
                    # Map model code to display name
                    model_display = {
                        'sbert': 'SBERT (Semantic)',
                        'tfidf': 'TF-IDF (Vector Space)',
                        'bm25': 'BM25 (Probabilistic)'
                    }.get(model, model)
                    
                    details = {
                        'original_query': query,
                        'model': model,
                        'model_name': model_display,
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
    top_k = request.args.get('top_k', 10, type=int)
    if top_k not in [5, 10]:
        top_k = 10
    
    cached_results = load_eval_cache(top_k)
    if not isinstance(cached_results, dict):
        cached_results = None
    
    return render_template('evaluation.html', results=cached_results, from_cache=cached_results is not None, top_k=top_k, top_k_options=[5, 10])


@app.route('/api/run-evaluation', methods=['POST'])
def api_run_evaluation():
    """API endpoint to run/generate evaluation results"""
    try:
        print("\n[Evaluation] Generating evaluation results...")
        
        # Get top_k from request
        top_k = request.json.get('top_k', 10) if request.json else 10
        if top_k not in [5, 10]:
            top_k = 10
        
        print(f"[Evaluation] Running evaluation with top_k={top_k}...")
        
        # Call run_full_evaluation from integrated_ir_system
        results = run_full_evaluation(ir_system, fetch_papers_func=fetch_papers, top_k=top_k)
        
        # Verify evaluation completed properly
        if not results or not any(results.get(m, {}).get('per_query') for m in ['sbert', 'tfidf', 'bm25']):
            print(f"[Evaluation] ⚠️ WARNING: Evaluation returned empty results!")
            return jsonify({'status': 'error', 'message': 'Evaluation produced no results'}), 500
        else:
            # Count queries that were evaluated for each model
            for model_key in results:
                query_count = len(results[model_key].get('per_query', []))
                print(f"[Evaluation] ✓ {model_key.upper()}: {query_count} queries evaluated")
        
        # Save to cache with top_k in filename
        save_eval_cache(results, top_k)
        print(f"[Evaluation] ✓ Saved evaluation results (top_k={top_k}) to cache!")
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

