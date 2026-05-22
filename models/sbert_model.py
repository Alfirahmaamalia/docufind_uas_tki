"""
Sentence-BERT model loader (singleton pattern).
Loads 'all-MiniLM-L6-v2' once and reuses across the application.
"""
import numpy as np
from config import SBERT_MODEL_NAME
from sentence_transformers import SentenceTransformer


class SBERTModel:
    """Singleton wrapper around sentence-transformers model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self):
        """Load the SBERT model. Call once at startup."""
        if self._model is None:
            print(f"[SBERT] Loading model '{SBERT_MODEL_NAME}'...")
            self._model = SentenceTransformer(SBERT_MODEL_NAME)
            print(f"[SBERT] Model loaded successfully.")
        return self
    
    def encode(self, texts, batch_size=32, show_progress=False):
        """
        Encode a list of texts into dense embeddings.
        
        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if self._model is None:
            self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text):
        """
        Encode a single text string.
        
        Args:
            text: String to encode.
            
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        if self._model is None:
            self.load()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding
    
    @property
    def model_name(self):
        return SBERT_MODEL_NAME
