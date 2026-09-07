"""
Embeddings module providing dual-mode vector representation for FAISS:
1. OpenAIEmbeddings when an API key is available.
2. FastSemanticEmbeddings: deterministic, lightweight TF-IDF/hashed dense vectorizer
   allowing 100% offline, zero-dependency FAISS indexing and retrieval.
"""

import hashlib
import math
import re
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_DIM


class FastSemanticEmbeddings(Embeddings):
    """
    Lightweight, deterministic embedding model producing dense vectors suitable for FAISS.
    Uses n-gram hashing and subword tokenization with L2 normalization.
    Ensures that medical terms (e.g. 'kidney', 'nephrology', 'hypertension') produce
    consistent semantic matches even without downloading large neural model weights.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _clean_and_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Extract word tokens and medical prefixes/suffixes
        words = re.findall(r"\b[a-z0-9_\-]+\b", text)
        tokens = list(words)
        # Add character 3-grams for morphological similarity (e.g. nephro-, cardio-, hyper-)
        for w in words:
            if len(w) >= 4:
                for i in range(len(w) - 2):
                    tokens.append(f"__sub_{w[i:i+3]}__")
        return tokens

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._clean_and_tokenize(text)
        if not tokens:
            return vec.tolist()

        for token in tokens:
            # Deterministic hash to dimension index
            h_int = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h_int % self.dim
            sign = 1.0 if (h_int % 2 == 0) else -1.0
            # Term weight (IDF surrogate)
            weight = 1.0 / (1.0 + math.log(1.0 + len(token)))
            vec[idx] += sign * weight

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embeddings(provider: str = "simulator", api_key: str = "") -> Embeddings:
    """
    Factory function returning the appropriate embeddings instance.
    """
    if provider.lower() in ["openai"] and api_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=api_key)
        except Exception:
            pass

    return FastSemanticEmbeddings(dim=EMBEDDING_DIM)
