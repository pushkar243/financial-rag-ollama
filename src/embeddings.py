from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import os

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        logger.info("Generating embeddings")
        return self.model.encode(texts)

class FAISSStore:
    def __init__(self, dimension=384):  # MiniLM dimension
        self.index = faiss.IndexFlatL2(dimension)
        
    def save(self, path: str):
        faiss.write_index(self.index, path)
        
    def load(self, path: str):
        self.index = faiss.read_index(path)
        
    def add_embeddings(self, embeddings: np.ndarray):
        self.index.add(embeddings)
        
    def search(self, query_embedding: np.ndarray, k=5):
        return self.index.search(query_embedding, k)