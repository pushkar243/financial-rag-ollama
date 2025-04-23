from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedder, faiss_store):
        self.embedder = embedder
        self.faiss_store = faiss_store

    def retrieve(self, query: str, chunks: List[str], k=5) -> List[str]:
        logger.info(f"Retrieving top {k} chunks")
        query_embedding = self.embedder.embed([query])
        _, indices = self.faiss_store.search(query_embedding, k)
        return [chunks[i] for i in indices[0]]