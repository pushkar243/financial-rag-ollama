import ollama
import logging
from typing import List

logger = logging.getLogger(__name__)

class OllamaGenerator:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        
    def generate(self, query: str, context: List[str]) -> str:
        logger.info("Generating response with Ollama")
        context_str = "\n".join(context)
        prompt = f"""Answer the question based on the context below.
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:"""
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        return response['response']