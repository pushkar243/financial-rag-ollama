import logging
from document_loader import PDFLoader
from text_splitter import TextSplitter
from embeddings import Embedder, FAISSStore
from retrieval import Retriever
from generation import OllamaGenerator
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize components
    pdf_loader = PDFLoader("data/financial_docs/doc1.pdf")
    text_splitter = TextSplitter()
    embedder = Embedder()
    faiss_store = FAISSStore()
    generator = OllamaGenerator()
    
    # Load and process document
    text = pdf_loader.load()
    chunks = text_splitter.split_text(text)
    
    # Generate and store embeddings
    embeddings = embedder.embed(chunks)
    faiss_store.add_embeddings(embeddings)
    
    # Save embeddings for future use
    os.makedirs("models", exist_ok=True)
    faiss_store.save("models/embeddings.faiss")
    
    # Initialize retriever
    retriever = Retriever(embedder, faiss_store)
    
    # Query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Retrieve relevant chunks
        relevant_chunks = retriever.retrieve(query, chunks)
        
        # Generate answer
        answer = generator.generate(query, relevant_chunks)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()