# Agentic-AI Financial Risk Assessment

# Financial Document RAG System with Ollama (Gemma 2B)

![RAG Pipeline](https://miro.medium.com/v2/resize:fit:1400/1*5ZLci3SuR0zM_QlZOADv8Q.png)  
*Retrieval-Augmented Generation for financial documents using local LLMs*

## 📌 Overview

A modular RAG (Retrieval-Augmented Generation) system that:
- Processes financial PDFs (annual reports, SEC filings, etc.)
- Uses **FAISS** for efficient document retrieval
- Generates answers using **Gemma 2B** via Ollama (running locally)
- Works entirely offline after setup

## 🛠️ Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| LLM Runtime         | Ollama                              |
| Language Model      | Gemma 2B                           |
| Embeddings          | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store        | FAISS (CPU/GPU)                    |
| Document Processing | PyPDF2 + LangChain                 |
| Python Framework    | Python 3.9+                        |

## 🚀 Installation

### Prerequisites
1. Install [Ollama](https://ollama.ai/)
2. Pull Gemma 2B model:
   ```bash
   ollama pull gemma:2b

Project Setup
Clone the repository:

bash
git clone https://github.com/pushkar3/financial-rag-ollama.git
cd financial-rag-ollama
Install dependencies:

bash
pip install -r requirements.txt
Place your financial documents in data/financial_docs/

🏃‍♂️ Usage
Basic Usage
bash
# Start Ollama server (in separate terminal)
ollama serve

# Run the RAG system
python src/main.py
Example Workflow
The system will:

Load and chunk your PDFs

Generate embeddings (saved to models/embeddings.faiss)

Launch interactive question-answering

Sample queries:

Enter your question (or 'quit' to exit): tell me about revenue of Uber?
2025-04-23 13:22:49,803 - retrieval - INFO - Retrieving top 5 chunks
2025-04-23 13:22:49,804 - embeddings - INFO - Generating embeddings
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.84it/s] 
2025-04-23 13:22:49,859 - generation - INFO - Generating response with Ollama
2025-04-23 13:23:59,078 - httpx - INFO - HTTP Request: POST http://127.0.0.1:11434/api/generate "HTTP/1.1 200 OK"

Answer: Sure, here is the answer to the question:

The revenue of Uber increased by 17% year-over-year to $37,281 million in 2023, compared to $31,877 million in 2022. This increase was primarily driven by the growth of the Mobility and Delivery business, which increased by 14% and 12,204%, respectively.        

🔧 Customization
Model Options
Try different Ollama models:

bash
ollama pull gemma:7b  # Larger model
Then modify generation.py:

python
generator = OllamaGenerator(model_name="gemma:7b")
Change embedding model in embeddings.py:

python
class Embedder:
    def __init__(self, model_name="all-mpnet-base-v2"):  # Larger model

🤝 Contribution
Contributions welcome! Please open an issue or PR for:

Additional document formats (Word, HTML)

UI improvements

Performance optimizations

💡 Pro Tip: Run ollama serve & to keep the server running in background
