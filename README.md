# Multimodal Retrieval-Augmented Generation (RAG) System Using GPT-4o and LangChain

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system that extracts text, tables, and images from PDFs, summarizes each element using different LLMs (GPT-3.5-turbo, GPT-4o), stores the embeddings into a ChromaDB vector database, and allows semantic question answering using LangChain.

## 🚀 Project Overview
- PDF Parsing: Extracts text, table structures, and images from complex PDFs using Unstructured and Tesseract OCR.
- Summarization:
  - Text and Table summaries generated using GPT-3.5-Turbo.
  - Image understanding and description using GPT-4o.
- Storage:
  - Summaries embedded and stored in ChromaDB.
  - Original documents stored in InMemoryStore.
- Retrieval-Augmented Question Answering (RAG):
  - Queries answered using semantic retrieval based on stored summaries and context.

## 🛠️ Technologies Used
- Python 3.10+
- LangChain
- OpenAI API (GPT-3.5-Turbo and GPT-4o)
- ChromaDB (Vector Database)
- Unstructured (for PDF parsing)
- Pytesseract (OCR for image extraction)
- dotenv (Environment variable management)

## 📦 Installation
```bash
git clone https://github.com/yourusername/multimodal-rag-project.git
cd multimodal-rag-project
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
