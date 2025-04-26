# Multimodal Retrieval-Augmented Generation (RAG) System with LLM Testing

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system using OpenAI's GPT-4o and GPT-3.5-turbo models. It supports document extraction from PDFs (text, tables, images), LLM testing, multimodal summarization, and structured database management using ChromaDB and LangChain.

Developed a Multimodal RAG system that processes text, tables, and images from PDFs. Summarized documents using GPT-3.5-Turbo for text and table summarization and GPT-4o for advanced image understanding. Conducted evaluation of different LLM models on summarization tasks. Built a scalable vector database (ChromaDB) to store embeddings and managed original documents separately using an InMemoryStore for efficient retrieval. Enabled semantic search and retrieval using LangChain’s MultiVectorRetriever.

**Technologies Used**: Python 3.10+, LangChain, OpenAI API (gpt-3.5-turbo, gpt-4o), ChromaDB, Unstructured (PDF parsing), Pytesseract (OCR).

**Installation Steps**:
- Clone this repository:  
`git clone https://github.com/yourusername/multimodal-rag-llm-testing.git`
- Create a virtual environment and activate it:  
`python -m venv env`  
`source env/bin/activate` (for Windows: `env\Scripts\activate`)
- Install dependencies:  
`pip install -r requirements.txt`
- Create a `.env` file and add your OpenAI API key:  
`OPENAI_API_KEY=your_openai_api_key_here`

**How to Run**:  
Run `python multimodal_rag_testing.py` to:
- Extract text, tables, and images from a sample PDF.
- Summarize each content type using LLMs.
- Store embeddings and documents.
- Perform semantic retrieval and question answering.

**Learning Outcomes**:  
- Build a Multimodal RAG system from scratch.
- Summarize text, tables, and images separately using different LLMs.
- Compare outputs between GPT-3.5 and GPT-4o.
- Create an intelligent database that supports fast retrieval.

**Sample Questions You Can Ask After Running the System**:
- What is the ROI mentioned in the document?
- What products are displayed in the images?
- How much did the company sell in 2023?
- Summarize the financial statement of the company.

**Key Features**:  
- Multimodal processing (Text + Table + Image).
- LLM evaluation and comparison.
- Fast vector retrieval using ChromaDB.
- Storage of raw documents and summaries separately.

**License**:  
This project is licensed under the MIT License.

**Connect**:  
If you find this project helpful, feel free to connect or contribute!
