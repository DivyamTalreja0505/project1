```bash
# Multimodal Retrieval-Augmented Generation (RAG) System Using GPT-4o and LangChain

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system that extracts text, tables, and images from PDFs, summarizes each element using different LLMs (GPT-3.5-turbo, GPT-4o), stores the embeddings into a ChromaDB vector database, and allows semantic question answering using LangChain.

# 🚀 Project Overview
- PDF Parsing: Extracts text, table structures, and images from complex PDFs using Unstructured and Tesseract OCR.
- Summarization:
  - Text and Table summaries generated using GPT-3.5-Turbo.
  - Image understanding and description using GPT-4o.
- Storage:
  - Summaries embedded and stored in ChromaDB.
  - Original documents stored in InMemoryStore.
- Retrieval-Augmented Question Answering (RAG):
  - Queries answered using semantic retrieval based on stored summaries and context.

# 🛠️ Technologies Used
- Python 3.10+
- LangChain
- OpenAI API (GPT-3.5-Turbo and GPT-4o)
- ChromaDB (Vector Database)
- Unstructured (for PDF parsing)
- Pytesseract (OCR for image extraction)
- dotenv (Environment variable management)

# 📦 Installation
git clone https://github.com/yourusername/multimodal-rag-project.git
cd multimodal-rag-project
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

# Create a .env file inside your project directory with the following content:
# OPENAI_API_KEY="your-openai-api-key-here"
# GROQ_API_KEY="your-groq-api-key-here"
# TAVILY_API_KEY="your-tavily-api-key-here"
# LANGCHAIN_API_KEY="your-langchain-api-key-here"
# LANGCHAIN_PROJECT="your-project-name-here"

# Install Tesseract OCR manually:
# Windows: https://github.com/tesseract-ocr/tesseract
# macOS (Homebrew): brew install tesseract
# Linux: sudo apt install tesseract-ocr

# 📋 How to Run
python multimodal_rag_system.py

# Make sure your PDF is placed inside the project directory.
# Edit the filename if needed inside the script:
# filename = os.path.join(input_path, "startupai-financial-report-v2.pdf")

# Example Questions You Can Ask:
# What do you see in the images?
# What is the name of the company?
# What is the product displayed in the image?
# How much are the total expenses of the company?
# What is the ROI?
# How much did the company sell in 2023 and in 2022?

# 🧩 Main Python Code Structure (Simplified)
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from unstructured.partition.pdf import partition_pdf
import pytesseract

_ = load_dotenv(find_dotenv())

chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4o = ChatOpenAI(model="gpt-4o", max_tokens=1024)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "figures")

raw_pdf_elements = partition_pdf(
    filename=os.path.join(input_path, "startupai-financial-report-v2.pdf"),
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=output_path,
)

# Summarization Functions
from langchain.schema.messages import HumanMessage, AIMessage

def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

def summarize_table(table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ])
    ]
    response = chain_gpt_4o.invoke(prompt)
    return response.content

# Save embeddings and raw data into ChromaDB and InMemoryStore
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

vectorstorev2 = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
storev2 = InMemoryStore()
id_key = "doc_id"
retrieverv2 = MultiVectorRetriever(vectorstore=vectorstorev2, docstore=storev2, id_key=id_key)

def add_documents_to_retriever(summaries, original_contents):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
    retrieverv2.vectorstore.add_documents(summary_docs)
    retrieverv2.docstore.mset(list(zip(doc_ids, original_contents)))

# Then build the final RAG chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

prompt_template = """Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

chain = (
    {"context": retrieverv2, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Finally run example questions like:
# chain.invoke("What do you see in the images?")
# chain.invoke("What is the name of the company?")
# chain.invoke("What is the product displayed in the image?")

# ✨ Features
# - Full multimodal processing: text, tables, images.
# - Summarization and extraction using different LLMs.
# - Semantic vector-based search and retrieval.
# - OCR support for non-readable images.
# - Flexible and scalable architecture using ChromaDB and LangChain.

# 📚 Learning Outcomes
# - Building real-world Multimodal Retrieval-Augmented Generation systems.
# - Integrating GPT models with structured and unstructured data.
# - Setting up semantic search retrievers with LangChain.
# - Secure environment management with dotenv.

# 📜 License
# This project is licensed under the MIT License.

# 🤝 Contributing
# Contributions are welcome! Fork the repository, create a new branch, and submit a pull request.

# 🔗 Connect
# Email: divyamtalreja16@gmail.com
# LinkedIn: https://linkedin.com/in/divyam-talreja/
```
