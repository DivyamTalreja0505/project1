# Multimodal Retrieval-Augmented Generation (RAG) System Using GPT-4o and LangChain

# Project Description
This project implements a Multimodal Retrieval-Augmented Generation (RAG) system to extract, summarize, and retrieve information from complex PDF files. It uses GPT-3.5-turbo and GPT-4o to process text, tables, and images. The system stores embeddings in ChromaDB for semantic search and preserves original files in InMemoryStore for complete retrieval. LangChain is used to build an end-to-end question-answering pipeline. The project extracts text, table structures, and images from PDFs using Unstructured and Tesseract OCR. Summarization of text and tables is handled by GPT-3.5-turbo, while image understanding is powered by GPT-4o. It supports semantic search, scalable retrieval, and a flexible architecture suitable for enterprise-grade analytics. Technologies used include Python 3.10+, LangChain, OpenAI APIs (GPT-3.5-Turbo, GPT-4o), ChromaDB, Unstructured, Pytesseract, and dotenv.

# Installation Steps
# 1. Clone the Repository
git clone https://github.com/DivyamTalreja0505/project2

# 2. Navigate to Project Folder
cd llm-rag-project

# 3. Create a Virtual Environment
python -m venv env

# 4. Activate Virtual Environment
# For Linux/Mac
source env/bin/activate
# For Windows
env\Scripts\activate

# 5. Install Requirements
pip install -r requirements.txt

# Environment Setup
# Create a .env file in the root folder with the following keys:
OPENAI_API_KEY="your-openai-api-key-here"
GROQ_API_KEY="your-groq-api-key-here"
TAVILY_API_KEY="your-tavily-api-key-here"
LANGCHAIN_API_KEY="your-langchain-api-key-here"
LANGCHAIN_PROJECT="your-project-name-here"

# Tesseract OCR Installation
# Windows Download Link:
https://github.com/tesseract-ocr/tesseract
# macOS Installation
brew install tesseract
# Linux Installation
sudo apt install tesseract-ocr

# Running the Project
python multimodal_rag_system.py

# Make sure your PDF file is placed inside the project directory. Update the filename in the script if needed:
filename = os.path.join(input_path, "startupai-financial-report-v2.pdf")

# Example Questions Supported
# - What do you see in the images?
# - What is the name of the company?
# - What is the product displayed in the image?
# - How much are the total expenses of the company?
# - What is the ROI?
# - How much did the company sell in 2023 and 2022?

# Main Python Code Structure
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

# Saving Embeddings and Documents
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

# Building the Final RAG Chain
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

# Example Usage
chain.invoke("What do you see in the images?")
chain.invoke("What is the name of the company?")
chain.invoke("What is the product displayed in the image?")

# Features
 - Multimodal extraction and summarization from PDFs
 - Semantic search and retrieval powered by vector databases
 - Vision and text processing using GPT-4o and GPT-3.5-turbo
 - OCR-based extraction for scanned documents
 - Scalable and flexible architecture with LangChain

# Learning Outcomes
 - Building practical Multimodal Retrieval-Augmented Generation systems
 - Combining LLMs and vector search for enterprise applications
 - Structuring unstructured data for advanced analytics and automation

# License
 This project is licensed under the MIT License.

# Contributing
 Contributions are welcome. Fork the repository, create a branch, and submit a pull request.


