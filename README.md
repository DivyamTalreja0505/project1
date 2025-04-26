```bash
Multimodal Retrieval-Augmented Generation (RAG) System Using GPT-4o and LangChain

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system that extracts text, tables, and images from PDFs, summarizes each element using different LLMs (GPT-3.5-turbo, GPT-4o), stores the embeddings into a ChromaDB vector database, and allows semantic question answering using LangChain.

Project Overview
- PDF Parsing: Extracts text, table structures, and images from complex PDFs using Unstructured and Tesseract OCR.
- Summarization: Text and Table summaries generated using GPT-3.5-Turbo, Image understanding using GPT-4o.
- Storage: Summaries embedded and stored in ChromaDB, Original documents stored in InMemoryStore.
- Retrieval-Augmented Question Answering (RAG): Queries answered using semantic retrieval based on stored summaries and context.

Technologies Used
- Python 3.10+
- LangChain
- OpenAI API (GPT-3.5-Turbo and GPT-4o)
- ChromaDB
- Unstructured
- Pytesseract
- dotenv

Installation
git clone https://github.com/yourusername/multimodal-rag-project.git
cd multimodal-rag-project
python -m venv env
source env/bin/activate  # For Windows use: env\Scripts\activate
pip install -r requirements.txt

Create a .env file:
OPENAI_API_KEY="your-openai-api-key-here"
GROQ_API_KEY="your-groq-api-key-here"
TAVILY_API_KEY="your-tavily-api-key-here"
LANGCHAIN_API_KEY="your-langchain-api-key-here"
LANGCHAIN_PROJECT="your-project-name-here"

Install Tesseract OCR manually:
For Windows: https://github.com/tesseract-ocr/tesseract
For macOS: brew install tesseract
For Linux: sudo apt install tesseract-ocr

How to Run
python multimodal_rag_system.py

Make sure your PDF is placed inside the project directory.
If needed edit filename:
filename=os.path.join(input_path, "startupai-financial-report-v2.pdf")

Example Questions
What do you see in the images?
What is the name of the company?
What is the product displayed in the image?
How much are the total expenses of the company?
What is the ROI?
How much did the company sell in 2023 and 2022?

Main Python Code Structure
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

Saving embeddings and documents
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

Building the final RAG chain
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

chain.invoke("What do you see in the images?")
chain.invoke("What is the name of the company?")
chain.invoke("What is the product displayed in the image?")

Features
- Full multimodal extraction from PDFs.
- Summarization using specialized LLMs.
- Semantic search with LangChain and ChromaDB.
- OCR support with Tesseract.
- End-to-end scalable architecture.

Learning Outcomes
- Building advanced Multimodal RAG pipelines.
- Integrating GPT-4o for vision and text understanding.
- Implementing semantic retrieval and search systems.
- Working with unstructured and structured data in AI.

License
This project is licensed under the MIT License.

Contributing
Contributions are welcome! Fork the repository, create a branch, and submit a pull request.

Connect
Email: divyamtalreja16@gmail.com
LinkedIn: https://linkedin.com/in/divyam-talreja/
```
