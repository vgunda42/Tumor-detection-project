# PDF Chatbot for Medical field with LangChain and Streamlit

A sophisticated document question-answering system that enables users to upload PDF documents and interact with their content through natural language queries. The application leverages LangChain, Ollama, and FAISS for efficient document processing and retrieval.

## 🚀 Features

PDF document upload and processing
Interactive chat interface using Streamlit
Semantic search using FAISS vector store
Context-aware question answering using LLaMA model
Late chunking strategy for optimal document processing
Session state management for conversation history
Custom exception handling and logging

## 🛠️ Technical Architecture

Frontend: Streamlit web interface
Document Processing: PyPDF2 for PDF text extraction
Embeddings: Jina AI embeddings (v3)
Vector Store: FAISS for similarity search
LLM Integration: Ollama with LLaMA 3.2
Framework: LangChain for orchestration

## 📋 Prerequisites

Python 3.8+
Ollama installed and running locally
Jina AI API key

## 🔧 Installation

Clone the repository:

git clone `repository-url`
cd `Bharatkumar_kori_PdfBotforMedical`

Create and activate a virtual environment:

Linux:
Install minicaond
`conda create -n <environment name>`
`conda activate environment name`

Install required packages:
sudo apt install pip
pip install -r requirements.txt

Set up environment variables:
Create .env file #for storing senstive/secret keys#

## Add your Jina AI API key

echo "JINA_API_KEY=your_api_key_here" >> .env
📦 Dependencies
plaintextCopystreamlit
streamlit-chat
langchain
PyPDF2
python-dotenv
faiss-cpu
jina
ollama

## 🚀 Usage

Start the application:

bashCopystreamlit run app.py

Open your browser and navigate to `http://localhost:8501`
Upload a PDF document using the file uploader
Start asking questions about the document content

💡 Key Components
PDF Processing

Utilizes PyPDF2 for text extraction
Maintains document metadata including page numbers and source
Implements recursive text splitting for optimal chunk size

Vector Search

Employs FAISS for efficient similarity search
Uses Jina AI embeddings for semantic understanding
Implements late chunking strategy for better context preservation

Question Answering

Custom prompt template for structured responses
Uses LangChain's QA chain with "stuff" method
Retrieves top 4 most relevant document chunks

## 📝 Logging

The application implements comprehensive logging:

Session start/end
Message generation
Error tracking
PDF processing status

## ⚠️ Error Handling

Custom exception handling is implemented through:

CustomException class for structured error reporting
Detailed error messages with stack traces
Graceful degradation on failure

## 🔜 Future Improvements

 Add support for multiple document formats
 Enhance prompt engineering for better responses
 Implement caching for faster responses
