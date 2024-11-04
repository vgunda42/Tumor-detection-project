# Import all necessary libraries
from src.logger import logging
from src.exception import CustomException
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from typing import List


# Initialize the model
model = OllamaLLM(model="llama3.2", extra_fields_behavior="allow")

# Define a custom prompt template
prompt_template = """
Context: {context}
Question: {question}
Answer: Let's analyze this step by step:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Load environment variables
load_dotenv()
jina_api_key = os.getenv("JINA_API_KEY")

def process_pdf(pdf_file) -> List[dict]:
    """Process PDF and return list of documents with metadata"""
    reader = PdfReader(pdf_file)
    documents = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            doc = {
                "page_content": text,
                "metadata": {"page": i + 1, "source": pdf_file.name}
            }
            documents.append(doc)
    
    return documents

def get_ans(query: str, docs: List[dict]):
    """Get answer using late chunking and contextual retrieval"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Create text chunks
        texts = []
        for doc in docs:
            chunks = text_splitter.split_text(doc["page_content"])
            for chunk in chunks:
                texts.append({
                    "page_content": chunk,
                    "metadata": doc["metadata"]
                })
        
        # Create embeddings
        embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v3"
        )
        
        # Create vector store
        docsearch = FAISS.from_texts(
            [doc["page_content"] for doc in texts],
            embeddings,
            metadatas=[doc["metadata"] for doc in texts]
        )
        
        # Retrieve relevant documents
        retrieved_docs = docsearch.similarity_search(query, k=4)
        
        # Initialize QA chain
        qa_chain = load_qa_chain(
            llm=model,
            chain_type="stuff",
            prompt=PROMPT,
            verbose=True
        )
        
        # Run the chain with proper input format
        response = qa_chain.run(input_documents=retrieved_docs, question=query)
        
        return response
        
    except Exception as e:
        logging.error(f"Error in get_ans: {str(e)}")
        raise CustomException(error_message="Error in processing question", error_detail=e)

try:
    logging.info("session started")
    st.header('Document chat bot')
    
    # Initialize session states
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    if 'documents' not in st.session_state:
        st.session_state['documents'] = None
    
    uploaded = st.file_uploader('choose a file')
    
    def clear_text_input():
        global input_text
        input_text = ""
    
    def get_text():
        global input_text
        input_text = st.text_input('Ask a question', key='input', on_change=clear_text_input)
        return input_text
    
    def clear_history():
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['documents']= []
    
    if uploaded:
        # Process PDF only once and store in session state
        if st.session_state['documents'] is None:
            st.session_state['documents'] = process_pdf(uploaded)
            logging.info("pdf data saved")
        
        user_input = get_text()
        if st.button('post'):
            output = get_ans(user_input, st.session_state['documents'])
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
    
    if st.button('Clear History'):
        clear_history()
    
    logging.info('message generated')
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=f"user_message_{i}")
        message(st.session_state['generated'][i], key=str(i))

except Exception as e:
    raise CustomException(error_message="An error occurred", error_detail=e)