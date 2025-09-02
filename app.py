import os
import base64
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import streamlit as st
from dotenv import load_dotenv

# Evaluation imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy, answer_similarity, answer_correctness,
        context_precision, context_recall, faithfulness
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

import math

# Load environment variables from .env file
load_dotenv()

# Load Azure OpenAI credentials from .env file
required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY", 
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_EMBEDDING_DEPLOYMENT_NAME",
    "AZURE_EMBEDDING_API_VERSION"
]

missing_vars = [var for var in required_env_vars if var not in os.environ]
if missing_vars:
    st.error(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}. Please add them to your .env file.")
    st.stop()

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}

class DocumentProcessor:
    def __init__(self):
        self.document_metadata = {}
        self.processed_documents = []
        
    def add_document_metadata(self, doc_id: str, filename: str, file_type: str, chunk_count: int):
        """Store metadata for each processed document"""
        self.document_metadata[doc_id] = {
            "filename": filename,
            "file_type": file_type,
            "chunk_count": chunk_count,
            "processed_at": datetime.now().isoformat()
        }
    
    def get_source_info(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        source_info = []
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
            
            source_info.append({
                "index": i + 1,
                "filename": filename,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        return source_info

def calculate_perplexity(text: str, llm) -> float:
    """Calculate perplexity score for generated text"""
    try:
        # Simple perplexity approximation using token probabilities
        tokens = text.split()
        if len(tokens) == 0:
            return float('inf')
        
        # For simplicity, we'll use a basic perplexity calculation
        # In practice, you'd need access to token probabilities from the model
        log_prob_sum = 0
        for token in tokens:
            # This is a simplified approximation
            prob = 1.0 / len(tokens)  # Uniform probability assumption
            log_prob_sum += math.log(prob) if prob > 0 else -10
        
        perplexity = math.exp(-log_prob_sum / len(tokens))
        return min(perplexity, 1000)  # Cap at 1000 for display
    except:
        return 0.0

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate"""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        
        if not reference_tokens or not candidate_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

def calculate_rouge_score(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    if not NLTK_AVAILABLE:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge-1": scores['rouge1'].fmeasure,
            "rouge-2": scores['rouge2'].fmeasure,
            "rouge-l": scores['rougeL'].fmeasure
        }
    except:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

def evaluate_with_ragas(question: str, answer: str, contexts: List[str], ground_truth: str = None):
    """Evaluate using RAGAS metrics"""
    if not RAGAS_AVAILABLE or not ground_truth:
        return {}
    
    try:
        # Prepare data for RAGAS
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        # Define metrics
        metrics = [answer_relevancy, faithfulness, context_precision]
        if ground_truth:
            metrics.extend([answer_correctness, context_recall])
        
        # Evaluate
        result = evaluate(data, metrics=metrics)
        return result
    except Exception as e:
        st.warning(f"RAGAS evaluation failed: {e}")
        return {}

def display_pdf(uploaded_file):
    """Display an uploaded PDF file in the Streamlit sidebar."""
    base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="400px"
        type="application/pdf"
    ></iframe>
    """
    st.sidebar.markdown("### PDF Preview")
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

def load_uploaded_files(uploaded_files, doc_processor):
    """Load and process multiple documents with metadata tracking"""
    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in SUPPORTED_EXTENSIONS:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                loader = SUPPORTED_EXTENSIONS[file_extension](temp_file_path)
                documents = []
                
                if file_extension == ".pdf":
                    try:
                        documents = loader.load()
                    except Exception as e:
                        st.warning(f"Error loading PDF file {uploaded_file.name}: {e}")
                        continue
                else:
                    documents = loader.load()
                
                # Add document source information to metadata
                doc_id = f"doc_{len(doc_processor.document_metadata) + 1}"
                for doc in documents:
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['doc_id'] = doc_id
                    doc.metadata['file_type'] = file_extension
                
                doc_processor.add_document_metadata(
                    doc_id, uploaded_file.name, file_extension, len(documents)
                )
                
                all_documents.extend(documents)
                os.unlink(temp_file_path)
                
                st.sidebar.success(f"✅ Loaded {uploaded_file.name} ({len(documents)} chunks)")
                
            except Exception as e:
                st.warning(f"Error loading file {uploaded_file.name}: {e}")
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
    
    return all_documents

# Initialize document processor
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

st.set_page_config(page_title="Advanced Document-Based Chatbot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("📄 Document Upload & Management")
    
    uploaded_files = st.file_uploader(
        "Upload multiple documents (PDF, DOCX, CSV, XLSX)",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True,
        help="You can upload multiple documents and ask questions from any of them"
    )

    # Document processing section
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "evaluation_history" not in st.session_state:
        st.session_state.evaluation_history = []
    
    # Process documents button
    if uploaded_files and st.button("🔄 Process Documents"):
        st.session_state.documents_processed = False  # Reset processing state
        st.session_state.doc_processor = DocumentProcessor()  # Reset processor
    
    if uploaded_files and not st.session_state.documents_processed:
        with st.spinner("Processing uploaded documents..."):
            documents = load_uploaded_files(uploaded_files, st.session_state.doc_processor)
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Initialize Azure OpenAI Embeddings
                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                    openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY")
                )
                
                vectorstore = FAISS.from_documents(texts, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.documents_processed = True
                st.success(f"🎉 Successfully processed {len(texts)} chunks from {len(uploaded_files)} documents!")
            else:
                st.error("No valid documents were processed. Please upload supported files.")
    
    # Display document metadata
    if st.session_state.documents_processed and st.session_state.doc_processor.document_metadata:
        st.subheader("📋 Processed Documents")
        for doc_id, metadata in st.session_state.doc_processor.document_metadata.items():
            with st.expander(f"📄 {metadata['filename']}"):
                st.write(f"**Type:** {metadata['file_type']}")
                st.write(f"**Chunks:** {metadata['chunk_count']}")
                st.write(f"**Processed:** {metadata['processed_at'][:19]}")
    
    # Display PDFs in sidebar
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".pdf":
                display_pdf(uploaded_file)

# Main interface
st.title("🤖 Advanced Document-Based Chatbot")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'**👤 You:** {message["content"]}')
    else:
        st.markdown(f'**🤖 ChatBot:** {message["content"]}')
        
        # Display source information if available
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"""
                    **Source {source['index']}:** {source['filename']}
                    
                    *Preview:* {source['content_preview']}
                    """)

# User input section
st.subheader("💬 Ask Questions")
user_query = st.text_area(
    "Ask a question about your documents:",
    key="user_input",
    placeholder="e.g., What are the main topics discussed in the documents?"
)

if st.button("📤 Send", type="primary"):
    if user_query and st.session_state.documents_processed:
        with st.spinner("🔍 Processing your query..."):
            # Initialize Azure OpenAI Chat model
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=0.1
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"  # Explicitly set the output key
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=memory,
                return_source_documents=True
            )
            
            try:
                result = qa_chain.invoke({"question": user_query})
                response = result['answer']
                source_documents = result.get('source_documents', [])
                
                # Get source information
                source_info = st.session_state.doc_processor.get_source_info(source_documents)
                
                # Prepare response message
                response_data = {
                    "role": "assistant",
                    "content": response,
                    "sources": source_info
                }
                
            except Exception as e:
                response_data = {
                    "role": "assistant", 
                    "content": f"Error querying documents: {e}",
                    "sources": []
                }
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append(response_data)
            st.rerun()
    
    elif not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first!")
    elif not user_query:
        st.warning("⚠️ Please enter a question!")
