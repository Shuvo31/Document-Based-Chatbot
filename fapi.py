import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import math
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Form, BackgroundTasks
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Environment and evaluation imports
from dotenv import load_dotenv

# Evaluation imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Document-Based Chatbot API",
    description="A FastAPI-based document chatbot with Azure OpenAI and evaluation metrics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectorstore = None
document_processor = None
chat_memory = None

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = []
    evaluation: Optional[Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    enable_evaluation: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]] = {}
    processing_time: float

class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    processed_at: str

class ProcessedDocumentsResponse(BaseModel):
    documents: Dict[str, DocumentInfo]
    total_chunks: int
    status: str

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

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate"""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        if not reference_tokens or not candidate_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        return round(score, 4)
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

def calculate_rouge_score(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    if not NLTK_AVAILABLE:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference.lower(), candidate.lower())
        return {
            "rouge-1": round(scores['rouge1'].fmeasure, 4),
            "rouge-2": round(scores['rouge2'].fmeasure, 4),
            "rouge-l": round(scores['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

def calculate_perplexity(text: str) -> float:
    """Calculate a simple perplexity approximation"""
    try:
        tokens = text.split()
        if len(tokens) == 0:
            return float('inf')
        
        # Simplified perplexity calculation
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        if unique_tokens == 0:
            return 1000.0
        
        # Simple approximation based on vocabulary diversity
        perplexity = total_tokens / unique_tokens
        return round(min(perplexity, 1000.0), 2)
    except:
        return 0.0

async def load_uploaded_files(files: List[UploadFile]) -> List[Document]:
    """Load and process multiple documents"""
    global document_processor
    all_documents = []
    
    for uploaded_file in files:
        file_extension = Path(uploaded_file.filename).suffix.lower()
        if file_extension in SUPPORTED_EXTENSIONS:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await uploaded_file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                # Load document
                loader = SUPPORTED_EXTENSIONS[file_extension](temp_file_path)
                documents = []
                
                if file_extension == ".pdf":
                    try:
                        documents = loader.load()
                    except Exception as e:
                        print(f"Error loading PDF file {uploaded_file.filename}: {e}")
                        continue
                else:
                    documents = loader.load()
                
                # Add document metadata
                doc_id = f"doc_{len(document_processor.document_metadata) + 1}"
                for doc in documents:
                    doc.metadata['source'] = uploaded_file.filename
                    doc.metadata['doc_id'] = doc_id
                    doc.metadata['file_type'] = file_extension
                
                document_processor.add_document_metadata(
                    doc_id, uploaded_file.filename, file_extension, len(documents)
                )
                
                all_documents.extend(documents)
                os.unlink(temp_file_path)
                
            except Exception as e:
                print(f"Error processing file {uploaded_file.filename}: {e}")
    
    return all_documents

def validate_azure_credentials():
    """Validate Azure OpenAI credentials"""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_EMBEDDING_DEPLOYMENT_NAME",
        "AZURE_EMBEDDING_API_VERSION"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise HTTPException(
            status_code=500,
            detail=f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}"
        )

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Document-Based Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-documents/",
            "chat": "/chat/",
            "documents": "/documents/",
            "health": "/health/"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    try:
        validate_azure_credentials()
        return {
            "status": "healthy",
            "nltk_available": NLTK_AVAILABLE,
            "vectorstore_initialized": vectorstore is not None,
            "document_processor_initialized": document_processor is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/upload-documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    global vectorstore, document_processor
    
    try:
        validate_azure_credentials()
        
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > 10:  # Limit number of files
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Load documents
        documents = await load_uploaded_files(files)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents were processed")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return ProcessedDocumentsResponse(
            documents={
                doc_id: DocumentInfo(**metadata) 
                for doc_id, metadata in document_processor.document_metadata.items()
            },
            total_chunks=len(texts),
            status=f"Successfully processed {len(texts)} chunks from {len(files)} documents"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with uploaded documents"""
    global vectorstore, document_processor, chat_memory
    
    start_time = datetime.now()
    
    try:
        validate_azure_credentials()
        
        if not vectorstore:
            raise HTTPException(status_code=400, detail="No documents uploaded. Please upload documents first.")
        
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Initialize Azure OpenAI Chat model
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.1
        )
        
        # Initialize memory if not exists
        if not chat_memory:
            chat_memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=chat_memory,
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain.invoke({"question": request.question})
        response = result['answer']
        source_documents = result.get('source_documents', [])
        
        # Get source information
        source_info = document_processor.get_source_info(source_documents) if document_processor else []
        
        # Calculate evaluation metrics
        evaluation = {}
        if request.enable_evaluation:
            # Calculate perplexity
            evaluation['perplexity'] = calculate_perplexity(response)
            
            # Calculate BLEU and ROUGE if ground truth provided
            if request.ground_truth and request.ground_truth.strip():
                evaluation['bleu_score'] = calculate_bleu_score(request.ground_truth, response)
                evaluation['rouge_scores'] = calculate_rouge_score(request.ground_truth, response)
            
            # Add context information
            evaluation['context_chunks_used'] = len(source_documents)
            evaluation['response_length'] = len(response.split())
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            answer=response,
            sources=source_info,
            evaluation=evaluation if request.enable_evaluation else {},
            processing_time=round(processing_time, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/documents/", response_model=ProcessedDocumentsResponse)
async def get_processed_documents():
    """Get information about processed documents"""
    global document_processor, vectorstore
    
    if not document_processor:
        return ProcessedDocumentsResponse(
            documents={},
            total_chunks=0,
            status="No documents processed"
        )
    
    total_chunks = 0
    if vectorstore and hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
        total_chunks = vectorstore.index.ntotal
    
    return ProcessedDocumentsResponse(
        documents={
            doc_id: DocumentInfo(**metadata) 
            for doc_id, metadata in document_processor.document_metadata.items()
        },
        total_chunks=total_chunks,
        status="Documents processed successfully"
    )

@app.delete("/documents/")
async def clear_documents():
    """Clear all processed documents and reset the system"""
    global vectorstore, document_processor, chat_memory
    
    vectorstore = None
    document_processor = None
    chat_memory = None
    
    return {"message": "All documents cleared successfully"}

@app.post("/evaluate-text/")
async def evaluate_text(
    reference_text: str = Form(...),
    candidate_text: str = Form(...),
    enable_bleu: bool = Form(True),
    enable_rouge: bool = Form(True)
):
    """Evaluate text using BLEU and ROUGE scores"""
    try:
        evaluation = {}
        
        if enable_bleu:
            evaluation['bleu_score'] = calculate_bleu_score(reference_text, candidate_text)
        
        if enable_rouge:
            evaluation['rouge_scores'] = calculate_rouge_score(reference_text, candidate_text)
        
        evaluation['perplexity'] = calculate_perplexity(candidate_text)
        evaluation['reference_length'] = len(reference_text.split())
        evaluation['candidate_length'] = len(candidate_text.split())
        
        return JSONResponse(content={
            "evaluation": evaluation,
            "nltk_available": NLTK_AVAILABLE
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating text: {str(e)}")

if __name__ == "__main__":
    # Initialize global variables
    document_processor = None
    vectorstore = None
    chat_memory = None
    
    # Run the FastAPI app
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )