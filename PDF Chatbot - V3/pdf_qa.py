import fitz  # PyMuPDF
from typing import List, Optional, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import logging
import hashlib
from pathlib import Path
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "phi3": {
        "model_name": "phi3",
        "temperature": 0.7,
        "context_window": 2048,
        "prompt_template": """You are an intelligent assistant trained to answer questions based only on the provided context. The context is extracted from a PDF uploaded by the user.

Your job is to:
- Read the context carefully.
- Answer the user's question using only the information present in the context.
- If the answer is not found in the context, respond with "The answer is not available in the provided PDF."
- Do not make up any information, and do not use external knowledge or assumptions.
- If the question is not related to the context, respond with "The question is not related to the provided PDF."
- Do not use any other information than the context provided.
==========
Context:
{context}
==========

Question: {question}

Answer:"""
    },
    "mistral": {
        "model_name": "mistral",
        "temperature": 0.7,
        "context_window": 4096,
        "prompt_template": """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a clear and concise response.

        Context: {context}

        Question: {question}
        Answer:"""
    },
    "llama2": {
        "model_name": "llama2",
        "temperature": 0.7,
        "context_window": 4096,
        "prompt_template": """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a detailed and well-structured response.

        Context: {context}

        Question: {question}
        Answer:"""
    }
}

class PDFQASystem:
    def __init__(self, model_name: str = "phi3", chunk_size: int = 700, chunk_overlap: int = 150):
        """Initialize the PDF QA system with specified model and chunking parameters."""
        try:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CONFIGS.keys())}")
            
            logger.info(f"Initializing PDFQASystem with model: {model_name}")
            self.model_config = MODEL_CONFIGS[model_name]
            self.model_name = model_name
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
            # Initialize embeddings
            logger.info("Initializing embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize LLM with model-specific configuration
            logger.info(f"Initializing LLM with model: {self.model_config['model_name']}")
            self.llm = Ollama(
                model=self.model_config["model_name"],
                temperature=self.model_config["temperature"]
            )
            
            self.vector_store = None
            self.qa_chain = None
            self.processed_files: Dict[str, str] = {}  # filename -> hash mapping
            
            # Create index directory if it doesn't exist
            self.index_dir = Path("faiss_index")
            self.index_dir.mkdir(exist_ok=True)
            
            logger.info("PDFQASystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PDFQASystem: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of file based on name and size."""
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        return hashlib.md5(f"{file_name}:{file_size}".encode()).hexdigest()

    def load_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text from PDF file and return text and page count."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Loading PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))  # sort top to bottom, left to right
                text += "\n".join([b[4] for b in blocks if b[4].strip()]) + "\n"
            
            if not text.strip():
                logger.warning(f"PDF appears to be empty or contains no text: {pdf_path}")
                raise ValueError("PDF appears to be empty or contains no text")
            
            logger.info(f"Successfully loaded PDF with {len(doc)} pages")
            return text, len(doc)
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        try:
            logger.info(f"Creating chunks with size={self.chunk_size}, overlap={self.chunk_overlap}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_vector_store(self, chunks: List[str], is_new_index: bool = True):
        """Create or update FAISS vector store from text chunks."""
        try:
            if is_new_index:
                logger.info("Creating new vector store")
                self.vector_store = FAISS.from_texts(
                    chunks, 
                    self.embeddings,
                    normalize_L2=True
                )
            else:
                logger.info("Updating existing vector store")
                self.vector_store.add_texts(chunks)
                
            logger.info("Vector store created/updated successfully")
            
        except Exception as e:
            logger.error(f"Error creating/updating vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt."""
        try:
            logger.info("Setting up QA chain")
            PROMPT = PromptTemplate(
                template=self.model_config["prompt_template"],
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 5,
                        #"score_threshold": 0.5
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "document_variable_name": "context"
                }
            )
            logger.info("QA chain set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def process_pdf(self, pdf_path: str, progress_callback=None) -> bool:
        """Process PDF file and update the QA system. Returns True if successful."""
        try:
            file_hash = self._compute_file_hash(pdf_path)
            
            # Check if we already have this file indexed
            if file_hash in self.processed_files.values():
                logger.info(f"File already indexed: {pdf_path}")
                return True

            # Load PDF and get page count
            text, page_count = self.load_pdf(pdf_path)
            if progress_callback:
                progress_callback(0.2, f"Loaded {page_count} pages")

            # Create chunks
            chunks = self.create_chunks(text)
            if progress_callback:
                progress_callback(0.4, f"Created {len(chunks)} chunks")
            
            # Create new index if none exists, otherwise update
            is_new_index = self.vector_store is None
            self.create_vector_store(chunks, is_new_index)
            if progress_callback:
                progress_callback(0.8, "Created/updated vector store")
            
            if is_new_index:
                self.setup_qa_chain()
                if progress_callback:
                    progress_callback(0.9, "Set up QA chain")
            
            # Save the vector store
            self.save_vector_store()
            if progress_callback:
                progress_callback(1.0, "Saved vector store")
            
            # Update processed files tracking
            self.processed_files[os.path.basename(pdf_path)] = file_hash
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def save_vector_store(self):
        """Save the vector store to disk."""
        try:
            if self.vector_store:
                logger.info("Saving vector store to disk")
                self.vector_store.save_local(str(self.index_dir))
                logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def load_vector_store(self) -> bool:
        """Load the vector store from disk if it exists."""
        try:
            index_path = self.index_dir / "index.faiss"
            if index_path.exists():
                logger.info("Loading vector store from disk")
                self.vector_store = FAISS.load_local(str(self.index_dir), self.embeddings)
                self.setup_qa_chain()
                logger.info("Vector store loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def answer_question(self, question: str) -> dict:
        """Get answer for a question."""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized. Please process a PDF first.")
            
            logger.info(f"Answering question: {question}")
            result = self.qa_chain({"query": question})
            logger.info("Successfully generated answer")
            
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            logger.error(traceback.format_exc())
            raise 