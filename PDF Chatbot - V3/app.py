import streamlit as st
import tempfile
import os
from pdf_qa import DocumentQASystem, MODEL_CONFIGS
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config and styling
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS to make the uploader more visible and style chat messages
st.markdown("""
    <style>
    .stFileUploader {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        color: black;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: column;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: black;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: black;
    }
    .status-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Document Q&A Chatbot")
st.write("Upload one or more documents (PDF, DOCX) and ask questions about their content!")

# Initialize session state
if 'qa_system' not in st.session_state:
    try:
        st.session_state.qa_system = DocumentQASystem()
        logger.info("Initialized QA system")
    except Exception as e:
        st.error(f"Error initializing QA system: {str(e)}")
        logger.error(f"Error initializing QA system: {str(e)}")
        st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection with descriptions
    st.subheader("Model Selection")
    model = st.selectbox(
        "Select Model",
        list(MODEL_CONFIGS.keys()),
        index=list(MODEL_CONFIGS.keys()).index("mistral"),
        help="Choose the language model to use for answering questions"
    )
    
    # Display model information
    st.markdown("### Model Information")
    
    # Get model style description
    prompt = MODEL_CONFIGS[model]['prompt_template']
    style_desc = "Standard response"
    if "Provide a" in prompt:
        style_desc = prompt.split("Provide a")[1].split("response")[0].strip()
    elif "specialized in" in prompt:
        style_desc = "Technical and code-focused"
    elif "natural and conversational" in prompt:
        style_desc = "Conversational and natural"
    
    st.markdown(f"""
        <div class='model-info'>
            <b>Model:</b> {MODEL_CONFIGS[model]['model_name']}<br>
            <b>Context Window:</b> {MODEL_CONFIGS[model]['context_window']} tokens<br>
            <b>Temperature:</b> {MODEL_CONFIGS[model]['temperature']}<br>
            <b>Style:</b> {style_desc}
        </div>
    """, unsafe_allow_html=True)
    
    # Chunking parameters
    st.subheader("Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Size of text chunks for processing"
    )
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=200,
        step=10,
        help="Overlap between chunks"
    )
    
    # Update QA system if parameters changed
    if (st.session_state.qa_system.model_name != model or
        st.session_state.qa_system.chunk_size != chunk_size or
        st.session_state.qa_system.chunk_overlap != chunk_overlap):
        try:
            st.session_state.qa_system = DocumentQASystem(
                model_name=model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.success(f"Successfully switched to {model} model")
            logger.info(f"Switched to {model} model")
        except Exception as e:
            st.error(f"Error switching model: {str(e)}")
            logger.error(f"Error switching model: {str(e)}")

# File uploader with multiple file support
st.markdown("### 📄 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Drag and drop your files here or click to browse",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Supported formats: PDF, DOCX files"
)

if uploaded_files:
    # Create columns for progress bars
    cols = st.columns(len(uploaded_files))
    
    # Process each uploaded file
    for idx, uploaded_file in enumerate(uploaded_files):
        # Get the correct file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Initialize progress bar
            progress_bar = cols[idx].progress(0)
            status_text = cols[idx].empty()
            
            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            # Process the file
            success = st.session_state.qa_system.process_file(
                tmp_file_path,
                progress_callback=update_progress
            )
            
            if success:
                cols[idx].success(f"✅ {uploaded_file.name}")
                logger.info(f"Successfully processed {uploaded_file.name}")
            else:
                cols[idx].warning(f"⚠️ {uploaded_file.name}")
                logger.warning(f"Failed to process {uploaded_file.name}")
        
        except Exception as e:
            cols[idx].error(f"❌ {uploaded_file.name}: {str(e)}")
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    # Question input
    st.markdown("### ❓ Ask a Question")
    question = st.text_input("Type your question about the document content here:")
    
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Check if vector store is initialized
                if not st.session_state.qa_system.vector_store:
                    st.error("No documents have been processed yet. Please upload and process documents first.")
                    logger.error("Attempted to answer question without processed documents")
                    st.stop()

                result = st.session_state.qa_system.answer_question(question)
                logger.info(f"Generated answer for question: {question}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model
                })
                
                # Display chat history
                st.markdown("### 💬 Chat History")
                for chat in st.session_state.chat_history:
                    with st.expander(f"Q: {chat['question']} ({chat['timestamp']}) - Model: {chat['model']}"):
                        st.markdown("**Answer:**")
                        st.write(chat["answer"])
                        
                        st.markdown("**Sources:**")
                        for i, source in enumerate(chat["sources"], 1):
                            st.markdown(f"<div class='source-box'><b>Source {i}:</b><br>{source}</div>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                logger.error(f"Error generating answer: {str(e)}")

    # Download chat history
    if st.session_state.chat_history:
        chat_text = ""
        for chat in st.session_state.chat_history:
            chat_text += f"Q: {chat['question']} ({chat['timestamp']}) - Model: {chat['model']}\n"
            chat_text += f"A: {chat['answer']}\n"
            chat_text += "Sources:\n"
            for i, source in enumerate(chat["sources"], 1):
                chat_text += f"Source {i}:\n{source}\n"
            chat_text += "\n" + "="*50 + "\n\n"
        
        st.download_button(
            "💾 Download Q&A Log",
            chat_text,
            file_name="qa_log.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Please upload one or more documents to begin.")