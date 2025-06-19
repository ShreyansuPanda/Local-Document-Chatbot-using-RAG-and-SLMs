# 📄 Local Document Q&A Chatbot using RAG and Small Language Models (SLMs)

Welcome to the **Document Q&A Chatbot**!  
This project leverages **Retrieval-Augmented Generation (RAG)** architecture along with lightweight **Small Language Models (SLMs)** like *Phi-3* and *Mistral* to build an **offline, privacy-preserving document assistant**. It allows users to query documents (PDF, DOCX, CSV, XLSX) and get intelligent, context-aware answers—**without relying on external APIs**.

---

## 🎯 Problem Statement

Information buried in large documents is often hard to retrieve manually. This project solves that problem by combining **semantic search** with **local LLM inference**, enabling a smart assistant that can understand and answer user queries based on the document's content in real-time.

---

## 📂 Supported Document Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
  
---

## 🚀 Concepts Used

- **Retrieval-Augmented Generation (RAG):**  
  Augments language model generation with retrieved context from documents for more accurate answers.

- **Small Language Models (SLMs):**  
  Uses efficient, open-source models like Phi-3 and Mistral that can run locally without the need for cloud APIs.

- **LangChain & FAISS:**  
  Utilized for chunking, embedding, semantic vector search, and LLM chaining for question-answering tasks.

- **Streamlit UI:**  
  Interactive and user-friendly interface for document upload, model selection, querying, and chat export.

---

## 🔍 Code Functionality

### 1. **Document Parsing**
- Parses multiple file types:
  - PDFs via `PyMuPDF`
  - Word files via `python-docx`
- Text is cleaned and segmented into logical chunks for semantic retrieval

### 2. **Embedding and Vector Search**
- Embeds document chunks using `HuggingFaceEmbeddings`
- Stores embeddings in a **FAISS** index for fast and accurate retrieval

### 3. **Question Answering (RAG)**
- Retrieves top relevant chunks using semantic similarity
- Generates an answer using a selected local SLM (Phi-3/Mistral) based on the retrieved content

### 4. **Interactive Chat UI**
- Streamlit interface includes:
  - Document upload
  - Model selection
  - Query input
  - Real-time answer generation
  - Chat history with export option

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.10+
- Basic GPU or CPU (no cloud dependencies!)
- Ollama installed and running locally
   - Download from: https://ollama.ai/
   - Install and run the Ollama service
   - Pull all of the supported models:
     ```bash
     ollama pull phi3
     # and
     ollama pull mistral
     # and
     ollama pull llama2
     ```
- Required packages:

```bash
pip install -r requirements.txt
```
---

### 📌 Steps to Run
1. Clone the repository
```bash
git clone https://github.com/ShreyansuPanda/Local-Document-Chatbot-using-RAG-and-SLMs.git
```
2. Navigate to the project directory
```bash
cd "PDF Chatbot - V3"
```
3. Open CMD and run the script
```bash
streamlit run app.py
```
---

## 🗂️ File Structure
```sh
PDF Chatbot - V3/
│
├── app.py                             # Streamlit interface for chat
├── pdf_qa.py                          # Core logic for document parsing, retrieval, and generation
└── requirements.txt                   # All required dependencies
```
--- 
## 🌟 Show Your Support
If you like this project, consider giving it a ⭐ on GitHub. Contributions are also welcome!
