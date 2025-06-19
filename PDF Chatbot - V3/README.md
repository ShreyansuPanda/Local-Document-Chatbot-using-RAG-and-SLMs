# Document Q&A Chatbot

A powerful chatbot that can answer questions about the content of your documents (PDF, DOCX, CSV, and XLSX) using LangChain and local LLMs via Ollama.

## Features

- Support for multiple document formats:
  - PDF files
  - DOCX (Microsoft Word) files
  - CSV files
  - XLSX (Microsoft Excel) files
- Chat history with expandable Q&A pairs
- Download Q&A transcript
- Configurable chunking parameters
- Persistent vector store
- Support for multiple LLM models (phi3, mistral, llama2)
- Beautiful Streamlit UI

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
   - Download from: https://ollama.ai/
   - Install and run the Ollama service
   - Pull at least one of the supported models:
     ```bash
     ollama pull phi3
     # or
     ollama pull mistral
     # or
     ollama pull llama2
     ```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd document-chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure Ollama is running in the background

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Upload one or more documents (PDF, DOCX, CSV, or XLSX) using the file uploader
2. Wait for the files to be processed
3. Type your question in the text input
4. View the answer and source context in the chat history
5. Download the Q&A transcript using the download button

## Configuration

You can adjust the following settings in the sidebar:
- Select different LLM models
- Adjust chunk size and overlap for text processing
- View and manage chat history

## Troubleshooting

- If you get an error about Ollama not being available, make sure the Ollama service is running
- If document processing fails, check that the file is not corrupted or password-protected
- For memory issues with large documents, try reducing the chunk size in the configuration
- For CSV and XLSX files, ensure they are properly formatted and contain text data

## Error Handling

The system includes error handling for:
- Invalid document files
- Empty documents
- Processing errors
- Question answering errors

## License

MIT License 