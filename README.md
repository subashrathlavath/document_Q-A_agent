# Document Q&A Agent

This project is a command-line-based Question & Answering agent that uses a large language model to answer questions about a collection of PDF documents.

It uses a FAISS vector index for efficient similarity search and Google's Gemini model for generating answers.

## Features

- Ingest multiple PDF documents.
- Ask questions about the documents from the command line.
- Search and download research papers from Arxiv.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd document_Q-A_agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set your API Key:**
    Set your Gemini API key as an environment variable.
    ```bash
    set GEMINI_API_KEY=<your-api-key>
    ```

4.  **Add Documents:**
    Place your PDF files into the `documents` directory.

5.  **Ingest Documents:**
    Run the ingestion script to process the documents and create the vector index.
    ```bash
    python ingest.py
    ```

## Usage

To ask a question, run the `app.py` script with your question as an argument:
```bash
python app.py "What is the main topic of the documents?"
```

To search Arxiv:
```bash
python app.py "search arxiv: <your query>"
```

To download a paper from an Arxiv URL:
```bash
python app.py "download: <arxiv_pdf_url>"
```