import os
import faiss
import json
import arxiv
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import sys

# --- Configuration ---
HISTORY_DIR = "history"
DATA_DIR = "data"
DOCUMENTS_DIR = "documents"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOCUMENTS_DATA_PATH = os.path.join(DATA_DIR, "documents_data.jsonl")

# Configure the Gemini API
# IMPORTANT: Replace with your actual Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please set your GEMINI_API_KEY environment variable.")
    model = None

# --- Helper Functions ---
def save_chat_history(prompt, response):
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    history_file = os.path.join(HISTORY_DIR, "chat_history.jsonl")
    with open(history_file, "a") as f:
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

def search_arxiv(query):
    try:
        search = arxiv.Search(query=query, max_results=3)
        results = []
        for result in search.results():
            results.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "pdf_url": result.pdf_url
            })
        return results
    except Exception as e:
        return f"An error occurred while searching Arxiv: {e}"

def download_pdf(url, directory=DOCUMENTS_DIR):
    try:
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[url.split('/')[-1]])))
        filename = f"{paper.title.replace(' ', '_').replace('/', '_')}.pdf"
        filepath = os.path.join(directory, filename)
        paper.download_pdf(dirpath=directory, filename=filename)
        return filepath
    except Exception as e:
        return f"Failed to download PDF: {e}"

# --- Main Application Logic ---
def main():
    if not all([os.path.exists(FAISS_INDEX_PATH), os.path.exists(DOCUMENTS_DATA_PATH)]):
        print("It seems the document ingestion has not been run yet.")
        print("Please run `python ingest.py` first to process your PDF documents.")
        return

    # Load the FAISS index, document data, and sentence transformer model
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENTS_DATA_PATH, "r", encoding="utf-8") as f:
            documents_data = [json.loads(line) for line in f]
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("Usage: python app.py <your question>")
        print("Or: python app.py download: <pdf_url>")
        return

    if prompt.lower().startswith("download:"):
        url = prompt[len("download:"):].strip()
        print(f"Downloading {url}...")
        filepath = download_pdf(url)
        if isinstance(filepath, str) and os.path.exists(filepath):
            print(f"Successfully downloaded to {filepath}")
            print("Please run `python ingest.py` again to include the new document.")
        else:
            print(filepath)
        return

    if prompt.lower().startswith("search arxiv:"):
        query = prompt[len("search arxiv:"):].strip()
        print(f"Searching Arxiv for: {query}")
        results = search_arxiv(query)
        if isinstance(results, str):
            print(results)
        else:
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Title: {result['title']}")
                print(f"Authors: {', '.join(result['authors'])}")
                print(f"Summary: {result['summary']}")
                print(f"PDF URL: {result['pdf_url']}")
            # In a command-line tool, we can't ask for input to download.
            # We can suggest the next step.
            print("\nTo download a paper, run the script again with the paper's URL.")
        return

    if not model:
        print("The Gemini model is not available. Please check your API key.")
        return

    # Find relevant document chunks
    query_embedding = sentence_model.encode([prompt], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 most relevant chunks

    retrieved_chunks = [documents_data[i] for i in I[0]]

    # Prepare the context for the LLM
    context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
    llm_prompt = f"""
    Based on the following context from a research paper, please answer the user's question.

    Context:
    ---
    {context}
    ---

    Question: {prompt}

    Answer:
    """

    try:
        response = model.generate_content(llm_prompt)
        answer = response.text
        print(f"\nAnswer:\n{answer}")
        save_chat_history(prompt, answer)
    except Exception as e:
        print(f"An error occurred while generating the answer: {e}")

if __name__ == "__main__":
    main()