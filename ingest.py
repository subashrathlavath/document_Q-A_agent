
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

def ingest_documents(documents_dir="documents", data_dir="data"):
    """
    Ingests PDF documents from a directory, extracts text, creates embeddings,
    and saves a FAISS index and the text data.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF documents found in the 'documents' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to ingest.")

    # Using a smaller, faster model for local development
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_texts = []
    all_metadata = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(documents_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text:
                    # Simple chunking by page. More advanced chunking could be added here.
                    all_texts.append(text)
                    all_metadata.append({"document": pdf_file, "page": page_num + 1})
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    if not all_texts:
        print("No text could be extracted from the PDF files.")
        return

    print("Generating embeddings for the extracted text...")
    embeddings = model.encode(all_texts, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create and save the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(data_dir, "faiss_index.bin"))

    # Save the texts and metadata
    with open(os.path.join(data_dir, "documents_data.jsonl"), "w", encoding="utf-8") as f:
        for text, meta in zip(all_texts, all_metadata):
            f.write(json.dumps({"text": text, "metadata": meta}) + "\n")

    print("Ingestion complete.")
    print(f"FAISS index and document data saved to the '{data_dir}' directory.")

if __name__ == "__main__":
    ingest_documents()
