import os
import re
import shutil
import fitz  
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import (
    PDF_PATH,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
def clean_text(text: str) -> str:
    """
    Basic cleaning while preserving Arabic regulatory text.
    """
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
    
def create_chunks(text, page_number):
    """
    Create overlapping character-based chunks.
    Each chunk retains the original page number.
    """
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        if end < text_length:
            possible_breaks = [
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("؟", start, end),
                text.rfind("؛", start, end),
                text.rfind("،", start, end),
            ]
            best_break = max(possible_breaks)
            if best_break > start + (CHUNK_SIZE * 0.5):
                end = best_break + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                }
            )
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
def extract_pdf_chunks():
    print(f"PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    all_chunks = []
    print(f"Pages: {len(doc)}")
    for page_index in range(len(doc)):
        page_number = page_index + 1
        page = doc[page_index]
        text = page.get_text("text")
        page_chunks = create_chunks(
            text,
            page_number
        )
        all_chunks.extend(page_chunks)
        print(
            f"\rProcessed page "
            f"{page_number}/{len(doc)} "
            f"| chunks: {len(all_chunks)}",
            end=""
        )
    doc.close()
    print("\n")
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
def build_chroma(chunks):
    if os.path.exists(CHROMA_DIR):
        print("Removing old Chroma database...")
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(
        f"Loading embedding model:\n"
        f"{EMBEDDING_MODEL}"
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )
    collection = client.get_or_create_collection(
        name="sama_finance"
    )
    documents = []
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        documents.append(chunk["text"])
        metadatas.append(
            {
                "source": os.path.basename(PDF_PATH),
                "page": chunk["page"],
                "chunk_id": i,
            }
        )

        ids.append(
            f"chunk_{i}"
        )

    print("\nCreating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("\nAdding chunks to Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    print(f"PDF: {os.path.basename(PDF_PATH)}")
    print(f"Pages processed: {len(set(c['page'] for c in chunks))}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Database: {CHROMA_DIR}")

def main():
    chunks = extract_pdf_chunks()
    if not chunks:
        raise RuntimeError(
            "No text was extracted from the PDF."
        )
    build_chroma(chunks)
if __name__ == "__main__":
    main()
