import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "sama_pdfs")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------

# Automatically find the PDF inside data/sama_pdfs
pdf_files = [
    f for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

if len(pdf_files) == 0:
    raise FileNotFoundError(
        f"No PDF found in: {PDF_DIR}"
    )

if len(pdf_files) > 1:
    raise RuntimeError(
        f"Multiple PDFs found in {PDF_DIR}.\n"
        f"Please keep only the PDF you want to index.\n"
        f"Found: {pdf_files}"
    )

PDF_PATH = os.path.join(PDF_DIR, pdf_files[0])


# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

# Character-based chunks.
# We deliberately keep these moderate because the document
# is regulatory and individual requirements can be precise.

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

VECTOR_TOP_K = 15
BM25_TOP_K = 15
RRF_CANDIDATE_K = 20
FINAL_TOP_K = 5

# Reciprocal Rank Fusion constant
RRF_K = 60


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

LLM_MODEL = "aya"

TEMPERATURE = 0.0


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

MAX_CONTEXT_CHUNKS = 5