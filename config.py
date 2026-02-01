# config.py
import os
from pathlib import Path
from dotenv import load_dotenv


# Always load .env from the project root (same folder as config.py)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# Dataset name (useful for future multi-dataset projects)
DATASET_NAME = os.getenv("DATASET_NAME", "fia")

# PDF cleaning (header/footer removal)
CLEAN_HEADERS_FOOTERS = os.getenv("CLEAN_HEADERS_FOOTERS", "1") == "1"
HF_MIN_PAGE_FRACTION = float(os.getenv("HF_MIN_PAGE_FRACTION", "0.6"))  # >=60% pages
HF_MIN_LINE_LEN = int(os.getenv("HF_MIN_LINE_LEN", "8"))
HF_MAX_LINE_LEN = int(os.getenv("HF_MAX_LINE_LEN", "180"))
HF_MAX_REMOVE_PER_PAGE = int(os.getenv("HF_MAX_REMOVE_PER_PAGE", "6"))  # safety cap

# Chunking
CHUNKER = os.getenv("CHUNKER", "sentence")  # "sentence" or "overlap"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
OVERLAP = int(os.getenv("OVERLAP", "100"))  # for overlap chunker (chars)
OVERLAP_SENTENCES = int(os.getenv("OVERLAP_SENTENCES", "1"))  # for sentence chunker (units)

# Chroma
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# If user didn't set collection explicitly, choose based on dataset + chunker
_default_collection = os.getenv("CHROMA_COLLECTION", f"{DATASET_NAME}_{CHUNKER}")
CHROMA_COLLECTION = _default_collection

# Reranking
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RECALL_K = int(os.getenv("RECALL_K", "40"))          # how many to fetch from vector db
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4.1-mini")
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "900"))  # per chunk snippet

# Retrieval
TOP_K = int(os.getenv("TOP_K", "6"))
