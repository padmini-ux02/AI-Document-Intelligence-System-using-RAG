"""
config.py — Centralized configuration for Nexus AI.
All tunable constants live here; values can be overridden via environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Models ───────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "GPT-3.5 Turbo (Fast)":  "gpt-3.5-turbo",
    "GPT-4o Mini (Balanced)": "gpt-4o-mini",
    "GPT-4o (Best)":          "gpt-4o",
}
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Chunking ─────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_LEN         = 40  # chars

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K          = int(os.getenv("TOP_K", "6"))
BM25_WEIGHT            = float(os.getenv("BM25_WEIGHT", "0.4"))   # weight in hybrid fusion
TFIDF_WEIGHT           = float(os.getenv("TFIDF_WEIGHT", "0.6"))  # weight in hybrid fusion
RERANK_TOP_K           = int(os.getenv("RERANK_TOP_K", "3"))      # chunks to keep after re-ranking
QUERY_EXPANSION_TERMS  = int(os.getenv("QUERY_EXPANSION_TERMS", "3"))

# ── Embeddings / TF-IDF ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "8000"))
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ── Analytics ─────────────────────────────────────────────────────────────────
KEYWORD_TOP_N  = int(os.getenv("KEYWORD_TOP_N", "15"))
TOPICS_TOP_N   = int(os.getenv("TOPICS_TOP_N", "8"))
CLUSTER_COUNT  = int(os.getenv("CLUSTER_COUNT", "4"))

# ── Sessions ──────────────────────────────────────────────────────────────────
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "nexus_sessions")

# ── Web Loader ────────────────────────────────────────────────────────────────
WEB_REQUEST_TIMEOUT = int(os.getenv("WEB_REQUEST_TIMEOUT", "15"))
MAX_WEB_CHARS       = int(os.getenv("MAX_WEB_CHARS", "50000"))
