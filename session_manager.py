"""
session_manager.py — Save / load / export knowledge base sessions for Nexus AI.
Sessions are stored in a local `nexus_sessions/` directory as pickle files.
"""
import os
import json
import pickle
import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import SESSIONS_DIR


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def list_sessions() -> List[Dict[str, Any]]:
    """Return metadata for all saved sessions, sorted newest-first."""
    _ensure_sessions_dir()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".meta.json"):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                sessions.append(meta)
            except Exception:
                pass
    return sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)


def save_session(name: str, rag_system, chat_history: List[Dict]) -> str:
    """
    Persist a RAGSystem and its chat history to disk.
    Returns the session ID (slug).
    """
    _ensure_sessions_dir()
    slug = _slugify(name)
    timestamp = datetime.datetime.now().isoformat()

    # Pickle the RAG system (vector store + vectorizer)
    pkl_path = os.path.join(SESSIONS_DIR, f"{slug}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(rag_system, f)

    # Save metadata & chat history as JSON
    meta = {
        "id": slug,
        "name": name,
        "created_at": timestamp,
        "chunk_count": rag_system.get_chunk_count() if rag_system else 0,
        "qa_count": sum(1 for m in chat_history if m.get("role") == "user"),
        "history": chat_history,
    }
    meta_path = os.path.join(SESSIONS_DIR, f"{slug}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return slug


def load_session(session_id: str) -> Tuple[Any, List[Dict], Dict]:
    """
    Load a saved session.
    Returns (rag_system, chat_history, meta).
    """
    _ensure_sessions_dir()
    pkl_path  = os.path.join(SESSIONS_DIR, f"{session_id}.pkl")
    meta_path = os.path.join(SESSIONS_DIR, f"{session_id}.meta.json")

    if not os.path.exists(pkl_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Session '{session_id}' not found.")

    with open(pkl_path, "rb") as f:
        rag_system = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return rag_system, meta.get("history", []), meta


def delete_session(session_id: str):
    """Delete a saved session from disk."""
    _ensure_sessions_dir()
    for ext in [".pkl", ".meta.json"]:
        path = os.path.join(SESSIONS_DIR, f"{session_id}{ext}")
        if os.path.exists(path):
            os.remove(path)


def export_chat_markdown(chat_history: List[Dict], session_name: str = "Nexus AI") -> str:
    """Convert chat history to a readable Markdown document."""
    lines = [f"# {session_name} — Chat Export\n"]
    lines.append(f"*Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")
    for i, msg in enumerate(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"\n### 🧑 You\n{content}\n")
        elif role == "assistant":
            conf = msg.get("confidence", 0)
            lines.append(f"\n### 🧠 Nexus AI *(Confidence: {int(conf*100)}%)*\n{content}\n")
            sources = msg.get("sources", [])
            if sources:
                lines.append("\n**Sources:**\n")
                for s in sources:
                    lines.append(f"- `{s['source']}` (relevance: {int(s['score']*100)}%) — {s['content'][:120]}…\n")
    return "".join(lines)


def export_chat_json(chat_history: List[Dict], session_name: str = "Nexus AI") -> str:
    """Convert chat history to a JSON string."""
    payload = {
        "session": session_name,
        "exported_at": datetime.datetime.now().isoformat(),
        "messages": chat_history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    import re
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:40].strip("_") or "session"
