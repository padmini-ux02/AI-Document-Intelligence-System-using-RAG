"""
RAG Pipeline — Zero-dependency local mode using sklearn TF-IDF.
If an OpenAI API key is provided, uses OpenAI embeddings + GPT for accurate answers.
"""
import os
import re
import json
import heapq
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Document Loaders ────────────────────────────────────────────────────────
def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def _load_docx(path: str) -> str:
    import docx2txt
    return docx2txt.process(path)

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}

# ─── Text Chunker ─────────────────────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks. 
    Uses a robust, recursive-style approach in pure Python to avoid 
    heavy dependencies like sentence-transformers or torch.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if len(c) > 40]
    except Exception:
        # Robust Fallback Splitter (Pure Python)
        # If LangChain splitter fails due to environment issues (like missing torch)
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunk = text[start:]
            else:
                # Try to find a good breaking point (newline or space) near the end
                last_space = text.rfind(' ', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_space, last_newline)
                
                if break_point > start + (chunk_size // 2):
                    end = break_point
                    chunk = text[start:end]
                else:
                    chunk = text[start:end]
            
            if len(chunk.strip()) > 40:
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
            if start >= text_len or (end >= text_len):
                break
                
        return chunks

# ─── Document struct ──────────────────────────────────────────────────────────
class DocChunk:
    def __init__(self, text: str, source: str, page: int = 0):
        self.text = text
        self.source = source
        self.page = page

# ─── Local TF-IDF Vector Store ───────────────────────────────────────────────
class LocalVectorStore:
    """Lightweight vector store using TF-IDF + cosine similarity. No torch needed."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = None
        self.chunks: List[DocChunk] = []

    def add_chunks(self, chunks: List[DocChunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 4) -> List[Tuple[DocChunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_k = heapq.nlargest(k, range(len(sims)), key=lambda i: sims[i])
        return [(self.chunks[i], float(sims[i])) for i in top_k]

# ─── OpenAI Vector Store (when API key present) ───────────────────────────────
class OpenAIVectorStore:
    """FAISS-backed vector store using OpenAI text-embedding-3-small."""

    def __init__(self, api_key: str):
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.faiss_store = None
        self._FAISS = FAISS
        self.chunks: List[DocChunk] = []

    def add_chunks(self, chunks: List[DocChunk]):
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.docstore.document import Document
        self.chunks = chunks
        docs = [Document(page_content=c.text, metadata={"source": c.source, "page": c.page}) for c in chunks]
        self.faiss_store = self._FAISS.from_documents(docs, self.embeddings)

    def search(self, query: str, k: int = 4) -> List[Tuple[DocChunk, float]]:
        if not self.faiss_store:
            return []
        # FAISS similarity_search_with_score returns L2 distance by default
        # Lower distance = Higher similarity
        results_with_score = self.faiss_store.similarity_search_with_score(query, k=k)
        out = []
        for doc, distance in results_with_score:
            chunk = DocChunk(doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            # Rough conversion of L2 distance to a 0-1 similarity score
            # Higher distance means lower similarity. 
            similarity = 1.0 / (1.0 + distance)
            out.append((chunk, float(similarity)))
        return out

# ─── RAG System ───────────────────────────────────────────────────────────────
class RAGSystem:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_openai = bool(api_key)
        self.vector_store = None
        self.chat_history: List[Dict] = []
        self._llm = None
        if self.use_openai:
            os.environ["OPENAI_API_KEY"] = api_key

    def _get_llm(self):
        if self._llm is None and self.use_openai:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        return self._llm

    # ── Document Ingestion ───────────────────────────────────────────────
    def process_and_index(self, file_paths: List[str]) -> int:
        all_chunks: List[DocChunk] = []
        for path in file_paths:
            ext = os.path.splitext(path)[-1].lower()
            loader = LOADERS.get(ext)
            if not loader:
                continue
            try:
                raw = loader(path)
                texts = split_into_chunks(raw)
                fname = os.path.basename(path)
                all_chunks.extend(DocChunk(t, fname) for t in texts)
            except Exception as e:
                print(f"[WARN] Could not load {path}: {e}")

        if not all_chunks:
            return 0

        if self.use_openai:
            self.vector_store = OpenAIVectorStore(self.api_key)
        else:
            self.vector_store = LocalVectorStore()

        try:
            self.vector_store.add_chunks(all_chunks)
        except Exception as e:
            err_msg = str(e).lower()
            if "invalid_api_key" in err_msg or "401" in err_msg:
                raise ValueError("❌ Invalid OpenAI API Key. Please check your key in the sidebar.")
            elif "insufficient_quota" in err_msg or "429" in err_msg:
                raise ValueError("⚠️ OpenAI Quota Exceeded! Your API key has no balance left. \n\n**PRO-TIP:** Clear the API Key in the sidebar to switch to **Local Mode** (no cost!).")
            raise Exception(f"Failed to index documents: {str(e)}")
            
        return len(all_chunks)

    # ── Query ────────────────────────────────────────────────────────────
    def get_response(self, query: str) -> Dict[str, Any]:
        if not self.vector_store:
            return {"answer": "⚠️ No documents indexed yet.", "sources": [], "confidence": 0.0}

        results = self.vector_store.search(query, k=6)
        if not results:
            return {"answer": "No relevant content found.", "sources": [], "confidence": 0.0}

        sources = [{"content": chunk.text, "source": chunk.source, "score": round(score, 3)} for chunk, score in results]
        # Look at the best score for confidence, rather than average, for a more optimistic rate
        max_score = float(max([s["score"] for s in sources]))

        if self.use_openai:
            answer = self._openai_answer(query, results)
            # Optimistic confidence boost
            confidence = min(0.99, max_score * 1.5)
        else:
            # Local fallback: Extract the exact most relevant sentences from top chunks
            # instead of returning the entire broad chunk.
            best_sentences = []
            for chunk, score in results[:3]:
                if score > 0.05:
                    # Simple sentence splitter
                    sens = [s.strip() for s in chunk.text.replace("? ", ". ").replace("! ", ". ").split(". ") if len(s.strip()) > 10]
                    # Score each sentence in the chunk against the query
                    try:
                        s_vecs = self.vector_store.vectorizer.transform(sens)
                        q_vec = self.vector_store.vectorizer.transform([query])
                        s_sims = cosine_similarity(q_vec, s_vecs).flatten()
                        best_idx = np.argmax(s_sims)
                        if s_sims[best_idx] > 0.1:
                            best_sentences.append(sens[best_idx])
                    except:
                        continue
            
            if not best_sentences:
                answer = "No exact match found in the documents."
            else:
                # Take top 2 unique best sentences for an "exact short answer"
                seen = set()
                unique_sens = [x for x in best_sentences if not (x in seen or seen.add(x))]
                answer = " ".join(unique_sens[:2])
            
            confidence = max_score

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "sources": sources, "confidence": round(confidence, 2)}

    def _openai_answer(self, query: str, results: List[Tuple[DocChunk, float]]) -> str:
        context = "\n\n".join([f"[{i+1}] {r[0].text}" for i, r in enumerate(results)])
        history_text = ""
        for m in self.chat_history[-6:]:
            history_text += f"{m['role'].capitalize()}: {m['content']}\n"

        prompt = f"""You are a precise document-intelligence assistant.
Answer the user's question EXACTLY and SHORTLY using ONLY the provided context.

STRICT RULES:
1. MAXIMUM 2 SENTENCES. Be extremely concise.
2. Provide the EXACT answer requested. No intro, no filler.
3. If not in context, say: "Information not found."
4. Format: One single dense line.

CONTEXT:
{context}

USER QUESTION: {query}

Exact Short Answer:"""
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            # Post-process to remove all double newlines and extra spaces
            clean_ans = " ".join(response.content.split())
            return clean_ans
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    # ── Summarization ────────────────────────────────────────────────────
    def generate_summary(self) -> str:
        if self.vector_store is None or not self.vector_store.chunks:
            return "No documents to summarize."
        
        # Use first few chunks for a quick summary
        store = self.vector_store
        sample_chunks = store.chunks[:8]
        sample = " ".join(c.text for c in sample_chunks)[:4000]
        if self.use_openai:
            try:
                llm = self._get_llm()
                r = llm.invoke(f"Summarize this document content in 3 sentences:\n\n{sample}")
                return r.content
            except Exception as e:
                return f"Summary error: {e}"
        return sample[:500] + "..."

    # ── Keywords ─────────────────────────────────────────────────────────
    def extract_keywords(self, top_n: int = 8) -> List[str]:
        if not self.vector_store or not self.vector_store.chunks:
            return []
        text = " ".join(c.text for c in self.vector_store.chunks[:20])
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=200)
            m = vec.fit_transform([text[:15000]])
            scores = m.toarray().flatten()
            names = vec.get_feature_names_out()
            top = np.argsort(scores)[::-1][:top_n]
            return [names[i] for i in top]
        except:
            return []

    # ── Topics (using regex NER fallback without spacy) ──────────────────
    def extract_topics(self, top_n: int = 6) -> List[str]:
        if not self.vector_store or not self.vector_store.chunks:
            return []
        text = " ".join(c.text for c in self.vector_store.chunks[:10])[:5000]
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                os.system("python -m spacy download en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            ents = [e.text for e in doc.ents if e.label_ in ("ORG", "GPE", "PERSON", "PRODUCT", "EVENT")]
            return list(dict.fromkeys(ents))[:top_n]
        except:
            # Fallback: find Title-Case words as rough topic extraction
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
            if not words:
                return []
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            return [str(t[0]) for i, t in enumerate(sorted_items) if i < top_n]

    def get_chunk_count(self) -> int:
        if not self.vector_store:
            return 0
        return len(self.vector_store.chunks)
