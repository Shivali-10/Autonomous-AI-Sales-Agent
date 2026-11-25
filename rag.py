"""FAISS + sentence-transformers RAG pipeline with offline TF-IDF fallback."""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from database import record_knowledge_doc


DEFAULT_KB_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
DEFAULT_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


class KnowledgeBaseRAG:
    """Manages document chunking, vector indexing in FAISS, and grounded retrieval."""

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.kb_dir = Path(kb_dir or os.getenv("KNOWLEDGE_BASE_DIR", DEFAULT_KB_DIR))
        self.index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR", DEFAULT_INDEX_DIR))
        self.db_path = db_path
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self.backend_mode: str = "uninitialized"
        self._st_model = None
        self._tfidf: Optional[TfidfVectorizer] = None

        if not self._load_persisted_index():
            self.rebuild_index()

    def _get_sentence_transformer(self):
        """Lazy-load SentenceTransformer (local cache first to avoid offline delays); return None if unavailable."""
        if self._st_model is not None:
            return self._st_model
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._st_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
                return self._st_model
            except Exception:
                self._st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                return self._st_model
        except Exception:
            return None

    @staticmethod
    def read_document_text(file_path: Path) -> str:
        """Extract plain text from .txt, .md, .pdf, .csv, or .json files."""
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".csv"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".json":
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            try:
                parsed = json.loads(raw)
                return json.dumps(parsed, indent=2)
            except Exception:
                return raw
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 90) -> List[str]:
        """Split text into overlapping semantic chunks respecting paragraph/sentence boundaries."""
        cleaned = re.sub(r"\r\n?", "\n", text).strip()
        if not cleaned:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}".strip() if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) <= chunk_size:
                    current = para
                else:
                    # Sentence-level split for oversized paragraphs
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    buf = ""
                    for sent in sentences:
                        if len(buf) + len(sent) + 1 <= chunk_size:
                            buf = f"{buf} {sent}".strip() if buf else sent
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = sent[:chunk_size]
                    current = buf

        if current:
            chunks.append(current)

        # Add lightweight overlap from previous chunk tail for continuity
        if overlap > 0 and len(chunks) > 1:
            overlapped: List[str] = [chunks[0]]
            for idx in range(1, len(chunks)):
                prev_tail = chunks[idx - 1][-overlap:].strip()
                if prev_tail and prev_tail not in chunks[idx]:
                    overlapped.append(f"...{prev_tail}\n{chunks[idx]}")
                else:
                    overlapped.append(chunks[idx])
            return overlapped

        return chunks

    def _encode_texts(self, texts: List[str], fit_tfidf: bool = False) -> np.ndarray:
        """Encode texts to L2-normalized float32 embeddings using SentenceTransformer or TF-IDF fallback."""
        model = self._get_sentence_transformer()
        if model is not None:
            try:
                vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                self.backend_mode = f"sentence-transformers ({EMBEDDING_MODEL_NAME}) + FAISS"
                return np.asarray(vecs, dtype="float32")
            except Exception:
                pass

        # Deterministic offline TF-IDF fallback
        if fit_tfidf or self._tfidf is None:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2048, stop_words="english")
            mat = self._tfidf.fit_transform(texts).toarray().astype("float32")
        else:
            mat = self._tfidf.transform(texts).toarray().astype("float32")

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self.backend_mode = "TF-IDF Vectorizer + FAISS (Offline Fallback)"
        return mat

    def rebuild_index(self) -> int:
        """Read all files in knowledge_base/, chunk them, build FAISS index, and persist to disk."""
        supported = {".txt", ".md", ".pdf", ".csv", ".json"}
        files = sorted(
            [p for p in self.kb_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
        )

        self.chunks = []
        for fpath in files:
            text = self.read_document_text(fpath)
            if not text.strip():
                continue
            doc_chunks = self.chunk_text(text)
            for idx, chunk in enumerate(doc_chunks):
                self.chunks.append(
                    {
                        "source": fpath.name,
                        "chunk_id": idx,
                        "text": chunk,
                    }
                )
            record_knowledge_doc(
                filename=fpath.name,
                title=fpath.stem.replace("_", " ").title(),
                chunk_count=len(doc_chunks),
                char_count=len(text),
                db_path=self.db_path,
            )

        if not self.chunks:
            self.index = None
            self.backend_mode = "empty"
            return 0

        texts = [c["text"] for c in self.chunks]
        embeddings = self._encode_texts(texts, fit_tfidf=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self._persist_index()
        return len(self.chunks)

    def _persist_index(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_dir / "kb.faiss"))
        meta = {
            "chunks": self.chunks,
            "backend_mode": self.backend_mode,
        }
        (self.index_dir / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._tfidf is not None:
            with open(self.index_dir / "tfidf.pkl", "wb") as f:
                pickle.dump(self._tfidf, f)

    def _load_persisted_index(self) -> bool:
        faiss_file = self.index_dir / "kb.faiss"
        chunks_file = self.index_dir / "chunks.json"
        if not faiss_file.exists() or not chunks_file.exists():
            return False
        try:
            meta = json.loads(chunks_file.read_text(encoding="utf-8"))
            self.chunks = meta.get("chunks", [])
            self.backend_mode = meta.get("backend_mode", "persisted")
            self.index = faiss.read_index(str(faiss_file))
            tfidf_file = self.index_dir / "tfidf.pkl"
            if tfidf_file.exists():
                with open(tfidf_file, "rb") as f:
                    self._tfidf = pickle.load(f)
            return len(self.chunks) > 0
        except Exception:
            return False

    def add_uploaded_file(self, filename: str, file_bytes: bytes) -> int:
        """Save an uploaded knowledge file into knowledge_base/ and rebuild FAISS index."""
        safe_name = Path(filename).name
        target = self.kb_dir / safe_name
        target.write_bytes(file_bytes)
        return self.rebuild_index()

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Retrieve top-k semantically relevant chunks from FAISS."""
        if not query or not query.strip() or self.index is None or not self.chunks:
            return []

        if "TF-IDF" in self.backend_mode and self._tfidf is not None:
            mat = self._tfidf.transform([query]).toarray().astype("float32")
            norm = np.linalg.norm(mat, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            q_vec = mat / norm
        else:
            q_vec = self._encode_texts([query], fit_tfidf=False)

        # Guard dimension mismatch if model switched between runs
        if q_vec.shape[1] != self.index.d:
            self.rebuild_index()
            if self.index is None:
                return []
            if "TF-IDF" in self.backend_mode and self._tfidf is not None:
                mat = self._tfidf.transform([query]).toarray().astype("float32")
                norm = np.linalg.norm(mat, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                q_vec = mat / norm
            else:
                q_vec = self._encode_texts([query], fit_tfidf=False)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_meta = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    text=chunk_meta["text"],
                    source=chunk_meta["source"],
                    chunk_id=int(chunk_meta["chunk_id"]),
                    score=float(score),
                )
            )
        return results
