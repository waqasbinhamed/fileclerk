"""File discovery, embedding, and ChromaDB indexing."""
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

import chromadb
import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

from extractor import SUPPORTED_EXTENSIONS, extract_text

console = Console()
logger = logging.getLogger(__name__)


def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_id(path_str: str) -> str:
    return hashlib.md5(path_str.encode()).hexdigest()


def discover_files(root: "str | Path") -> Iterator[Path]:
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self.chroma_path = config.get("chroma_path", ".chromadb")
        self.collection_name = config.get("collection_name", "documents")
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 16)

        self._client = None
        self._collection = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy-initialised heavy resources
    # ------------------------------------------------------------------

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            console.print(f"[blue]Loading embedding model {self.embedding_model_name}...[/blue]")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    # ------------------------------------------------------------------
    # Core indexing logic
    # ------------------------------------------------------------------

    def get_indexed_hashes(self) -> dict:
        results = self.collection.get(include=["metadatas"])
        return {
            m["path"]: m["hash"]
            for m in results["metadatas"]
            if m and "path" in m and "hash" in m
        }

    def remove_stale(self, valid_paths: set) -> None:
        results = self.collection.get(include=["metadatas"])
        stale_ids = [
            id_
            for id_, m in zip(results["ids"], results["metadatas"])
            if m and m.get("path") not in valid_paths
        ]
        if stale_ids:
            console.print(f"[yellow]Removing {len(stale_ids)} stale entries from index[/yellow]")
            self.collection.delete(ids=stale_ids)

    def index_path(self, root: "str | Path", force: bool = False) -> None:
        root = Path(root)
        indexed_hashes = {} if force else self.get_indexed_hashes()

        files = list(discover_files(root))
        console.print(f"[green]Found {len(files)} supported files under {root}[/green]")

        valid_paths = {str(f.resolve()) for f in files}
        self.remove_stale(valid_paths)

        to_index = [
            (p, get_file_hash(p))
            for p in files
            if force or indexed_hashes.get(str(p.resolve())) != get_file_hash(p)
        ]
        console.print(f"[green]{len(to_index)} file(s) need indexing[/green]")

        if not to_index:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing...", total=len(to_index))
            for i in range(0, len(to_index), self.batch_size):
                batch = to_index[i : i + self.batch_size]
                self._index_batch(batch)
                progress.advance(task, len(batch))
                _log_ram("batch")

    def _index_batch(self, batch: list) -> None:
        extractions, hashes = [], []
        for path, file_hash in batch:
            result = extract_text(path)
            if result is None:
                continue
            extractions.append(result)
            hashes.append(file_hash)

        if not extractions:
            return

        texts = [e["text_snippet"] for e in extractions]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        ids = [_path_id(e["path"]) for e in extractions]
        metadatas = [
            {
                "path": e["path"],
                "filename": e["filename"],
                "filetype": e["filetype"],
                "hash": h,
                "date_indexed": datetime.now().isoformat(),
                "classification": "",
                "confidence": 0.0,
                "destination": "",
                "char_count": e["char_count"],
                "extraction_method": e["extraction_method"],
            }
            for e, h in zip(extractions, hashes)
        ]

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # ------------------------------------------------------------------
    # Classification update
    # ------------------------------------------------------------------

    def update_classification(self, path: str, classification: dict) -> None:
        doc_id = _path_id(path)
        existing = self.collection.get(ids=[doc_id], include=["metadatas"])
        if not existing["ids"]:
            return

        meta = existing["metadatas"][0]
        meta.update(
            {
                "classification": classification.get("category", ""),
                "confidence": float(classification.get("confidence", 0.0)),
                "destination": classification.get("suggested_folder", ""),
            }
        )
        self.collection.update(ids=[doc_id], metadatas=[meta])

    # ------------------------------------------------------------------
    # Query / review helpers
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 10) -> list:
        embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        return [
            {**m, "similarity": round(1.0 - d, 4)}
            for m, d in zip(results["metadatas"][0], results["distances"][0])
        ]

    def get_needs_review(self) -> list:
        results = self.collection.get(include=["metadatas"])
        return [m for m in results["metadatas"] if m and m.get("classification") == "needs_review"]

    def get_all_classified(self) -> list:
        results = self.collection.get(include=["metadatas"])
        return [
            m
            for m in results["metadatas"]
            if m and m.get("classification") not in ("", "needs_review", None)
        ]

    def get_all_documents(self) -> tuple:
        """Returns (metadatas, documents) for all indexed docs."""
        results = self.collection.get(include=["metadatas", "documents"])
        return results["metadatas"], results["documents"]


def _log_ram(step: str) -> None:
    mem = psutil.virtual_memory()
    logger.debug("RAM after %s: %.1f%% (%dMB used)", step, mem.percent, mem.used // 1024 ** 2)
