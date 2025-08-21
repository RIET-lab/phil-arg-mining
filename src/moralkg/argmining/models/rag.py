"""
RAG class:
Provide retrieval capabilities for End2End and CoT. This class does not perform final text generation.
"""


class RAG:
    def __init__(
        self,
        *,
        embedder: str | None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        top_k: int = 5,
        keep_index: bool = False,
    ) -> None:
        from moralkg import get_logger

        self.embedder_name = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_default = top_k
        self.keep_index = keep_index

        self.logger = get_logger(__name__)

        # Runtime
        self._embeddings = None
        self._splitter = None
        self._vs = None

    def build(self) -> None:
        """Instantiate embeddings, splitter, and prepare vector store holder."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedder_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        self._vs = None

    def destroy(self) -> None:
        """Release resources. No-operation if keep_index=True."""
        if self.keep_index:
            return
        self._vs = None
        self._splitter = None
        self._embeddings = None

    def add(self, corpus: list[str] | list[dict] | str) -> None:
        """
        Add documents to the index.
        Accepts either:
          - list[str] of file paths
          - list[dict] with keys {"id": str, "text": str, optional "metadata": dict}
          - str path to a text file
        """
        from pathlib import Path
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document

        if self._embeddings is None or self._splitter is None:
            raise RuntimeError("RAG.add called before build")

        # Normalize input to a list of (doc_id, text, metadata)
        items: list[tuple[str, str, dict]] = []

        def _read_file(p: Path) -> str:
            return p.read_text(encoding="utf-8", errors="ignore")

        if isinstance(corpus, str):
            p = Path(corpus)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"RAG.add: file not found: {p}")
            items.append((p.stem, _read_file(p), {"path": str(p)}))
        elif isinstance(corpus, list):
            if len(corpus) == 0:
                return
            if isinstance(corpus[0], str):
                for fp in corpus:  # type: ignore[index]
                    p = Path(fp)
                    if p.exists() and p.is_file():
                        items.append((p.stem, _read_file(p), {"path": str(p)}))
            elif isinstance(corpus[0], dict):
                for rec in corpus:  # type: ignore[index]
                    doc_id = str(rec.get("id") or len(items))
                    text = str(rec.get("text") or "")
                    md = rec.get("metadata") or {}
                    items.append((doc_id, text, md))
            else:
                raise TypeError("RAG.add: list must contain paths or dict records")
        else:
            raise TypeError("RAG.add: unsupported corpus type")

        # Chunk and convert to Documents
        docs: list[Document] = []
        metadatas: list[dict] = []
        for doc_id, text, md in items:
            if not text:
                continue
            splits = self._splitter.create_documents([text], metadatas=[{"id": doc_id, **md}])
            for idx, d in enumerate(splits):
                start = d.metadata.get("start_index")
                end = (start + len(d.page_content)) if isinstance(start, int) else None
                d.metadata.update({
                    "chunk_index": idx,
                    "chunk_id": f"{doc_id}#chunk_{idx}",
                    "offsets": [start, end] if start is not None and end is not None else None,
                })
                docs.append(d)
                metadatas.append(d.metadata)

        if not docs:
            return

        if self._vs is None:
            self._vs = FAISS.from_documents(docs, self._embeddings)
        else:
            self._vs.add_documents(docs)

    # Retrieval
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if self._vs is None:
            return []
        k = int(top_k or self.top_k_default)
        results = self._vs.similarity_search_with_score(query, k=k)
        contexts: list[dict] = []
        for doc, score in results:
            md = dict(doc.metadata)
            contexts.append({
                "id": str(md.get("id")),
                "chunk_id": str(md.get("chunk_id")),
                "text": doc.page_content,
                "score": float(score),
                "offsets": md.get("offsets"),
                "metadata": {k: v for k, v in md.items() if k not in {"chunk_id", "offsets", "id"}},
            })
        return contexts
