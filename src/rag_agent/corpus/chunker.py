"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        file_path = Path(file_path)

        # 1. Validate the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. Route by file type (this workshop uses Markdown only)
        suffix = file_path.suffix.lower()
        if suffix in (".md", ".markdown"):
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # 3. Work out metadata (topic, difficulty, ...) once for the file
        metadata = self._infer_metadata(file_path, metadata_overrides)

        # 4. Turn each raw piece into a DocumentChunk with a stable ID
        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            text = raw["text"].strip()
            if not text:
                continue
            chunk_id = VectorStoreManager.generate_chunk_id(metadata.source, text)
            chunks.append(
                DocumentChunk(chunk_id=chunk_id, chunk_text=text, metadata=metadata)
            )

        logger.info(f"Chunked {file_path.name} into {len(chunks)} chunks")

        # 5. Return the list
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        # TODO: implement — iterate and collect, handle per-file errors
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # TODO: implement using langchain_community.document_loaders.PyPDFLoader
        # and langchain.text_splitter.RecursiveCharacterTextSplitter
        raise NotImplementedError

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        text = file_path.read_text(encoding="utf-8")

        # Step 1: split the document at its headings (#, ##, ###),
        # so each section becomes its own piece.
        sections = self._split_by_headers(text)

        # Step 2: any section that is still too long is cut into
        # smaller overlapping windows.
        raw_chunks: list[dict] = []
        for header, body in sections:
            for piece in self._split_long_text(body, chunk_size, chunk_overlap):
                # Keep the heading with the text so the chunk has context
                chunk_text = f"{header}\n\n{piece}" if header else piece
                raw_chunks.append({"text": chunk_text, "header": header})
        return raw_chunks

    @staticmethod
    def _split_by_headers(text: str) -> list[tuple[str, str]]:
        """
        Split Markdown text into (header_path, body) sections.

        Tries LangChain's MarkdownHeaderTextSplitter first; if that
        package is unavailable, falls back to a simple line-based parser.
        """
        headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter

            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            sections = []
            for doc in splitter.split_text(text):
                header = " > ".join(
                    doc.metadata[key] for key in ("h1", "h2", "h3") if key in doc.metadata
                )
                sections.append((header, doc.page_content))
            return sections
        except ImportError:
            pass

        # Fallback: walk the lines and start a new section at each heading
        sections: list[tuple[str, str]] = []
        current_headers: dict[int, str] = {}
        body_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(body_lines).strip()
            if body:
                header = " > ".join(current_headers[lvl] for lvl in sorted(current_headers))
                sections.append((header, body))
            body_lines.clear()

        for line in text.splitlines():
            stripped = line.lstrip()
            level = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= level <= 3 and stripped[level:level + 1] == " ":
                flush()
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = stripped[level:].strip()
            else:
                body_lines.append(line)
        flush()
        return sections

    @staticmethod
    def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Cut text into pieces of at most chunk_size characters, with
        chunk_overlap characters shared between neighbouring pieces.
        Tries to break at paragraph, line or word boundaries.
        """
        text = text.strip()
        if len(text) <= chunk_size:
            return [text] if text else []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)
        except ImportError:
            pass

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer to cut at a paragraph break, then a newline, then a space
                for sep in ("\n\n", "\n", " "):
                    cut = text.rfind(sep, start + chunk_overlap + 1, end)
                    if cut != -1:
                        end = cut
                        break
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
        return [p for p in pieces if p]

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        # Known topics, written the way the rest of the project expects them
        known_topics = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmann": "BoltzmannMachine",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }
        difficulties = {"beginner", "intermediate", "advanced"}
        bonus_topics = {"SOM", "BoltzmannMachine", "GAN"}

        # e.g. "lstm_intermediate.md" → ["lstm", "intermediate"]
        parts = file_path.stem.lower().split("_")

        # Difficulty = last part if it is a known level, else a default
        if len(parts) > 1 and parts[-1] in difficulties:
            difficulty = parts[-1]
            topic_part = "_".join(parts[:-1])
        else:
            difficulty = "intermediate"
            topic_part = "_".join(parts)

        # Topic = known name if we recognise it, otherwise the filename itself
        topic = known_topics.get(topic_part.replace("_", ""), topic_part.upper())

        values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": topic in bonus_topics,
        }

        # Anything passed in explicitly wins over what we guessed
        if overrides:
            for key, value in overrides.items():
                if key in values and value is not None:
                    values[key] = value
            if "topic" in overrides and "is_bonus" not in overrides:
                values["is_bonus"] = values["topic"] in bonus_topics

        return ChunkMetadata(**values)
