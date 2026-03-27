#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections import Counter
import io
import hashlib
import json
import math
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*",
    category=UserWarning,
)

from dotenv import load_dotenv
from instructor import from_mistral
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from pypdf import PdfReader

load_dotenv(dotenv_path=Path(".env"), override=False)


PDF_DIR = Path("pdfs")
RAG_STORE_ROOT = Path(".rag_store")
PART_B_ANALYSIS_PATH = Path("part_b_analysis.txt")
REQUIREMENTS_PATH = Path("requirements.txt")

DEFAULT_COLLECTION_NAME = "verified_knowledge"
DEFAULT_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
DEFAULT_EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3
DEFAULT_SUPPORT_THRESHOLD = 0.2
DEFAULT_TEMPERATURE = 0.0
CHUNK_SIZE_OPTIONS = [100, 300, 500, 900, 1500, 3000]
TOP_K_OPTIONS = [1, 3, 5, 10]

REFUSAL_TEXT = "I don't know based on the provided document."
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

SUMMARY_HINTS = (
    "what is this document about",
    "what is the document about",
    "summary",
    "summar",
    "overview",
    "main topic",
    "main topics",
    "takeaway",
    "key requirement",
    "key requirements",
    "instruction",
    "instructions",
    "deliverable",
    "deliverables",
    "deadline",
    "deadlines",
    "phase",
    "phases",
    "outcome",
    "outcomes",
    "skill",
    "skills",
    "process",
    "workflow",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    persist_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    support_threshold: float
    chat_model: str
    embedding_model: str
    temperature: float = DEFAULT_TEMPERATURE
    rebuild: bool = False


@dataclass(frozen=True)
class PageWindowRequest:
    kind: str
    requested_count: int | None = None
    requested_start: int | None = None
    requested_end: int | None = None


@dataclass(frozen=True)
class PageWindow:
    kind: str
    start_page: int
    end_page: int
    requested_count: int | None = None
    requested_start: int | None = None
    requested_end: int | None = None
    truncated: bool = False


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\b\w+\b", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MistralRagasEmbeddings(BaseRagasEmbedding):
    def __init__(self, model: str, api_key: str | None) -> None:
        super().__init__()
        self.embeddings = MistralAIEmbeddings(model=model, api_key=api_key)

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await self.embeddings.aembed_query(text)


def ensure_mistral_key() -> bool:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    return bool(os.getenv("MISTRAL_API_KEY"))


def show_text_file(path: Path, title: str) -> None:
    if not path.exists():
        print(f"{title} not found at {path.resolve()}")
        return
    print(f"\n{title}\n{'=' * len(title)}\n")
    print(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "document"


def source_store_name(source_path: Path, chunk_size: int) -> str:
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return normalize_name(f"{source_path.stem}_{digest}_c{chunk_size}")


def chunk_cache_path(config: AppConfig) -> Path:
    return config.persist_dir / "chunks.jsonl"


def list_source_files(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(
        path
        for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
    )


def page_count_for_source(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def preferred_default_source(files: list[Path]) -> Path:
    qualified = [path for path in files if (page_count_for_source(path) or 0) > 10]
    if qualified:
        return max(qualified, key=lambda path: page_count_for_source(path) or 0)
    return files[0]


def print_source_menu(pdf_dir: Path, files: list[Path]) -> None:
    print(f"\nAvailable source files in {pdf_dir.resolve()}:")
    for index, path in enumerate(files, start=1):
        pages = page_count_for_source(path)
        if pages is None:
            print(f"  {index}. {path.name}")
        else:
            label = "meets 10+ page requirement" if pages > 10 else "too short for the rubric"
            print(f"  {index}. {path.name} ({pages} pages) [{label}]")


def choose_source_interactively(pdf_dir: Path) -> Path:
    files = list_source_files(pdf_dir)
    if not files:
        raise FileNotFoundError(
            f"No PDF, Markdown, or text files were found in {pdf_dir.resolve()}.\n"
            "Put your source documents in that folder and run the script again."
        )

    print_source_menu(pdf_dir, files)
    default_source = preferred_default_source(files)
    default_index = files.index(default_source) + 1
    while True:
        raw = input(f"Select a file [1-{len(files)}] (press Enter for {default_index}): ").strip()
        if not raw:
            return default_source
        try:
            index = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= index <= len(files):
            return files[index - 1]
        print("Choose one of the listed numbers.")


def choose_chunk_size_interactively() -> int:
    print("\nChoose a chunk size for indexing:")
    print("Chunk size controls how much text goes into each piece before indexing.")
    print("Smaller chunks are more precise. Larger chunks keep more context together.")
    print("Chunk size is measured in characters for RecursiveCharacterTextSplitter.")
    for index, size in enumerate(CHUNK_SIZE_OPTIONS, start=1):
        label = "smallest" if index == 1 else "largest" if index == len(CHUNK_SIZE_OPTIONS) else "medium"
        print(f"  {index}. {size} characters ({label})")

    while True:
        raw = input(f"Select a chunk size [1-{len(CHUNK_SIZE_OPTIONS)}] (press Enter for 900): ").strip()
        if not raw:
            return DEFAULT_CHUNK_SIZE
        try:
            index = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= index <= len(CHUNK_SIZE_OPTIONS):
            return CHUNK_SIZE_OPTIONS[index - 1]
        print("Choose one of the listed numbers.")


def choose_top_k_interactively() -> int:
    print("\nChoose how many chunks to check for each question:")
    print("K is how many pieces of the PDF the assistant reads before answering.")
    print("Smaller K is tighter. Larger K is better for summaries and broad questions.")
    for index, value in enumerate(TOP_K_OPTIONS, start=1):
        if value == 1:
            label = "tightest"
        elif value == DEFAULT_TOP_K:
            label = "default"
        elif value == max(TOP_K_OPTIONS):
            label = "broadest"
        else:
            label = "more context"
        print(f"  {index}. {value} chunks ({label})")

    while True:
        raw = input(f"Select a K value [1-{len(TOP_K_OPTIONS)}] (press Enter for {DEFAULT_TOP_K}): ").strip()
        if not raw:
            return DEFAULT_TOP_K
        try:
            index = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= index <= len(TOP_K_OPTIONS):
            return TOP_K_OPTIONS[index - 1]
        print("Choose one of the listed numbers.")


def resolve_source_path(source_arg: str | None, pdf_dir: Path) -> Path:
    if source_arg:
        candidate = Path(source_arg).expanduser()
        if not candidate.exists() and not candidate.is_absolute():
            folder_candidate = pdf_dir / candidate
            if folder_candidate.exists():
                candidate = folder_candidate
        if not candidate.exists():
            available = [path.name for path in list_source_files(pdf_dir)]
            raise FileNotFoundError(
                f"Source file not found: {candidate.resolve()}\n"
                f"Available files in {pdf_dir.resolve()}: {available}"
            )
        return candidate
    return choose_source_interactively(pdf_dir)


def resolve_chunk_size(chunk_size_arg: int | None) -> int:
    if chunk_size_arg is not None:
        return chunk_size_arg
    return choose_chunk_size_interactively()


def resolve_top_k(top_k_arg: int | None) -> int:
    if top_k_arg is not None:
        return top_k_arg
    return choose_top_k_interactively()


def normalize_chunk_overlap(chunk_size: int, requested_overlap: int | None) -> int:
    if requested_overlap is None:
        overlap = max(20, chunk_size // 5)
    else:
        overlap = requested_overlap
    overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
    return max(0, overlap)


def load_source_documents(source_path: Path) -> list[Document]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(source_path), mode="page")
        docs = loader.load()
        for index, doc in enumerate(docs, start=1):
            page_number = int(doc.metadata.get("page", index - 1)) + 1
            doc.metadata["source"] = str(source_path.resolve())
            doc.metadata["source_name"] = source_path.name
            doc.metadata["page_number"] = page_number
        return docs

    if suffix in {".md", ".markdown", ".txt"}:
        loader = TextLoader(str(source_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(source_path.resolve())
            doc.metadata["source_name"] = source_path.name
            doc.metadata["page_number"] = 1
        return docs

    raise ValueError(f"Unsupported source type: {source_path.suffix}")


def line_number_from_char_index(text: str, char_index: int) -> int:
    return text[: max(char_index, 0)].count("\n") + 1


def annotate_chunk_metadata(
    chunks: list[Document], source_docs: list[Document]
) -> list[Document]:
    original_by_page = {
        int(doc.metadata.get("page_number", 1)): doc.page_content for doc in source_docs
    }
    for chunk in chunks:
        page_number = int(chunk.metadata.get("page_number") or chunk.metadata.get("page", 0)) or 1
        original_text = original_by_page.get(page_number, chunk.page_content)
        start_index = int(chunk.metadata.get("start_index") or 0)
        end_index = min(start_index + len(chunk.page_content), len(original_text))
        line_start = line_number_from_char_index(original_text, start_index)
        line_end = line_number_from_char_index(original_text, end_index)
        if line_end < line_start:
            line_end = line_start
        chunk.metadata["page_number"] = page_number
        chunk.metadata["line_start"] = line_start
        chunk.metadata["line_end"] = line_end
    return chunks


def split_documents(
    source_docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(source_docs)
    return annotate_chunk_metadata(chunks, source_docs)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_value(value: Any) -> float:
    if hasattr(value, "value"):
        return safe_float(getattr(value, "value"), 0.0)
    if isinstance(value, dict) and "value" in value:
        return safe_float(value["value"], 0.0)
    return safe_float(value, 0.0)


def citation_label(metadata: dict[str, Any]) -> str:
    page_number = int(metadata.get("page_number") or metadata.get("page", 0)) or 1
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    if line_start is None or line_end is None:
        return f"p. {page_number}"
    if int(line_start) == int(line_end):
        return f"p. {page_number}, line {int(line_start)}"
    return f"p. {page_number}, lines {int(line_start)}-{int(line_end)}"


def truncate_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 1, 1)].rstrip() + "…"


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\w+\b", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def lexical_support_score(question: str, doc: Document) -> float:
    question_tokens = content_tokens(question)
    if not question_tokens:
        return 0.0
    doc_tokens = content_tokens(doc.page_content)
    if not doc_tokens:
        return 0.0
    overlap = len(question_tokens & doc_tokens)
    return overlap / len(question_tokens)


def retrieval_terms(text: str) -> list[str]:
    words = [
        token
        for token in re.findall(r"\b\w+\b", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    if not words:
        return []
    terms = list(words)
    terms.extend(f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1))
    return terms


def write_chunk_cache(config: AppConfig, chunks: list[Document]) -> None:
    cache_path = chunk_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for doc in chunks:
            handle.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


def load_cached_chunks(config: AppConfig) -> list[Document]:
    cache_path = chunk_cache_path(config)
    if not cache_path.exists():
        return []
    chunks: list[Document] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            page_content = str(payload.get("page_content", ""))
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            chunks.append(Document(page_content=page_content, metadata=metadata))
    return chunks


def rank_cached_chunks(question: str, chunks: list[Document], top_k: int) -> tuple[list[Document], list[float], float]:
    if not chunks:
        return [], [], 0.0

    query_terms = retrieval_terms(question)
    if not query_terms:
        return [], [], 0.0

    query_term_set = set(query_terms)
    query_counter = Counter(query_terms)
    normalized_question = re.sub(r"\s+", " ", question.lower()).strip()
    tokenized_docs: list[list[str]] = []
    df: Counter[str] = Counter()
    total_terms = 0

    for doc in chunks:
        terms = retrieval_terms(doc.page_content)
        tokenized_docs.append(terms)
        total_terms += len(terms)
        df.update(set(terms))

    avg_doc_len = total_terms / len(chunks) if chunks else 1.0
    doc_scores: list[tuple[float, float, Document]] = []

    for doc, terms in zip(chunks, tokenized_docs):
        if not terms:
            continue
        term_counts = Counter(terms)
        doc_len = len(terms) or 1
        bm25_score = 0.0
        for term, query_frequency in query_counter.items():
            freq = term_counts.get(term)
            if not freq:
                continue
            doc_freq = df.get(term, 0)
            idf = math.log(1.0 + (len(chunks) - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = freq * (1.5 + 1.0)
            denominator = freq + 1.5 * (1.0 - 0.75 + 0.75 * (doc_len / avg_doc_len))
            bm25_score += idf * (numerator / denominator) * min(query_frequency, 2)

        term_overlap = len(query_term_set & set(terms))
        coverage = term_overlap / len(query_term_set)
        exact_phrase = 1.0 if normalized_question and normalized_question in re.sub(
            r"\s+", " ", doc.page_content.lower()
        ) else 0.0
        page_bias = 0.15 if is_overview_question(question) and int(doc.metadata.get("page_number", 1)) <= 6 else 0.0
        score = bm25_score + (coverage * 1.5) + exact_phrase + page_bias
        doc_scores.append((score, coverage, doc))

    doc_scores.sort(key=lambda item: item[0], reverse=True)
    selected = doc_scores[:top_k]
    if not selected:
        return [], [], 0.0

    max_score = max(score for score, _, _ in selected)
    normalized_scores = [round(score / max_score, 4) if max_score > 0 else 0.0 for score, _, _ in selected]
    question_support = max(coverage for _, coverage, _ in selected)
    return [doc for _, _, doc in selected], normalized_scores, question_support


def is_overview_question(question: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    return any(hint in normalized for hint in SUMMARY_HINTS) or bool(
        re.search(r"\bsummar\w*\b", normalized)
    )


def parse_page_window_request(question: str) -> PageWindowRequest | None:
    normalized = re.sub(r"\s+", " ", question.lower()).strip()

    last_match = re.search(r"\b(?:last|final)\s+(\d+)\s+pages?\b", normalized)
    if last_match:
        return PageWindowRequest(kind="last", requested_count=int(last_match.group(1)))

    first_match = re.search(r"\b(?:first|opening|initial)\s+(\d+)\s+pages?\b", normalized)
    if first_match:
        return PageWindowRequest(kind="first", requested_count=int(first_match.group(1)))

    range_match = re.search(r"\bpages?\s+(\d+)\s*(?:to|-|through|thru)\s*(\d+)\b", normalized)
    if range_match:
        start_page = int(range_match.group(1))
        end_page = int(range_match.group(2))
        return PageWindowRequest(kind="range", requested_start=start_page, requested_end=end_page)

    return None


def resolve_page_window(request: PageWindowRequest, total_pages: int) -> PageWindow | None:
    if total_pages <= 0:
        return None

    if request.kind == "last" and request.requested_count is not None:
        requested_count = max(1, request.requested_count)
        end_page = total_pages
        start_page = max(1, total_pages - requested_count + 1)
        truncated = requested_count > total_pages
        return PageWindow(
            kind="last",
            start_page=start_page,
            end_page=end_page,
            requested_count=requested_count,
            truncated=truncated,
        )

    if request.kind == "first" and request.requested_count is not None:
        requested_count = max(1, request.requested_count)
        start_page = 1
        end_page = min(total_pages, requested_count)
        truncated = requested_count > total_pages
        return PageWindow(
            kind="first",
            start_page=start_page,
            end_page=end_page,
            requested_count=requested_count,
            truncated=truncated,
        )

    if request.kind == "range" and request.requested_start is not None and request.requested_end is not None:
        start_page = max(1, min(request.requested_start, request.requested_end))
        end_page = min(total_pages, max(request.requested_start, request.requested_end))
        if start_page > total_pages:
            return None
        truncated = end_page < max(request.requested_start, request.requested_end)
        return PageWindow(
            kind="range",
            start_page=start_page,
            end_page=end_page,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            truncated=truncated,
        )

    return None


def select_page_documents(source_docs: list[Document], start_page: int, end_page: int) -> list[Document]:
    return [
        doc
        for doc in source_docs
        if start_page <= int(doc.metadata.get("page_number", 1)) <= end_page
    ]


def page_window_label(window: PageWindow) -> str:
    if window.start_page == window.end_page:
        return f"p. {window.start_page}"
    return f"p. {window.start_page}-{window.end_page}"


def group_documents_for_summary(docs: list[Document], max_groups: int = 5) -> list[list[Document]]:
    if not docs:
        return []
    group_size = max(1, math.ceil(len(docs) / max_groups))
    return [docs[index : index + group_size] for index in range(0, len(docs), group_size)]


def build_page_window_context(docs: list[Document], per_page_limit: int = 650) -> str:
    blocks: list[str] = []
    for doc in docs:
        blocks.append(
            f"[{citation_label(doc.metadata)}]\n{truncate_text(doc.page_content, per_page_limit)}"
        )
    return "\n\n".join(blocks)


def serialize_page_window_sources(docs: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, group in enumerate(group_documents_for_summary(docs), start=1):
        page_numbers = sorted(int(doc.metadata.get("page_number", 1)) for doc in group)
        start_page = page_numbers[0]
        end_page = page_numbers[-1]
        snippet_parts = []
        for doc in group:
            snippet_parts.append(
                f"{citation_label(doc.metadata)}: {truncate_text(doc.page_content, 180)}"
            )
        sources.append(
            {
                "rank": index,
                "source": group[0].metadata.get("source_name") or group[0].metadata.get("source"),
                "page_number": f"{start_page}-{end_page}" if start_page != end_page else start_page,
                "line_start": group[0].metadata.get("line_start"),
                "line_end": group[0].metadata.get("line_end"),
                "citation": f"p. {start_page}-{end_page}" if start_page != end_page else f"p. {start_page}",
                "relevance_score": 1.0,
                "snippet": " ".join(snippet_parts),
            }
        )
    return sources


def extractive_page_window_answer(
    question: str, docs: list[Document], window: PageWindow
) -> str:
    prefix = f"Summary of {page_window_label(window)}:\n"
    if window.truncated:
        prefix = (
            f"The document is shorter than the requested page window, so I summarized "
            f"{page_window_label(window)} as available.\n\n"
        )

    lines: list[str] = []
    for doc in docs[:8]:
        lines.append(
            f"- {citation_label(doc.metadata)}: {truncate_text(doc.page_content, 220)}"
        )
    remaining = len(docs) - min(len(docs), 8)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more pages in the requested range.")
    if not lines:
        return REFUSAL_TEXT
    return prefix + "\n".join(lines)


def summarize_page_window_answer(
    question: str,
    docs: list[Document],
    window: PageWindow,
    config: AppConfig,
    llm_chain,
) -> str:
    context = build_page_window_context(docs)
    page_label = page_window_label(window)
    if llm_chain is not None:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You summarize only the provided pages from a document. "
                        "Do not use outside knowledge. "
                        "Write a concise summary of the requested page window. "
                        "Cite factual claims using page citations or page ranges in square brackets.",
                    ),
                    (
                        "human",
                        "Question: {question}\n\nRequested pages: {page_label}\n\nContext:\n{context}\n\nSummary:",
                    ),
                ]
            )
            answer = (
                prompt
                | build_chat_model(config)
                | StrOutputParser()
            ).invoke(
                {
                    "question": question,
                    "page_label": page_label,
                    "context": context,
                }
            )
            answer = answer.strip()
            if answer:
                if window.truncated:
                    return (
                        f"The document is shorter than the requested page window, so I summarized {page_label} as available.\n\n"
                        f"{answer}"
                    )
                return answer
        except Exception:
            pass

    return extractive_page_window_answer(question, docs, window)


def distance_to_support(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(float(distance), 0.0))


def retrieve_with_scores(
    vectorstore: Chroma, question: str, top_k: int
) -> dict[str, Any]:
    try:
        pairs = vectorstore.similarity_search_with_score(question, k=top_k)
        docs = [doc for doc, _ in pairs]
        scores = [distance_to_support(distance) for _, distance in pairs]
    except Exception:
        docs = vectorstore.similarity_search(question, k=top_k)
        scores = [lexical_support_score(question, doc) for doc in docs]

    question_support = max((lexical_support_score(question, doc) for doc in docs), default=0.0)
    return {
        "question": question,
        "source_docs": docs,
        "scores": scores,
        "question_support": question_support,
        "max_score": max(scores) if scores else 0.0,
        "allow_summary": is_overview_question(question),
    }


def format_context(docs: list[Document], scores: list[float]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs):
        score = scores[index] if index < len(scores) else None
        score_part = f" | relevance={score:.2f}" if score is not None else ""
        blocks.append(
            f"[Source {index + 1} | {citation_label(doc.metadata)}{score_part}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def serialize_sources(docs: list[Document], scores: list[float]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        score = scores[index] if index < len(scores) else None
        serialized.append(
            {
                "rank": index + 1,
                "source": doc.metadata.get("source_name") or doc.metadata.get("source"),
                "page_number": doc.metadata.get("page_number"),
                "line_start": doc.metadata.get("line_start"),
                "line_end": doc.metadata.get("line_end"),
                "citation": citation_label(doc.metadata),
                "relevance_score": round(score, 4) if score is not None else None,
                "snippet": doc.page_content.strip(),
            }
        )
    return serialized


def extractive_answer_from_payload(payload: dict[str, Any]) -> str:
    source_docs: list[Document] = payload.get("source_docs", [])
    if not source_docs:
        return REFUSAL_TEXT

    scores: list[float] = payload.get("scores", [])
    if scores:
        best_index = max(
            range(len(source_docs)),
            key=lambda index: scores[index] if index < len(scores) else -1.0,
        )
        best_doc = source_docs[best_index]
    else:
        best_doc = source_docs[0]

    citation = citation_label(best_doc.metadata)
    lines = [line.strip("•*- \t") for line in best_doc.page_content.splitlines() if line.strip()]
    if lines:
        answer = " ".join(lines[:4])
    else:
        answer = truncate_text(best_doc.page_content, 280)
    return f"{answer} [{citation}]"


def print_query_result(result: dict[str, Any]) -> None:
    print("\nAnswer:")
    print(result["answer"])
    if result.get("answer") == REFUSAL_TEXT:
        print("\nFeedback: that answer is not in the current PDF.")
    if "question_support" in result:
        print(f"\nQuestion support: {result['question_support']:.2f}")
    print("\nSources:")
    sources = result.get("sources") or []
    if not sources:
        print("  None")
        return
    for source in sources:
        score = source.get("relevance_score")
        score_text = f"{score:.2f}" if isinstance(score, float) else "n/a"
        print(f"  - {source['citation']} | relevance={score_text}")
        print(f"    {truncate_text(source['snippet'])}")


def print_index_summary(config: AppConfig, source_docs: list[Document], chunks: list[Document], backend: str) -> None:
    print("Index built successfully.")
    print(f"Source file: {config.source_path}")
    print(f"Pages/documents loaded: {len(source_docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size / overlap: {config.chunk_size} / {config.chunk_overlap}")
    print(f"Embedding backend: {backend}")
    print(f"Vector store: {config.persist_dir}")


def vectorstore_dir_for_backend(config: AppConfig, backend: str) -> Path:
    return config.persist_dir / backend


def manifest_path(config: AppConfig) -> Path:
    return config.persist_dir / "manifest.json"


def write_manifest(
    config: AppConfig,
    source_count: int,
    chunk_count: int,
    embedding_backend: str,
) -> None:
    config.persist_dir.mkdir(parents=True, exist_ok=True)
    store_dir = vectorstore_dir_for_backend(config, embedding_backend)
    store_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "source_path": str(config.source_path.resolve()),
        "source_mtime": config.source_path.stat().st_mtime if config.source_path.exists() else None,
        "collection_name": config.collection_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "support_threshold": config.support_threshold,
        "chat_model": config.chat_model,
        "embedding_model": config.embedding_model,
        "temperature": config.temperature,
        "source_count": source_count,
        "chunk_count": chunk_count,
        "embedding_backend": embedding_backend,
        "vectorstore_dir": str(store_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path(config).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(config: AppConfig) -> dict[str, Any] | None:
    path = manifest_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def embedding_backend_for_manifest(config: AppConfig) -> str:
    data = read_manifest(config)
    if data and data.get("embedding_backend") in {"mistral", "local-hash"}:
        return str(data["embedding_backend"])
    return "mistral"


def build_embeddings(config: AppConfig) -> Embeddings:
    return MistralAIEmbeddings(
        model=config.embedding_model,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def local_embeddings() -> HashEmbeddings:
    return HashEmbeddings()


def embedding_function_for_backend(config: AppConfig, backend: str):
    if backend == "local-hash":
        return local_embeddings()
    return build_embeddings(config)


def build_vectorstore_from_documents(
    source_docs: list[Document], config: AppConfig
) -> tuple[Chroma, list[Document], str]:
    chunks = split_documents(source_docs, config.chunk_size, config.chunk_overlap)
    attempts = [
        ("mistral", build_embeddings(config)),
        ("local-hash", local_embeddings()),
    ]

    last_error: Exception | None = None
    for backend, embedding in attempts:
        store_dir = vectorstore_dir_for_backend(config, backend)
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding,
                    collection_name=config.collection_name,
                    persist_directory=str(store_dir),
                )
            return vectorstore, chunks, backend
        except Exception as exc:
            last_error = exc
            if store_dir.exists():
                shutil.rmtree(store_dir)

    raise RuntimeError("Unable to build vector store.") from last_error


def load_vectorstore(config: AppConfig) -> Chroma:
    backend = embedding_backend_for_manifest(config)
    data = read_manifest(config) or {}
    store_dir = Path(data.get("vectorstore_dir") or vectorstore_dir_for_backend(config, backend))
    return Chroma(
        collection_name=config.collection_name,
        persist_directory=str(store_dir),
        embedding_function=embedding_function_for_backend(config, backend),
    )


def manifest_matches(config: AppConfig) -> bool:
    data = read_manifest(config)
    if not data:
        return False

    expected_source = str(config.source_path.resolve())
    if data.get("source_path") != expected_source:
        return False
    if data.get("collection_name") != config.collection_name:
        return False
    if data.get("chunk_size") != config.chunk_size:
        return False
    if data.get("chunk_overlap") != config.chunk_overlap:
        return False
    if safe_float(data.get("support_threshold"), -1.0) != safe_float(config.support_threshold, -1.0):
        return False
    if data.get("chat_model") != config.chat_model:
        return False
    if data.get("embedding_model") != config.embedding_model:
        return False
    if safe_float(data.get("temperature"), -1.0) != safe_float(config.temperature, -1.0):
        return False
    if data.get("embedding_backend", "mistral") not in {"mistral", "local-hash"}:
        return False
    expected_store_dir = str(
        vectorstore_dir_for_backend(config, data.get("embedding_backend", "mistral")).resolve()
    )
    if data.get("vectorstore_dir") != expected_store_dir:
        return False
    if not chunk_cache_path(config).exists():
        return False

    if config.source_path.exists():
        stored_mtime = data.get("source_mtime")
        if stored_mtime is None:
            return False
        if abs(float(stored_mtime) - config.source_path.stat().st_mtime) > 1e-6:
            return False

    return True


def rebuild_index(config: AppConfig) -> tuple[Chroma, list[Document], list[Document], str]:
    if not config.source_path.exists():
        raise FileNotFoundError(f"Source file not found: {config.source_path}")

    if config.persist_dir.exists():
        shutil.rmtree(config.persist_dir)
    config.persist_dir.mkdir(parents=True, exist_ok=True)

    source_docs = load_source_documents(config.source_path)
    vectorstore, chunks, embedding_backend = build_vectorstore_from_documents(source_docs, config)
    write_chunk_cache(config, chunks)
    write_manifest(config, len(source_docs), len(chunks), embedding_backend)
    return vectorstore, source_docs, chunks, embedding_backend


def ensure_vectorstore(config: AppConfig) -> Chroma:
    if not config.rebuild and config.persist_dir.exists() and manifest_matches(config):
        try:
            vectorstore = load_vectorstore(config)
            vectorstore.similarity_search("health check", k=1)
            return vectorstore
        except Exception:
            pass
    vectorstore, _, _, _ = rebuild_index(config)
    return vectorstore


def build_chat_model(config: AppConfig) -> ChatMistralAI:
    return ChatMistralAI(
        model_name=config.chat_model,
        temperature=config.temperature,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def build_rag_chain(vectorstore: Chroma, config: AppConfig):
    source_docs_for_windows = load_source_documents(config.source_path)
    cached_chunks = load_cached_chunks(config)

    def analyze_question(question: str) -> dict[str, Any]:
        page_request = parse_page_window_request(question)
        if page_request:
            source_docs = source_docs_for_windows
            if source_docs:
                window = resolve_page_window(page_request, len(source_docs))
                if window:
                    selected_docs = select_page_documents(source_docs, window.start_page, window.end_page)
                    if selected_docs:
                        return {
                            "question": question,
                            "source_docs": selected_docs,
                            "scores": [1.0] * len(selected_docs),
                            "question_support": 1.0,
                            "max_score": 1.0,
                            "allow_summary": True,
                            "page_window": window,
                            "context": build_page_window_context(selected_docs),
                            "sources": serialize_page_window_sources(selected_docs),
                        }
        if cached_chunks:
            docs, scores, question_support = rank_cached_chunks(question, cached_chunks, config.top_k)
            if docs:
                return {
                    "question": question,
                    "source_docs": docs,
                    "scores": scores,
                    "question_support": question_support,
                    "max_score": max(scores) if scores else 0.0,
                    "allow_summary": is_overview_question(question),
                }
        return retrieve_with_scores(vectorstore, question, config.top_k)

    retrieval = RunnableLambda(analyze_question)
    prepared = retrieval | RunnablePassthrough.assign(
        context=lambda payload: payload.get("context")
        or format_context(payload["source_docs"], payload["scores"])
    )

    llm_chain = None
    summary_llm_chain = None
    try:
        chat_model = build_chat_model(config)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a verified-knowledge assistant. Use only the retrieved context. "
                    "If the context truly does not contain enough information, reply exactly: "
                    '"I don\'t know based on the provided document." '
                    "Keep the answer concise and cite every factual claim with page and line citations.",
                ),
                (
                    "human",
                    "Question: {question}\n\nRetrieved context:\n{context}\n\nAnswer:",
                ),
            ]
        )
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You summarize only the provided pages from a document. "
                    "Do not use outside knowledge. "
                    "Write a concise summary of the requested page window. "
                    "Cite factual claims using page citations or page ranges in square brackets.",
                ),
                (
                    "human",
                    "Question: {question}\n\nRequested pages: {page_label}\n\nContext:\n{context}\n\nSummary:",
                ),
            ]
        )
        llm_chain = prompt | chat_model | StrOutputParser()
        summary_llm_chain = summary_prompt | chat_model | StrOutputParser()
    except Exception:
        llm_chain = None
        summary_llm_chain = None

    def generate_answer(payload: dict[str, Any]) -> str:
        if not payload["source_docs"]:
            return REFUSAL_TEXT
        if payload.get("page_window"):
            window = payload["page_window"]
            try:
                if summary_llm_chain is not None:
                    answer = summary_llm_chain.invoke(
                        {
                            "question": payload["question"],
                            "page_label": page_window_label(window),
                            "context": payload.get("context") or build_page_window_context(payload["source_docs"]),
                        }
                    )
                    answer = answer.strip()
                    if answer:
                        if window.truncated:
                            return (
                                f"The document is shorter than the requested page window, so I summarized {page_window_label(window)} as available.\n\n"
                                f"{answer}"
                            )
                        return answer
            except Exception:
                pass
            return extractive_page_window_answer(payload["question"], payload["source_docs"], window)
        relevance = max(
            safe_float(payload.get("question_support"), 0.0),
            safe_float(payload.get("max_score"), 0.0),
        )
        if not payload.get("allow_summary") and relevance < config.support_threshold:
            return REFUSAL_TEXT
        if llm_chain is None:
            return extractive_answer_from_payload(payload)
        try:
            answer = llm_chain.invoke(payload)
            return answer.strip() or REFUSAL_TEXT
        except Exception:
            return extractive_answer_from_payload(payload)

    return prepared | RunnableParallel(
        question=RunnableLambda(lambda payload: payload["question"]),
        answer=RunnableLambda(generate_answer),
        sources=RunnableLambda(
            lambda payload: payload.get("sources")
            or serialize_sources(payload["source_docs"], payload["scores"])
        ),
        question_support=RunnableLambda(lambda payload: payload["question_support"]),
        max_score=RunnableLambda(lambda payload: payload["max_score"]),
    )


def load_or_build_chain(config: AppConfig):
    vectorstore = ensure_vectorstore(config)
    return build_rag_chain(vectorstore, config)


def run_single_question(chain, question: str) -> dict[str, Any]:
    result = chain.invoke(question)
    print(f"\nQ: {question}")
    print_query_result(result)
    return result


def run_interactive(chain) -> None:
    print("Interactive mode. Press Enter on an empty line to exit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        if question.lower() in {"exit", "quit", ":q"}:
            return
        try:
            run_single_question(chain, question)
        except Exception as exc:
            print(f"Error while answering question: {exc}")


async def evaluate_samples(
    chain, config: AppConfig, questions: list[str] | None = None
) -> dict[str, Any]:
    if not questions:
        return {
            "mode": "skipped",
            "samples": [],
            "average_faithfulness": 0.0,
            "average_answer_relevancy": 0.0,
        }

    mode = "ragas"
    faithfulness = None
    answer_relevancy = None
    try:
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        evaluator_llm = InstructorLLM(
            client=from_mistral(mistral_client, use_async=True),
            model=config.chat_model,
            provider="mistral",
        )
        evaluator_embeddings = MistralRagasEmbeddings(
            model=config.embedding_model,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        faithfulness = Faithfulness(llm=evaluator_llm)
        answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    except Exception:
        mode = "heuristic"

    rows: list[dict[str, Any]] = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []

    for question in questions:
        result = chain.invoke(question)
        answer = result["answer"]
        contexts = [source["snippet"] for source in result["sources"]]
        if mode == "ragas" and faithfulness and answer_relevancy:
            try:
                faith_score = metric_value(
                    await faithfulness.ascore(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=contexts,
                    )
                )
                relev_score = metric_value(
                    await answer_relevancy.ascore(
                        user_input=question,
                        response=answer,
                    )
                )
            except Exception:
                mode = "heuristic"
                faith_score, relev_score = heuristic_scores(question, answer, contexts)
        else:
            faith_score, relev_score = heuristic_scores(question, answer, contexts)

        faithfulness_scores.append(faith_score)
        relevancy_scores.append(relev_score)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "faithfulness": round(faith_score, 4),
                "answer_relevancy": round(relev_score, 4),
                "sources": result["sources"],
            }
        )

    return {
        "mode": mode,
        "samples": rows,
        "average_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if rows else 0.0,
        "average_answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 4) if rows else 0.0,
    }


def heuristic_scores(question: str, answer: str, contexts: list[str]) -> tuple[float, float]:
    question_tokens = content_tokens(question)
    answer_tokens = content_tokens(answer)
    context_tokens = set()
    for context in contexts:
        context_tokens.update(content_tokens(context))

    faithfulness = 1.0 if answer.strip() == REFUSAL_TEXT else 0.2
    if answer_tokens and answer_tokens <= context_tokens:
        faithfulness = max(faithfulness, 0.8)

    if not question_tokens:
        answer_relevancy = 0.0
    else:
        overlap = len(question_tokens & answer_tokens)
        answer_relevancy = overlap / len(question_tokens)

    return faithfulness, answer_relevancy


def print_evaluation(summary: dict[str, Any]) -> None:
    if summary.get("mode") == "skipped":
        print("\nEvaluation skipped.")
        return
    print("\nRAGAS evaluation")
    if summary.get("mode") != "ragas":
        print("RAGAS was unavailable here, so the command fell back to heuristic scoring.")
    print(f"Average faithfulness: {summary['average_faithfulness']:.4f}")
    print(f"Average answer relevancy: {summary['average_answer_relevancy']:.4f}")
    print("\nPer-sample results:")
    for sample in summary["samples"]:
        print(f"- Q: {sample['question']}")
        print(f"  Faithfulness: {sample['faithfulness']:.4f}")
        print(f"  Answer relevancy: {sample['answer_relevancy']:.4f}")


def prompt_for_evaluation(chain, config: AppConfig) -> None:
    raw = input("Enter evaluation questions separated by ||, or press Enter to skip: ").strip()
    if not raw:
        print("Skipped evaluation questions.")
        return
    questions = [item.strip() for item in raw.split("||") if item.strip()]
    if not questions:
        print("Skipped evaluation questions.")
        return
    summary = asyncio.run(evaluate_samples(chain, config, questions=questions))
    print_evaluation(summary)


def build_config(
    args: argparse.Namespace, source_path: Path, chunk_size: int, top_k: int
) -> AppConfig:
    chunk_overlap = normalize_chunk_overlap(chunk_size, args.chunk_overlap)
    persist_dir = RAG_STORE_ROOT / source_store_name(source_path, chunk_size)
    return AppConfig(
        source_path=source_path,
        persist_dir=persist_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        support_threshold=DEFAULT_SUPPORT_THRESHOLD,
        chat_model=args.chat_model or DEFAULT_CHAT_MODEL,
        embedding_model=args.embedding_model or DEFAULT_EMBEDDING_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        rebuild=args.rebuild,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the verified knowledge assistant from the terminal."
    )
    parser.add_argument("--pdf-dir", default=str(PDF_DIR), help="Folder containing source PDFs")
    parser.add_argument("--source", help="Specific source file to use")
    parser.add_argument("--query", help="Ask one question and exit")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override chunk size; otherwise you will be prompted after selecting a PDF",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Override chunk overlap; otherwise it is derived from the chosen chunk size",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override retrieval count; otherwise you will be prompted after chunk size",
    )
    parser.add_argument("--chat-model", help="Override chat model")
    parser.add_argument("--embedding-model", help="Override embedding model")
    parser.add_argument("--show-analysis", action="store_true", help="Print Part B analysis and exit")
    parser.add_argument("--show-requirements", action="store_true", help="Print requirements.txt and exit")
    args = parser.parse_args()

    if args.show_analysis:
        show_text_file(PART_B_ANALYSIS_PATH, "Part B Analysis")
        return 0

    if args.show_requirements:
        show_text_file(REQUIREMENTS_PATH, "Requirements")
        return 0

    pdf_dir = Path(args.pdf_dir).expanduser()
    source_path = resolve_source_path(args.source, pdf_dir)
    chunk_size = resolve_chunk_size(args.chunk_size)
    if chunk_size not in CHUNK_SIZE_OPTIONS and chunk_size > 3000:
        print("Chunk size cannot exceed 3000. Choose a smaller value or one of the listed options.")
        return 1
    if chunk_size <= 0:
        print("Chunk size must be a positive number.")
        return 1
    top_k = resolve_top_k(args.top_k)
    if top_k <= 0:
        print("K must be a positive number.")
        return 1
    config = build_config(args, source_path, chunk_size, top_k)

    print(f"\nSelected source: {source_path.name}")
    print(f"Selected chunk size: {config.chunk_size}")
    print(f"Chunk overlap: {config.chunk_overlap}")
    print(f"Selected K value: {config.top_k}")
    print(f"Index location: {config.persist_dir.resolve()}")
    print("Building or loading the index...")

    if not ensure_mistral_key():
        print("No Mistral API key found in .env. The script will fall back when needed.")

    try:
        vectorstore = ensure_vectorstore(config)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    chain = build_rag_chain(vectorstore, config)

    if args.query:
        run_single_question(chain, args.query)
        return 0

    print("Index ready.")
    print("Ask any question about the selected document. Press Enter on a blank line to finish.")
    run_interactive(chain)
    prompt_for_evaluation(chain, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
