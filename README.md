# Verified Knowledge RAG Assignment

This project implements a terminal-based Retrieval-Augmented Generation (RAG) assistant for the class assignment. The assistant loads a PDF or text document, splits it into chunks, builds a local Chroma index, retrieves relevant passages, and answers user questions with page and line citations. It also includes a Part B reflection and a simple evaluation path.

## What It Does

- Lets the user choose a source document from the `pdfs/` folder.
- Lets the user choose a chunk size and `K` value before indexing.
- Builds or reuses a Chroma vector store for the selected source.
- Answers questions using only the selected document.
- Returns citations for the retrieved evidence.
- Refuses questions that are not supported by the document.
- Handles page-range questions like "summarize the last 40 pages" by summarizing the requested page window directly.

## Project Files

- `terminal_rag.py` - main program and only Python source file.
- `pdfs/` - folder that contains the documents you can index.
- `part_b_analysis.txt` - written reflection for Part B of the assignment.
- `requirements.txt` - Python dependencies for the project.
- `.env` - local environment settings, including your Mistral API key.

## Setup

1. Create and activate a virtual environment if you have not already.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Mistral API key to `.env`.

```env
MISTRAL_API_KEY=your_key_here
MISTRAL_CHAT_MODEL=mistral-large-latest
MISTRAL_EMBEDDING_MODEL=mistral-embed
```

## How To Run

Run the assistant from the project root:

```bash
./.venv/bin/python terminal_rag.py
```

At startup, the script will:

1. List the available documents in `pdfs/`.
2. Ask you to select one by number.
3. Ask you to choose a chunk size.
4. Ask you to choose `K`, which controls how many chunks are retrieved per question.
5. Build or load the index.
6. Prompt you for questions.

You can also pass a file directly:

```bash
./.venv/bin/python terminal_rag.py --source "pdfs/your_file.pdf"
```

## Useful Options

- `--chunk-size` sets the chunk size directly.
- `--top-k` sets the retrieval count directly.
- `--rebuild` forces the index to be rebuilt.
- `--show-analysis` prints the Part B writeup.
- `--show-requirements` prints the dependency list.

Example:

```bash
./.venv/bin/python terminal_rag.py --rebuild --chunk-size 300 --top-k 3
```

## Chunk Size and K

- `chunk_size` controls how large each text piece is before indexing.
- Smaller chunks are more precise.
- Larger chunks keep more context together and work better for broad summaries.
- `K` controls how many chunks are retrieved before the answer is generated.
- Smaller `K` is tighter and faster.
- Larger `K` gives the model more context for summary-style or broad questions.

## Demo Flow

For class, the clean demo is:

1. Select a PDF with more than 10 pages.
2. Pick a chunk size.
3. Pick a `K` value.
4. Ask one question that is clearly answered by the document.
5. Ask one question that is outside the document.
6. Show that the assistant cites the source and refuses unsupported questions.

## Part B Summary

The reflection file covers:

- Chunking trade-offs when `chunk_size` is changed from `3000` to `100`.
- The hallucination test for out-of-document questions.
- LCEL usage with `RunnableParallel` to return both the answer and source documents.

## Notes

- The project uses Mistral for generation and embeddings.
- If the Mistral API key is missing or invalid, the script falls back when possible so the app still runs.
- The `pdfs/` folder already contains sample documents, including a 256-page book that satisfies the assignment's page-count requirement.
