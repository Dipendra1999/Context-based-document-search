# Context-based document search

Semantic search over plain-text documents using [LangChain](https://python.langchain.com/), [Chroma](https://www.trychroma.com/) for vector storage, and OpenAI embeddings. The sample app indexes résumés in `resumes/` and retrieves the most relevant passages for a natural-language query.

## How it works

1. **Ingest**: All `.txt` files under `resumes/` are loaded and embedded with OpenAI `text-embedding-3-large` (512 dimensions).
2. **Store**: Embeddings are persisted under `vector_db/` so the next run can load the index without re-embedding.
3. **Search**: A query string is compared to stored chunks via similarity search; the top matches are returned.

Core logic lives in `Utils/vector_db_handler.py`. `app.py` wires paths, loads or creates the store, runs one example query, and prints the best match.

## Requirements

- Python 3.10+ recommended
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to the embeddings model

## Setup

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

## Run

From the project root (with the virtual environment activated):

```bash
python app.py
```

On first run, the app builds the vector index from `resumes/` and writes it to `vector_db/`. Later runs reuse that folder if it already exists.

To change the search question, edit the `query` string in `app.py`.

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Entry point: paths, load/create DB, example query |
| `Utils/vector_db_handler.py` | Chroma + OpenAI embeddings, ingest and similarity search |
| `resumes/` | Sample `.txt` documents to index |
| `vector_db/` | Persisted Chroma database (created at runtime) |

## Customizing

- **Documents**: Add or replace `.txt` files in `resumes/` (or change `files_directory` in `app.py`). To force a full re-index, delete the `vector_db/` directory and run again.
- **Collection / paths**: Adjust `files_directory`, `persist_directory`, and `collection_name` in `app.py`.
- **Top-k results**: `query_vector_store` accepts `top_k` (default `5`); `app.py` currently only prints the first result.

## License

Add a license file if you plan to distribute the project publicly.
