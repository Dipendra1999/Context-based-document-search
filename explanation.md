# In-depth code explanation

This document walks through every part of the **Context-based document search** project: what each file does, how the pieces connect, and the important behaviors and trade-offs hidden in a short codebase.

---

## 1. Big picture

The application answers: *“Which stored text is most similar in meaning to this natural-language question?”*

It does **not** do keyword matching. It converts text into **vectors** (lists of numbers that capture semantic similarity) using an OpenAI embedding model, stores those vectors in **Chroma**, and at query time embeds the question the same way and retrieves the vectors (and their text) that sit closest to the query vector in that space.

**End-to-end flow:**

1. `app.py` starts the program, loads secrets from the environment, and constructs a `VectorDBHandler`.
2. `VectorDBHandler.load_or_create_db()` either opens an existing Chroma database on disk or reads every `.txt` under `resumes/`, embeds them, and writes a new database under `vector_db/`.
3. `query_vector_store()` embeds the query string and runs a **similarity search** to get the top `k` `Document` objects.
4. `app.py` prints the **first** result’s full `page_content` (the raw text that was indexed for that item).

There is no web server, CLI arguments, or interactive loop in the current code; everything is driven by the constants and the single `query` string in `app.py`.

---

## 2. Dependencies (`requirements.txt`)

| Package | Role in *this* repo |
|--------|----------------------|
| `python-dotenv` | Loads `.env` so `OPENAI_API_KEY` is available without hard-coding. |
| `langchain-openai` | `OpenAIEmbeddings` for calling OpenAI’s embedding API. |
| `langchain-chroma` | `Chroma` vector store integration. |
| `langchain` / `langchain-core` (pulled in transitively) | `Document` type and shared abstractions. |

Packages such as `pypdf`, `pandas`, `scipy`, `langchain_ollama`, and `langchain-experimental` are listed but **not used** by `app.py` or `Utils/vector_db_handler.py`. They may be leftovers for future features (e.g. PDF ingestion or local models).

---

## 3. Entry point: `app.py`

### 3.1 Environment loading

```python
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` reads a `.env` file from the current working directory (and parents, per python-dotenv rules) and merges variables into `os.environ`. The OpenAI client used inside `OpenAIEmbeddings` picks up `OPENAI_API_KEY` from the environment. If the key is missing, embedding calls will fail at runtime when the API is invoked.

### 3.2 Configuration as module-level constants

```python
files_directory = "./resumes"
persist_directory = "./vector_db"
collection_name = "resumes_collection"
```

These are **relative paths** from the process **current working directory**, not necessarily the directory where `app.py` lives. Running `python app.py` from the project root is the intended layout so `./resumes` and `./vector_db` resolve correctly.

- **`files_directory`**: Where plain-text `.txt` sources are read during **create** (first-time index build).
- **`persist_directory`**: Where Chroma persists its SQLite (and related) files for the vector index.
- **`collection_name`**: Logical name of the collection inside Chroma. Loading and writing must use the **same** name so you reopen the same index.

### 3.3 Handler construction and index lifecycle

```python
vector_db_handler = VectorDBHandler(files_directory, persist_directory, collection_name)
vector_db_handler.load_or_create_db()
```

Construction only stores paths and builds an **`OpenAIEmbeddings`** instance; it does **not** touch disk for Chroma yet. The actual “open DB or ingest files” step is `load_or_create_db()`.

### 3.4 Query and output

```python
query = "I am looking for a person with communication skills. who knows kubernetes."
docs = vector_db_handler.query_vector_store(query)
```

`query_vector_store` defaults to `top_k=5`, so `docs` is a list of up to five `Document` objects, ordered by relevance as defined by Chroma’s similarity search. The script only **prints** `docs[0].page_content`—the full text of the single best-matching stored document **as one chunk** (see chunking note below).

The `try` / `except ValueError` only catches the explicit `ValueError` raised if `query_vector_store` is called when `self.vector_store` was never set (defensive; normal flow always calls `load_or_create_db()` first). API errors from OpenAI would surface as different exception types and are not caught here.

---

## 4. Core logic: `Utils/vector_db_handler.py`

### 4.1 Imports and embedding model

```python
from langchain_openai import OpenAIEmbeddings
self.embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=512,
)
```

**`text-embedding-3-large`** is a high-quality OpenAI embedding model. The **`dimensions=512`** argument requests a **reduced** embedding size (512 floats per text) instead of the model’s default maximum dimension. That:

- Lowers storage and index size in Chroma.
- Can slightly change retrieval quality versus full dimension; for many RAG/search setups 512 is still strong.

Every text passed to this embedding model—both documents at index time and the query at search time—must go through the **same** model and settings so vectors live in a comparable space.

### 4.2 `check_if_db_exists`

```python
return os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory)
```

This only checks that **`persist_directory` exists and is a directory**. It does **not** verify that Chroma was successfully populated or that the collection exists. Edge case: if `vector_db` was created as an empty folder, the code takes the “load existing” branch; behavior then depends on Chroma (may error or show an empty store). In normal use, the folder appears when Chroma first persists data after `create_vector_embedding`.

### 4.3 `create_vector_embedding` — ingestion

**Iterate `.txt` files:**

```python
for file_name in os.listdir(self.directory_path):
    if file_name.endswith(".txt"):
```

Only files ending in `.txt` are included; subdirectories are not walked. Order is filesystem-dependent unless sorted.

**Read whole file as one document:**

```python
content = file.read()
metadata = {"file_path": file_path}
documents.append(Document(page_content=content, metadata=metadata))
```

Each file becomes **one** LangChain `Document`:

- **`page_content`**: The entire file string. There is **no chunking** in this project. Very long files become one very long “document”; Chroma/LangChain may still split internally depending on version defaults, but the code’s intent is one logical doc per file.
- **`metadata`**: Only `file_path` is stored, useful for tracing which file a hit came from (not printed in `app.py`).

**Instantiate Chroma and add documents:**

```python
self.vector_store = Chroma(
    collection_name=self.collection_name,
    embedding_function=self.embedding_model,
    persist_directory=self.persist_directory
)
uuids = [str(uuid4()) for _ in range(len(documents))]
self.vector_store.add_documents(documents, ids=uuids)
```

- **`persist_directory`**: Data is written to disk so a later process can reopen it.
- **`ids=uuids`**: Each chunk/document gets a **stable random UUID** as its id in the store. That avoids collisions and lets you update/delete by id in more advanced workflows. Re-running ingestion after delete of `vector_db` generates **new** ids.

During `add_documents`, LangChain invokes the embedding model for each document’s text (batching may occur inside the library), so the first run triggers **N API calls** for N files (modulo batching).

### 4.4 `load_or_create_db` — branch behavior

**If directory exists:**

```python
self.vector_store = Chroma(
    collection_name=self.collection_name,
    embedding_function=self.embedding_model,
    persist_directory=self.persist_directory
)
```

Chroma loads existing persisted data. The **same** `embedding_function` must be used for **new** embeddings to stay consistent; for **query** embedding, LangChain uses this same function to embed the query string.

**If directory does not exist:**

`create_vector_embedding()` runs: reads all `.txt`, builds Chroma, embeds, persists.

**Important:** If you add new `.txt` files to `resumes/` but keep `vector_db/`, the program **will not** re-ingest them automatically; it will keep using the old index. To refresh, delete `vector_db/` (or change `persist_directory` / `collection_name`) and run again.

### 4.5 `query_vector_store`

```python
if self.vector_store is None:
    raise ValueError("Vector store not initialized. ...")
results = self.vector_store.similarity_search(query=query_text, k=top_k)
return results
```

**`similarity_search`**: Embeds `query_text` with `self.embedding_model`, compares to stored vectors (typically cosine similarity in the embedding space; exact metric is defined by Chroma + LangChain integration), returns the top **`k`** `Document` instances.

Default **`top_k=5`**: `app.py` only displays the first; the other four are computed but discarded from the user’s perspective—you could extend `app.py` to list all five or show `metadata["file_path"]` for each.

---

## 5. Data model: LangChain `Document`

A `Document` is a small struct:

- **`page_content`**: The searchable text.
- **`metadata`**: Arbitrary key-value pairs (here, `file_path`).

Similarity operates on the embedded representation of `page_content`; metadata is not embedded unless you explicitly build that into another pipeline.

---

## 6. Runtime and security notes

- **API key**: Must be present for any path that calls OpenAI (create and query). Keep `.env` out of version control.
- **Cost**: First-time indexing charges per token embedded; each query embeds the query string once (plus any internal retries).
- **Offline / privacy**: This stack is cloud-embodied for embeddings unless you swap `OpenAIEmbeddings` for a local model (your `requirements.txt` hints at Ollama for future use).

---

## 7. What this project does *not* do (yet)

Understanding limits helps when extending the code:

| Topic | Current behavior |
|--------|------------------|
| **Chunking** | One `Document` per file; no sentence/paragraph splits. |
| **PDFs / Word** | Not ingested despite `pypdf` in requirements. |
| **Hybrid search** | Vector only; no BM25/keyword blend. |
| **Re-ranking** | Raw top-k from similarity only. |
| **Filtering** | No metadata filters on query (e.g. by folder or tags). |
| **CLI / API** | Single hardcoded query in `app.py`. |

---

## 8. Mental model diagram

```text
  resumes/*.txt
        │
        ▼
  read as Document(page_content, metadata)
        │
        ▼
  OpenAIEmbeddings ──► vectors ──► Chroma (vector_db/)
        ▲
        │
  query string ──► same embeddings ──► similarity_search ──► List[Document]
```

---

## 9. File reference summary

| File | Responsibility |
|------|----------------|
| `app.py` | Load env, configure paths, run load/create, run one query, print top match. |
| `Utils/vector_db_handler.py` | Encapsulate Chroma lifecycle, file ingestion, embedding config, similarity search. |
| `resumes/` | Source `.txt` files for the demo corpus. |
| `vector_db/` | Generated persistence; safe to delete to force rebuild. |

This is the full executable surface area of the project as it exists today; deeper behavior (exact distance metric, batching, internal chunking) is defined by the installed versions of LangChain and Chroma, which you can confirm in their docs or source for your pinned versions.
