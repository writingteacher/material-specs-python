# Material Specs KB — Python RAG Pipeline

A retrieval-augmented generation (RAG) system that ingests manufacturer PDF datasheets from Google Drive, indexes them in a vector database, and serves precise technical answers through a web chat interface.

Built as a portfolio project to demonstrate a Python-based AI documentation pipeline. A companion no-code version of this system was built using Flowise.

**Live demo:** [material-specs-python.onrender.com](https://material-specs-python.onrender.com)

---

## What it does

Engineers and procurement teams can ask plain-language questions about industrial adhesives, sealants, and coatings — and get answers with exact values and source citations pulled directly from manufacturer PDFs.

**Example queries:**
- *What is the overlap shear strength of 3M 468MP?*
- *What surfaces is DOWSIL 732 suitable for?*
- *What is the service temperature range of Sikaflex Pro-3?*

If the answer isn't in the loaded specifications, the system says so. It does not guess.

---

## System architecture

```
Google Drive (33 PDFs)
        |
        v
  fetch.py — Google Drive API + PyMuPDF
        |
        v
  fetched_docs.json (raw extracted text)
        |
        v
  index_all.py — OpenAI text-embedding-3-small
        |
        v
  Pinecone (vector index — 83 vectors)
        |
        v
  app.py — Flask + GPT-4o-mini
        |
        v
  Chat UI (Render)
```

The pipeline runs in three stages: fetch, index, serve. Each stage is a separate script and can be run independently.

---

## Scripts

| Script | Purpose |
|---|---|
| `fetch.py` | Connects to Google Drive, downloads PDFs, extracts text, saves to `fetched_docs.json` |
| `index_all.py` | Chunks text, generates embeddings via OpenAI, upserts vectors to Pinecone |
| `query.py` | CLI tool for testing retrieval before deploying the web app |
| `app.py` | Flask web app — handles queries, retrieval, and the chat UI |

---

## Prerequisites

- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) account with a serverless index named `material-specs-python`
- An [OpenAI](https://platform.openai.com/) API key
- A Google Cloud service account with Drive API enabled and access to the target folder

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/writingteacher/material-specs-python.git
cd material-specs-python
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
GOOGLE_CREDENTIALS_FILE=credentials.json
DRIVE_FOLDER_ID=your-google-drive-folder-id
```

`DRIVE_FOLDER_ID` is the ID from the folder URL in Google Drive:
`https://drive.google.com/drive/folders/YOUR_FOLDER_ID_IS_HERE`

### 4. Add your Google service account credentials

Place your Google Cloud service account key file in the project root and name it `credentials.json`. The service account must have **Viewer** access to the target Drive folder.

---

## Running the pipeline

Run the scripts in order:

**Step 1 — Fetch PDFs from Google Drive**

```bash
python fetch.py
```

Outputs `fetched_docs.json` with extracted text from all PDFs found in the target folder and its subfolders.

**Step 2 — Index into Pinecone**

```bash
python index_all.py
```

Chunks the text, generates embeddings, and upserts vectors to Pinecone. Progress is logged to the terminal.

**Step 3 (optional) — Test retrieval from the CLI**

```bash
python query.py
```

Runs an interactive terminal session for testing queries against the index before deploying the web app.

**Step 4 — Run the web app**

```bash
python app.py
```

Starts the Flask app at `http://localhost:5000`.

---

## Deployment

This project is deployed on [Render](https://render.com) as a web service. The start command is:

```bash
gunicorn app:app
```

Set all environment variables from your `.env` file in the Render dashboard under **Environment**.

---

## Stack

| Layer | Technology |
|---|---|
| PDF ingestion | Google Drive API, PyMuPDF |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | Pinecone (serverless, AWS us-east-1) |
| LLM | GPT-4o-mini |
| Web framework | Flask |
| Deployment | Render |

---

## Related

[Case study: Material Specs KB — Python RAG Pipeline](https://rwhyte.com/case-study-material-specs-kb-python-rag-pipeline/)

[Material Specs KB — Flowise (no-code version)](https://rwhyte.com/case-study-ai-knowledge-assistant-material-specs/)