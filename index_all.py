import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ── Config ─────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "material-specs-python"
EMBED_MODEL      = "text-embedding-3-small"
CHUNK_SIZE       = 1000  # words — larger to preserve spec tables
CHUNK_OVERLAP    = 200   # words
INPUT_FILE       = "fetched_docs.json"

# ── Clients ────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(INDEX_NAME)


def chunk_text(text, title, file_id):
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    i      = 0

    while start < len(words):
        end        = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id":      f"{file_id}__chunk{i}",
            "text":    chunk_text,
            "title":   title,
            "file_id": file_id,
        })
        i     += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed(text):
    """Embed a single text string using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def index_all():
    print("=== Material Specs KB — Indexer ===\n")

    with open(INPUT_FILE, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents from {INPUT_FILE}\n")

    all_chunks  = []
    total_words = 0

    for doc in docs:
        chunks      = chunk_text(doc["text"], doc["title"], doc["file_id"])
        all_chunks += chunks
        total_words += len(doc["text"].split())
        print(f"  {doc['title']} → {len(chunks)} chunks")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print(f"Total words: {total_words:,}")
    print(f"\nStarting embedding...\n")

    BATCH_SIZE = 50
    upserted   = 0

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]

        vectors = []
        for chunk in batch:
            vector = embed(chunk["text"])
            vectors.append({
                "id":     chunk["id"],
                "values": vector,
                "metadata": {
                    "title":   chunk["title"],
                    "file_id": chunk["file_id"],
                    "text":    chunk["text"][:3000]
                }
            })
            time.sleep(0.1)

        index.upsert(vectors=vectors)
        upserted += len(vectors)
        print(f"  Upserted batch {i//BATCH_SIZE + 1} — {upserted}/{len(all_chunks)} vectors")

    print(f"\n=== Done ===")
    print(f"Total vectors upserted: {upserted}")
    print(f"Index: {INDEX_NAME}")


if __name__ == "__main__":
    index_all()