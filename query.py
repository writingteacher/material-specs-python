import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "material-specs-python"
EMBED_MODEL      = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
TOP_K            = 5

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(INDEX_NAME)


def embed(text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def query(question):
    vector  = embed(question)
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    context_chunks = []
    sources        = []

    for match in results.matches:
        context_chunks.append(match.metadata.get("text", ""))
        title = match.metadata.get("title", "Unknown")
        if title not in sources:
            sources.append(title)

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = """You are a technical assistant for industrial material specifications.
Answer questions using only the provided context from manufacturer PDFs.
Be precise — include exact values, units, and product names where available.
If the answer is not in the context, say: "I don't have that information in the loaded specifications." """

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources: {', '.join(sources)}\n")


if __name__ == "__main__":
    print("=== Material Specs KB — Query CLI ===")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            query(question)