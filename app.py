import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ── Config ─────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "material-specs-python"
EMBED_MODEL      = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
TOP_K            = 5

# ── Clients ────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(INDEX_NAME)

SYSTEM_PROMPT = """You are a technical assistant for industrial material specifications.
You help engineers, procurement teams, and contractors find precise technical data from
manufacturer product data sheets.

Answer questions using ONLY the context provided. Be precise — include exact values,
units, and product names where available. If the answer is not in the context, say:
"I don't have that information in the loaded specifications."

Do not guess or infer values not present in the context."""

# ── HTML ───────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Material Specs KB</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f2f5;
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    .container {
      width: 100%;
      max-width: 720px;
      height: 90vh;
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .header {
      background: #1c3557;
      color: white;
      padding: 20px 24px;
      flex-shrink: 0;
    }

    .header h1 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
    .header p  { font-size: 13px; color: #8aaccc; }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .message.user {
      background: #1c3557;
      color: white;
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }

    .message.assistant {
      background: #f5f7fa;
      color: #1c3557;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
    }

    .sources {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #dde3ea;
      font-size: 12px;
      color: #6680a0;
    }

    .sources-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #8aaccc;
      margin-bottom: 6px;
    }

    .source-cards {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 4px;
    }

    .source-card {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 11px;
      color: #1c3557;
      font-weight: 500;
      white-space: nowrap;
    }

    .source-card .doc-icon {
      font-size: 11px;
      opacity: 0.6;
    }

    .input-area {
      padding: 16px 24px;
      border-top: 1px solid #eee;
      display: flex;
      gap: 10px;
      flex-shrink: 0;
    }

    .input-area input {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid #ddd;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }

    .input-area input:focus { border-color: #1c3557; }

    .input-area button {
      padding: 12px 20px;
      background: #1c3557;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
      font-weight: 600;
    }

    .input-area button:hover    { background: #254a75; }
    .input-area button:disabled { background: #aaa; cursor: not-allowed; }

    .welcome {
      text-align: center;
      color: #9999aa;
      font-size: 13px;
      margin: auto;
      padding: 40px 20px;
    }

    .welcome h2 { font-size: 16px; color: #555; margin-bottom: 8px; }

    .suggestions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 16px;
    }

    .suggestion {
      background: #f5f7fa;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      color: #1c3557;
      cursor: pointer;
      text-align: left;
    }

    .suggestion:hover { background: #e8edf3; }

    .typing { font-style: italic; color: #9999aa; font-size: 13px; }

    .footer {
      text-align: center;
      font-size: 11px;
      color: #bbbbcc;
      padding: 8px;
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Material Specs KB</h1>
      <p>Ask questions about adhesives, sealants, coatings, and specialty chemicals</p>
    </div>

    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <h2>What would you like to know?</h2>
        <p>Ask about technical specifications, application guidelines, or product comparisons.</p>
        <div class="suggestions">
          <div class="suggestion" onclick="askSuggestion(this)">What are the specifications for 3M Adhesive Transfer Tape 468MP?</div>
          <div class="suggestion" onclick="askSuggestion(this)">What is the service temperature range of Sikaflex Pro-3?</div>
          <div class="suggestion" onclick="askSuggestion(this)">What is the cure time for Loctite AA 330?</div>
          <div class="suggestion" onclick="askSuggestion(this)">What surfaces is DOWSIL 732 suitable for?</div>
        </div>
      </div>
    </div>

    <div class="input-area">
      <input type="text" id="input" placeholder="Ask about a product or specification..." onkeydown="if(event.key==='Enter') sendMessage()">
      <button id="send-btn" onclick="sendMessage()">Ask</button>
    </div>

    <div class="footer">Built by <a href="https://rwhyte.com" style="color:#bbbbcc;">rwhyte.com</a> · RAG pipeline on 33 manufacturer PDFs · OpenAI + Pinecone</div>
  </div>

  <script>
    function askSuggestion(el) {
      document.getElementById('input').value = el.textContent;
      sendMessage();
    }

    function addMessage(text, role, sources) {
      const welcome = document.getElementById('welcome');
      if (welcome) welcome.remove();

      const msgs = document.getElementById('messages');
      const div  = document.createElement('div');
      div.className = `message ${role}`;
      div.textContent = text;

      if (sources && sources.length > 0) {
        const src = document.createElement('div');
        src.className = 'sources';

        const label = document.createElement('div');
        label.className = 'sources-label';
        label.textContent = 'Sources';
        src.appendChild(label);

        const cards = document.createElement('div');
        cards.className = 'source-cards';

        sources.forEach(name => {
          const card = document.createElement('div');
          card.className = 'source-card';
          // Clean up filename — remove underscores/hyphens, trim extensions
          const clean = name.replace(/[-_]/g, ' ').replace(/\.[^.]+$/, '');
          card.innerHTML = `<span class="doc-icon">📄</span>${clean}`;
          cards.appendChild(card);
        });

        src.appendChild(cards);
        div.appendChild(src);
      }

      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const input = document.getElementById('input');
      const btn   = document.getElementById('send-btn');
      const q     = input.value.trim();
      if (!q) return;

      input.value  = '';
      btn.disabled = true;

      addMessage(q, 'user');

      const typing = addMessage('Thinking...', 'assistant');
      typing.classList.add('typing');

      try {
        const res  = await fetch('/ask', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ question: q })
        });
        const data = await res.json();
        typing.remove();
        addMessage(data.answer, 'assistant', data.sources);
      } catch(e) {
        typing.textContent = 'Something went wrong. Please try again.';
        typing.classList.remove('typing');
      }

      btn.disabled = false;
      input.focus();
    }
  </script>
</body>
</html>
"""

app = Flask(__name__)


def expand_query(question):
    """Rewrite the user's question to fix typos and expand synonyms for better retrieval."""
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=100,
        messages=[
            {"role": "system", "content": (
                "You are a search query optimizer for industrial material specifications. "
                "Rewrite the user's question to fix any spelling errors and expand key terms "
                "with technical synonyms (e.g. 'cold' → 'low temperature, sub-zero, cold'; "
                "'sticky' → 'adhesion, bonding, tackiness'). "
                "Return only the rewritten query — no explanation, no punctuation changes."
            )},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()


def embed(text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a question.", "sources": []})

    expanded = expand_query(question)
    vector  = embed(expanded)
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    context_parts = []
    sources       = []

    for match in results.matches:
        meta  = match.metadata
        title = meta.get("title", "Unknown")
        context_parts.append(f"[{title}]\n{meta.get('text', '')}")
        if title not in sources:
            sources.append(title)

    context = "\n\n---\n\n".join(context_parts)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)