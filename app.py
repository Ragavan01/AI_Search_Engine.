"""
╔══════════════════════════════════════════════════════════╗
║              VIRGO AI ENGINE - app.py                   ║
║   Multi-model AI Search Engine with Streaming, RAG,     ║
║   Web Search, Image Analysis & Chat History             ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import time
import sqlite3
import base64
import mimetypes
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask, request, jsonify, Response, render_template_string,
    session, stream_with_context, send_from_directory
)
from flask_cors import CORS
from dotenv import load_dotenv

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv()

# ─── Optional imports (graceful degradation) ─────────────────────────────────
try:
    import openai
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))
    if ANTHROPIC_AVAILABLE:
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
    if GEMINI_AVAILABLE:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32).hex())
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

DB_PATH = "virgo.db"

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                model TEXT NOT NULL DEFAULT 'gpt-4o',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT PRIMARY KEY,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
            CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);
        """)

init_db()

# ─── Model Definitions ────────────────────────────────────────────────────────
MODELS = {
    # OpenAI
    "gpt-4o":             {"provider": "openai", "label": "GPT-4o",            "icon": "⬡", "tier": "flagship"},
    "gpt-4o-mini":        {"provider": "openai", "label": "GPT-4o Mini",        "icon": "⬡", "tier": "fast"},
    "gpt-4-turbo":        {"provider": "openai", "label": "GPT-4 Turbo",        "icon": "⬡", "tier": "pro"},
    "o1-preview":         {"provider": "openai", "label": "o1 Preview",         "icon": "⬡", "tier": "reasoning"},
    "o1-mini":            {"provider": "openai", "label": "o1 Mini",            "icon": "⬡", "tier": "reasoning"},
    # Anthropic
    "claude-opus-4-5":    {"provider": "anthropic", "label": "Claude Opus 4",   "icon": "◈", "tier": "flagship"},
    "claude-sonnet-4-5":  {"provider": "anthropic", "label": "Claude Sonnet 4", "icon": "◈", "tier": "pro"},
    "claude-haiku-3-5":   {"provider": "anthropic", "label": "Claude Haiku 3",  "icon": "◈", "tier": "fast"},
    # Google
    "gemini-1.5-pro":     {"provider": "gemini",    "label": "Gemini 1.5 Pro",  "icon": "◆", "tier": "flagship"},
    "gemini-1.5-flash":   {"provider": "gemini",    "label": "Gemini 1.5 Flash","icon": "◆", "tier": "fast"},
    "gemini-2.0-flash":   {"provider": "gemini",    "label": "Gemini 2.0 Flash","icon": "◆", "tier": "flagship"},
}

# ─── Web Search ───────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    if not SEARCH_AVAILABLE:
        return []
    try:
        with get_db() as db:
            cached = db.execute(
                "SELECT results FROM search_cache WHERE query=? AND created_at > datetime('now','-1 hour')",
                (query,)
            ).fetchone()
            if cached:
                return json.loads(cached["results"])

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })

        with get_db() as db:
            db.execute(
                "INSERT OR REPLACE INTO search_cache VALUES (?,?,CURRENT_TIMESTAMP)",
                (query, json.dumps(results))
            )
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def format_search_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["## Web Search Results\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"**[{i}] {r['title']}**")
        lines.append(f"{r['snippet']}")
        lines.append(f"Source: {r['url']}\n")
    return "\n".join(lines)

# ─── Token Counter ────────────────────────────────────────────────────────────
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 4  # Rough estimate

# ─── Streaming Generators ──────────────────────────────────────────────────────
def stream_openai(messages: list, model: str, system: str = None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    is_reasoning = model.startswith("o1")
    kwargs = dict(model=model, messages=msgs, stream=True)
    if not is_reasoning:
        kwargs["temperature"] = 0.7
        kwargs["max_tokens"] = 4096

    try:
        stream = openai_client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n[OpenAI Error: {str(e)}]"

def stream_anthropic(messages: list, model: str, system: str = None):
    try:
        # Fix model name mapping
        model_map = {
            "claude-opus-4-5": "claude-opus-4-5-20251101",
            "claude-sonnet-4-5": "claude-sonnet-4-5-20251022",
            "claude-haiku-3-5": "claude-haiku-3-5-20241022",
        }
        api_model = model_map.get(model, model)
        kwargs = dict(
            model=api_model,
            max_tokens=4096,
            messages=messages,
            stream=True,
        )
        if system:
            kwargs["system"] = system

        with anthropic_client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n[Anthropic Error: {str(e)}]"

def stream_gemini(messages: list, model: str, system: str = None):
    try:
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system or "You are Virgo AI, a powerful search engine assistant.",
        )
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            history.append({"role": role, "parts": [content]})

        chat = gemini_model.start_chat(history=history)
        last = messages[-1]["content"]
        if isinstance(last, list):
            last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))

        response = chat.send_message(last, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n[Gemini Error: {str(e)}]"

# ─── Smart Router ─────────────────────────────────────────────────────────────
def get_stream_generator(messages: list, model: str, system: str = None):
    info = MODELS.get(model, {})
    provider = info.get("provider", "openai")

    if provider == "openai" and OPENAI_AVAILABLE:
        return stream_openai(messages, model, system)
    elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
        return stream_anthropic(messages, model, system)
    elif provider == "gemini" and GEMINI_AVAILABLE:
        return stream_gemini(messages, model, system)
    else:
        def fallback():
            yield f"⚠️ Provider for model **{model}** is not configured. Please set the API key in your `.env` file."
        return fallback()

# ─── System Prompt ────────────────────────────────────────────────────────────
VIRGO_SYSTEM = """You are **Virgo AI**, a next-generation intelligent search engine assistant — powerful, precise, and beautifully articulate.

## Your Core Capabilities:
- **Deep Research**: Synthesize information from multiple perspectives
- **Web Search**: Access real-time search results when provided
- **Code Generation**: Write clean, production-ready code in any language
- **Analysis**: Examine documents, data, and images with expert-level insight
- **Reasoning**: Tackle complex multi-step problems with clarity
- **Creativity**: Generate content, ideas, and narratives with flair

## Formatting:
- Use rich Markdown: headers, code blocks, tables, bold/italic
- Structure long answers clearly with sections
- For code, always specify the language in fenced blocks
- Be concise yet comprehensive — quality over verbosity
- Cite search sources when using web results

## Tone:
Intelligent, confident, direct. Never sycophantic. Always accurate."""

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_UI)

@app.route("/api/models")
def get_models():
    available = {}
    for mid, info in MODELS.items():
        p = info["provider"]
        avail = (
            (p == "openai" and OPENAI_AVAILABLE) or
            (p == "anthropic" and ANTHROPIC_AVAILABLE) or
            (p == "gemini" and GEMINI_AVAILABLE)
        )
        available[mid] = {**info, "available": avail}
    return jsonify(available)

@app.route("/api/status")
def status():
    return jsonify({
        "openai": OPENAI_AVAILABLE,
        "anthropic": ANTHROPIC_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "search": SEARCH_AVAILABLE,
        "image": PIL_AVAILABLE,
        "version": "2.0.0",
        "name": "Virgo AI Engine",
    })

# ─── Chat CRUD ────────────────────────────────────────────────────────────────
@app.route("/api/chats", methods=["GET"])
def list_chats():
    with get_db() as db:
        rows = db.execute(
            "SELECT id, title, model, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/chats", methods=["POST"])
def create_chat():
    data = request.json or {}
    chat_id = str(uuid.uuid4())
    model = data.get("model", "gpt-4o")
    title = data.get("title", "New Chat")
    with get_db() as db:
        db.execute(
            "INSERT INTO chats (id, title, model) VALUES (?,?,?)",
            (chat_id, title, model)
        )
    return jsonify({"id": chat_id, "title": title, "model": model})

@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    with get_db() as db:
        chat = db.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        msgs = db.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC",
            (chat_id,)
        ).fetchall()
    return jsonify({
        "chat": dict(chat),
        "messages": [dict(m) for m in msgs]
    })

@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    with get_db() as db:
        db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return jsonify({"ok": True})

@app.route("/api/chats/<chat_id>/title", methods=["PATCH"])
def update_title(chat_id):
    data = request.json or {}
    title = data.get("title", "Untitled")
    with get_db() as db:
        db.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    return jsonify({"ok": True})

# ─── Upload Handler ───────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    fname = f"{uuid.uuid4()}_{f.filename}"
    fpath = UPLOAD_FOLDER / fname
    f.save(fpath)

    mime = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
    size = fpath.stat().st_size

    result = {"filename": fname, "original": f.filename, "mime": mime, "size": size}

    if mime.startswith("image/") and PIL_AVAILABLE:
        with Image.open(fpath) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode

    return jsonify(result)

# ─── Main Chat Stream ─────────────────────────────────────────────────────────
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json or {}
    chat_id = data.get("chat_id")
    user_message = data.get("message", "").strip()
    model = data.get("model", "gpt-4o")
    use_search = data.get("web_search", False)
    file_info = data.get("file", None)
    history = data.get("history", [])

    if not user_message and not file_info:
        return jsonify({"error": "Empty message"}), 400

    # ── Auto-create chat if needed ──────────────────────────────────────────
    if not chat_id:
        chat_id = str(uuid.uuid4())
        title = (user_message[:40] + "...") if len(user_message) > 40 else user_message
        with get_db() as db:
            db.execute("INSERT INTO chats (id, title, model) VALUES (?,?,?)", (chat_id, title, model))

    # ── Save user message ───────────────────────────────────────────────────
    msg_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            "INSERT INTO messages (id, chat_id, role, content) VALUES (?,?,?,?)",
            (msg_id, chat_id, "user", user_message)
        )
        db.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP, model=? WHERE id=?", (model, chat_id))

    # ── Web Search ──────────────────────────────────────────────────────────
    search_results = []
    search_context = ""
    if use_search and SEARCH_AVAILABLE:
        search_results = web_search(user_message)
        search_context = format_search_context(search_results)

    # ── Build messages for API ──────────────────────────────────────────────
    api_messages = []
    for h in history[-20:]:  # last 20 turns
        role = h.get("role", "user")
        content = h.get("content", "")
        api_messages.append({"role": role, "content": content})

    # Build final user content (with optional image)
    final_content = user_message
    if search_context:
        final_content = f"{search_context}\n\n---\n\n**User Question:** {user_message}"

    if file_info and file_info.get("mime", "").startswith("image/"):
        fpath = UPLOAD_FOLDER / file_info["filename"]
        if fpath.exists():
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = file_info["mime"]
            provider = MODELS.get(model, {}).get("provider", "openai")
            if provider == "openai":
                api_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": final_content}
                    ]
                })
            elif provider == "anthropic":
                api_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": final_content}
                    ]
                })
            else:
                api_messages.append({"role": "user", "content": final_content})
        else:
            api_messages.append({"role": "user", "content": final_content})
    else:
        api_messages.append({"role": "user", "content": final_content})

    # ── Stream Response ─────────────────────────────────────────────────────
    def generate():
        full_response = []
        assistant_id = str(uuid.uuid4())

        # Send metadata first
        yield f"data: {json.dumps({'type': 'meta', 'chat_id': chat_id, 'model': model, 'search': bool(search_results)})}\n\n"

        if search_results:
            yield f"data: {json.dumps({'type': 'search_results', 'results': search_results[:3]})}\n\n"

        try:
            gen = get_stream_generator(api_messages, model, VIRGO_SYSTEM)
            for chunk in gen:
                full_response.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            full_text = "".join(full_response)

            # Save assistant message
            with get_db() as db:
                db.execute(
                    "INSERT INTO messages (id, chat_id, role, content) VALUES (?,?,?,?)",
                    (assistant_id, chat_id, "assistant", full_text)
                )
                db.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (chat_id,))

            tokens = count_tokens(full_text)
            yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'msg_id': assistant_id})}\n\n"

        except Exception as e:
            err = str(e)
            yield f"data: {json.dumps({'type': 'error', 'message': err})}\n\n"
            traceback.print_exc()

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ─── Auto-Title Generation ────────────────────────────────────────────────────
@app.route("/api/chats/<chat_id>/autotitle", methods=["POST"])
def auto_title(chat_id):
    with get_db() as db:
        msgs = db.execute(
            "SELECT content FROM messages WHERE chat_id=? AND role='user' LIMIT 3",
            (chat_id,)
        ).fetchall()
    if not msgs:
        return jsonify({"title": "New Chat"})

    sample = " ".join(m["content"][:200] for m in msgs)
    prompt = f"Generate a short 3-6 word title for a conversation that starts with: {sample[:300]}\nReturn only the title, no quotes."

    title = "New Chat"
    try:
        if OPENAI_AVAILABLE:
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.7,
            )
            title = r.choices[0].message.content.strip().strip('"\'')
        elif ANTHROPIC_AVAILABLE:
            r = anthropic_client.messages.create(
                model="claude-haiku-3-5-20241022",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            title = r.content[0].text.strip().strip('"\'')
    except Exception:
        title = sample[:40] + ("..." if len(sample) > 40 else "")

    with get_db() as db:
        db.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    return jsonify({"title": title})

# ─── Suggest prompts ─────────────────────────────────────────────────────────
@app.route("/api/suggestions")
def get_suggestions():
    suggestions = [
        {"icon": "🔍", "text": "Search and summarize recent AI breakthroughs"},
        {"icon": "💻", "text": "Write a full-stack React + FastAPI app"},
        {"icon": "📊", "text": "Analyze this data and create visualizations"},
        {"icon": "🧬", "text": "Explain CRISPR gene editing in simple terms"},
        {"icon": "📝", "text": "Write a professional email to decline a meeting"},
        {"icon": "🌐", "text": "What are the latest developments in quantum computing?"},
        {"icon": "🎯", "text": "Create a 30-day learning plan for machine learning"},
        {"icon": "🔐", "text": "Explain zero-knowledge proofs with examples"},
    ]
    return jsonify(suggestions)

# ─── HTML UI (Embedded) ───────────────────────────────────────────────────────
HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Virgo AI Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
/* ═══════════════════════════════════════════════
   VIRGO AI ENGINE — Dark Luxury Theme
   ═══════════════════════════════════════════════ */
:root {
  --bg-0: #070810;
  --bg-1: #0d0f1a;
  --bg-2: #12141f;
  --bg-3: #1a1d2e;
  --bg-4: #22263a;
  --border: rgba(255,255,255,0.07);
  --border-bright: rgba(255,255,255,0.14);
  --text-1: #f0f2ff;
  --text-2: #9ba3c4;
  --text-3: #5a6282;
  --accent: #7b6ef6;
  --accent-2: #4fc3f7;
  --accent-glow: rgba(123,110,246,0.25);
  --danger: #ff5370;
  --success: #4caf7d;
  --gold: #f7c948;
  --font-body: 'Outfit', sans-serif;
  --font-mono: 'Space Mono', monospace;
  --radius: 12px;
  --sidebar-w: 260px;
  --transition: 0.2s cubic-bezier(0.4,0,0.2,1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { height: 100%; overflow: hidden; }

body {
  font-family: var(--font-body);
  background: var(--bg-0);
  color: var(--text-1);
  display: flex;
  font-size: 15px;
  line-height: 1.6;
}

/* Noise texture overlay */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
  opacity: 0.4;
}

/* ─── Scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-4); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ─── Sidebar ────────────────────────────────────── */
#sidebar {
  width: var(--sidebar-w);
  min-width: var(--sidebar-w);
  height: 100vh;
  background: var(--bg-1);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: transform var(--transition);
  z-index: 100;
  position: relative;
  overflow: hidden;
}

#sidebar::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 200px;
  background: radial-gradient(ellipse at 50% 0%, rgba(123,110,246,0.08) 0%, transparent 70%);
  pointer-events: none;
}

.sidebar-header {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}

.logo-icon {
  width: 34px;
  height: 34px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  box-shadow: 0 4px 16px var(--accent-glow);
}

.logo-text {
  font-size: 17px;
  font-weight: 700;
  background: linear-gradient(90deg, var(--text-1), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.02em;
}

.logo-sub {
  font-size: 10px;
  color: var(--text-3);
  font-family: var(--font-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  -webkit-text-fill-color: var(--text-3);
}

.new-chat-btn {
  width: 100%;
  padding: 10px 14px;
  background: linear-gradient(135deg, var(--accent), #6356d8);
  color: white;
  border: none;
  border-radius: 8px;
  font-family: var(--font-body);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all var(--transition);
  box-shadow: 0 4px 16px var(--accent-glow);
}

.new-chat-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px var(--accent-glow);
}

.new-chat-btn svg { width: 16px; height: 16px; }

/* Chat list */
.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 8px;
}

.chat-section-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-3);
  padding: 12px 8px 6px;
  font-family: var(--font-mono);
}

.chat-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--text-2);
  font-size: 13.5px;
  transition: all var(--transition);
  position: relative;
  group: true;
}

.chat-item:hover { background: var(--bg-3); color: var(--text-1); }
.chat-item.active { background: var(--bg-3); color: var(--text-1); }
.chat-item.active::before {
  content: '';
  position: absolute;
  left: 0; top: 20%; bottom: 20%;
  width: 2px;
  background: var(--accent);
  border-radius: 1px;
}

.chat-item-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-item-del {
  opacity: 0;
  background: none;
  border: none;
  color: var(--text-3);
  cursor: pointer;
  padding: 2px;
  border-radius: 4px;
  font-size: 14px;
  transition: all var(--transition);
  line-height: 1;
}

.chat-item:hover .chat-item-del { opacity: 1; }
.chat-item-del:hover { color: var(--danger); background: rgba(255,83,112,0.1); }

/* Provider status */
.provider-status {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  font-family: var(--font-mono);
  padding: 3px 8px;
  border-radius: 20px;
  background: var(--bg-3);
  border: 1px solid var(--border);
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-3);
}

.status-dot.on { background: var(--success); box-shadow: 0 0 6px var(--success); }

/* ─── Main Area ──────────────────────────────────── */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  position: relative;
}

/* Header */
#topbar {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(13,15,26,0.9);
  backdrop-filter: blur(20px);
  z-index: 10;
  flex-shrink: 0;
}

#model-select {
  appearance: none;
  background: var(--bg-3);
  border: 1px solid var(--border-bright);
  color: var(--text-1);
  padding: 7px 32px 7px 12px;
  border-radius: 8px;
  font-family: var(--font-body);
  font-size: 13.5px;
  cursor: pointer;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ba3c4' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  transition: all var(--transition);
}

#model-select:hover { border-color: var(--accent); }
#model-select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
#model-select option { background: var(--bg-2); }

.topbar-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.toggle-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-2);
  font-size: 12.5px;
  cursor: pointer;
  transition: all var(--transition);
  font-family: var(--font-body);
}

.toggle-btn:hover { border-color: var(--border-bright); color: var(--text-1); }
.toggle-btn.active { background: var(--accent-glow); border-color: var(--accent); color: var(--accent); }
.toggle-btn svg { width: 14px; height: 14px; }

/* ─── Messages ───────────────────────────────────── */
#messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 0 12px;
  scroll-behavior: smooth;
}

.welcome-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 40px 24px;
  text-align: center;
  gap: 24px;
}

.welcome-logo {
  font-size: 56px;
  line-height: 1;
  filter: drop-shadow(0 0 30px var(--accent));
  animation: pulse-glow 3s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% { filter: drop-shadow(0 0 20px var(--accent)); }
  50% { filter: drop-shadow(0 0 40px var(--accent)) drop-shadow(0 0 60px var(--accent-2)); }
}

.welcome-title {
  font-size: 32px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--text-1) 0%, var(--accent) 50%, var(--accent-2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.04em;
}

.welcome-sub {
  color: var(--text-2);
  font-size: 16px;
  max-width: 420px;
  line-height: 1.7;
}

.suggestions-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  width: 100%;
  max-width: 640px;
}

.suggestion-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  cursor: pointer;
  text-align: left;
  display: flex;
  gap: 10px;
  align-items: flex-start;
  transition: all var(--transition);
  font-size: 13.5px;
  color: var(--text-2);
}

.suggestion-card:hover {
  background: var(--bg-3);
  border-color: var(--border-bright);
  color: var(--text-1);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.suggestion-icon { font-size: 20px; line-height: 1.2; }

/* Message bubbles */
.msg-wrap {
  max-width: 820px;
  margin: 0 auto;
  padding: 6px 24px;
  animation: msgIn 0.3s ease;
}

@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.msg { display: flex; gap: 12px; }

.msg-avatar {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
  margin-top: 2px;
  font-family: var(--font-mono);
  font-weight: 700;
}

.msg.user .msg-avatar {
  background: linear-gradient(135deg, var(--accent), #6356d8);
  color: white;
  box-shadow: 0 2px 10px var(--accent-glow);
}

.msg.assistant .msg-avatar {
  background: linear-gradient(135deg, var(--bg-4), var(--bg-3));
  border: 1px solid var(--border-bright);
  font-size: 18px;
}

.msg-body { flex: 1; min-width: 0; }

.msg-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.msg-name { font-weight: 600; font-size: 14px; }
.msg.user .msg-name { color: var(--accent); }
.msg.assistant .msg-name { color: var(--text-2); }

.msg-time { font-size: 11px; color: var(--text-3); font-family: var(--font-mono); }

.msg-badge {
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 20px;
  background: var(--bg-4);
  border: 1px solid var(--border);
  color: var(--text-3);
  font-family: var(--font-mono);
}

.msg-content {
  color: var(--text-1);
  line-height: 1.75;
  font-size: 15px;
}

.msg.user .msg-content {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  display: inline-block;
  max-width: 100%;
}

/* Markdown Rendering */
.msg-content h1, .msg-content h2, .msg-content h3 {
  margin: 20px 0 8px;
  line-height: 1.3;
}
.msg-content h1 { font-size: 22px; }
.msg-content h2 { font-size: 18px; }
.msg-content h3 { font-size: 16px; color: var(--text-2); }
.msg-content p { margin: 8px 0; }
.msg-content ul, .msg-content ol { padding-left: 20px; margin: 8px 0; }
.msg-content li { margin: 4px 0; }
.msg-content strong { color: var(--text-1); font-weight: 600; }
.msg-content em { color: var(--text-2); }
.msg-content a { color: var(--accent-2); text-decoration: none; }
.msg-content a:hover { text-decoration: underline; }
.msg-content blockquote {
  border-left: 3px solid var(--accent);
  padding: 8px 16px;
  background: var(--bg-2);
  border-radius: 0 8px 8px 0;
  margin: 12px 0;
  color: var(--text-2);
}
.msg-content table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 14px;
}
.msg-content th, .msg-content td {
  border: 1px solid var(--border);
  padding: 8px 12px;
  text-align: left;
}
.msg-content th { background: var(--bg-3); font-weight: 600; color: var(--text-2); }
.msg-content tr:nth-child(even) { background: var(--bg-2); }

/* Code blocks */
.msg-content pre {
  background: #0d1117 !important;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0;
  margin: 12px 0;
  overflow: hidden;
  position: relative;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: rgba(255,255,255,0.04);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-3);
}

.copy-code-btn {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-3);
  padding: 3px 10px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  transition: all var(--transition);
}

.copy-code-btn:hover { border-color: var(--accent); color: var(--accent); }

.msg-content pre code {
  display: block;
  padding: 14px 16px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.6;
  font-family: var(--font-mono);
}

.msg-content code:not(pre code) {
  background: var(--bg-3);
  color: var(--accent-2);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 13px;
}

/* Search results widget */
.search-widget {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 14px;
  font-size: 13.5px;
}

.search-widget-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent-2);
  font-family: var(--font-mono);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.search-result-item {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.search-result-item:last-child { border-bottom: none; }
.search-result-title { font-weight: 600; color: var(--accent-2); font-size: 13px; }
.search-result-snippet { color: var(--text-2); font-size: 12.5px; margin-top: 2px; }
.search-result-url { color: var(--text-3); font-size: 11px; font-family: var(--font-mono); margin-top: 2px; }

/* Typing indicator */
.typing-indicator {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 4px 0;
}
.typing-dot {
  width: 7px; height: 7px;
  background: var(--accent);
  border-radius: 50%;
  animation: blink 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }

/* Message actions */
.msg-actions {
  display: flex;
  gap: 6px;
  margin-top: 10px;
  opacity: 0;
  transition: opacity var(--transition);
}

.msg-wrap:hover .msg-actions { opacity: 1; }

.msg-action-btn {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-3);
  padding: 4px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: var(--font-body);
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all var(--transition);
}
.msg-action-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }

/* Token counter */
.token-badge {
  font-size: 10px;
  color: var(--text-3);
  font-family: var(--font-mono);
  margin-top: 8px;
}

/* ─── Input Area ─────────────────────────────────── */
#input-area {
  padding: 16px 24px 20px;
  border-top: 1px solid var(--border);
  background: var(--bg-1);
  flex-shrink: 0;
}

.input-container {
  max-width: 820px;
  margin: 0 auto;
  position: relative;
}

.file-preview {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--text-2);
}

.file-preview-remove {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--danger);
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
  line-height: 1;
}

.input-box {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: var(--bg-2);
  border: 1px solid var(--border-bright);
  border-radius: 14px;
  padding: 12px 14px;
  transition: all var(--transition);
}

.input-box:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

#user-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--text-1);
  font-family: var(--font-body);
  font-size: 15px;
  resize: none;
  line-height: 1.5;
  max-height: 200px;
  min-height: 24px;
}

#user-input::placeholder { color: var(--text-3); }

.input-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.input-icon-btn {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  color: var(--text-3);
  transition: all var(--transition);
  font-size: 18px;
}

.input-icon-btn:hover { color: var(--text-1); background: var(--bg-3); border-color: var(--border); }

#send-btn {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, var(--accent), #6356d8);
  border: none;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition);
  box-shadow: 0 2px 12px var(--accent-glow);
  flex-shrink: 0;
}

#send-btn:hover { transform: scale(1.05); box-shadow: 0 4px 16px var(--accent-glow); }
#send-btn:active { transform: scale(0.95); }
#send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
#send-btn svg { width: 16px; height: 16px; color: white; }

.input-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-3);
}

.input-hints { display: flex; gap: 14px; align-items: center; }

kbd {
  background: var(--bg-4);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 5px;
  font-family: var(--font-mono);
  font-size: 10px;
}

/* ─── Misc ───────────────────────────────────────── */
#file-input { display: none; }

.stop-btn {
  background: var(--danger);
  border: none;
  color: white;
  padding: 6px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-family: var(--font-body);
  font-size: 13px;
  display: none;
  align-items: center;
  gap: 6px;
}
.stop-btn.visible { display: flex; }

/* Toast */
.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: var(--bg-3);
  border: 1px solid var(--border-bright);
  color: var(--text-1);
  padding: 12px 20px;
  border-radius: 10px;
  font-size: 14px;
  z-index: 9998;
  animation: toastIn 0.3s ease;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
@keyframes toastIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; } }

/* Responsive */
@media (max-width: 768px) {
  #sidebar { position: fixed; transform: translateX(-100%); }
  #sidebar.open { transform: translateX(0); }
  .suggestions-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ═══ SIDEBAR ════════════════════════════════════════ -->
<nav id="sidebar">
  <div class="sidebar-header">
    <div class="logo">
      <div class="logo-icon">♍</div>
      <div>
        <div class="logo-text">Virgo AI</div>
        <div class="logo-sub">Engine v2.0</div>
      </div>
    </div>
    <button class="new-chat-btn" onclick="newChat()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>
      New Chat
    </button>
  </div>

  <div class="chat-list" id="chat-list">
    <div class="chat-section-label">Recent</div>
  </div>

  <div class="provider-status" id="provider-status">
    <span class="status-badge"><span class="status-dot" id="dot-openai"></span>OpenAI</span>
    <span class="status-badge"><span class="status-dot" id="dot-anthropic"></span>Claude</span>
    <span class="status-badge"><span class="status-dot" id="dot-gemini"></span>Gemini</span>
    <span class="status-badge"><span class="status-dot" id="dot-search"></span>Search</span>
  </div>
</nav>

<!-- ═══ MAIN ════════════════════════════════════════════ -->
<main id="main">
  <!-- Topbar -->
  <header id="topbar">
    <select id="model-select" onchange="changeModel(this.value)">
      <optgroup label="⬡ OpenAI" id="group-openai"></optgroup>
      <optgroup label="◈ Anthropic" id="group-anthropic"></optgroup>
      <optgroup label="◆ Google" id="group-gemini"></optgroup>
    </select>

    <div class="topbar-actions">
      <button class="toggle-btn" id="search-toggle" onclick="toggleSearch()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        Web Search
      </button>
      <button class="stop-btn" id="stop-btn" onclick="stopGeneration()">
        <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
        Stop
      </button>
    </div>
  </header>

  <!-- Messages -->
  <div id="messages">
    <div class="welcome-screen" id="welcome">
      <div class="welcome-logo">♍</div>
      <h1 class="welcome-title">Virgo AI Engine</h1>
      <p class="welcome-sub">Next-generation AI search. Multi-model. Streaming. Intelligent.</p>
      <div class="suggestions-grid" id="suggestions-grid"></div>
    </div>
  </div>

  <!-- Input -->
  <div id="input-area">
    <div class="input-container">
      <div id="file-preview-area"></div>
      <div class="input-box">
        <textarea id="user-input" placeholder="Ask anything..." rows="1" maxlength="32000"></textarea>
        <div class="input-actions">
          <label class="input-icon-btn" for="file-input" title="Upload file">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
          </label>
          <input type="file" id="file-input" accept="image/*,.pdf,.txt,.md,.csv,.json" onchange="handleFileUpload(this)">
          <button id="send-btn" onclick="sendMessage()" title="Send (Enter)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
          </button>
        </div>
      </div>
      <div class="input-footer">
        <div class="input-hints">
          <span><kbd>Enter</kbd> Send</span>
          <span><kbd>Shift+Enter</kbd> New line</span>
        </div>
        <span id="char-count" style="color:var(--text-3); font-family:var(--font-mono); font-size:11px;"></span>
      </div>
    </div>
  </div>
</main>

<script>
// ═══════════════════════════════════════════════════════════
//  VIRGO AI ENGINE — Frontend Logic
// ═══════════════════════════════════════════════════════════
const state = {
  chatId: null,
  model: 'gpt-4o',
  webSearch: false,
  streaming: false,
  controller: null,
  uploadedFile: null,
  history: [],
  models: {},
};

// ─── Init ──────────────────────────────────────────────────
async function init() {
  await loadStatus();
  await loadModels();
  await loadSuggestions();
  await loadChats();
  setupInput();
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    setDot('openai', s.openai);
    setDot('anthropic', s.anthropic);
    setDot('gemini', s.gemini);
    setDot('search', s.search);
  } catch (e) { console.error(e); }
}

function setDot(name, on) {
  const el = document.getElementById(`dot-${name}`);
  if (el) el.classList.toggle('on', on);
}

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    state.models = await r.json();
    const groups = { openai: 'group-openai', anthropic: 'group-anthropic', gemini: 'group-gemini' };
    for (const [id, info] of Object.entries(state.models)) {
      const g = document.getElementById(groups[info.provider]);
      if (!g) continue;
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `${info.icon} ${info.label}`;
      if (!info.available) opt.disabled = true;
      if (id === state.model) opt.selected = true;
      g.appendChild(opt);
    }
  } catch (e) { console.error(e); }
}

async function loadSuggestions() {
  try {
    const r = await fetch('/api/suggestions');
    const suggestions = await r.json();
    const grid = document.getElementById('suggestions-grid');
    suggestions.forEach(s => {
      const card = document.createElement('button');
      card.className = 'suggestion-card';
      card.innerHTML = `<span class="suggestion-icon">${s.icon}</span><span>${s.text}</span>`;
      card.onclick = () => { document.getElementById('user-input').value = s.text; sendMessage(); };
      grid.appendChild(card);
    });
  } catch (e) { console.error(e); }
}

async function loadChats() {
  try {
    const r = await fetch('/api/chats');
    const chats = await r.json();
    const list = document.getElementById('chat-list');
    list.innerHTML = '<div class="chat-section-label">Recent</div>';
    chats.forEach(c => addChatItem(c));
  } catch (e) { console.error(e); }
}

function addChatItem(chat, prepend = false) {
  const list = document.getElementById('chat-list');
  const el = document.createElement('div');
  el.className = 'chat-item';
  el.id = `chat-${chat.id}`;
  el.innerHTML = `
    <span style="font-size:14px;">${MODELS_ICON(chat.model)}</span>
    <span class="chat-item-title">${escHtml(chat.title)}</span>
    <button class="chat-item-del" onclick="deleteChat('${chat.id}', event)" title="Delete">×</button>
  `;
  el.onclick = (e) => { if (!e.target.classList.contains('chat-item-del')) loadChat(chat.id); };
  if (prepend) {
    const label = list.querySelector('.chat-section-label');
    label ? label.after(el) : list.prepend(el);
  } else {
    list.appendChild(el);
  }
}

function MODELS_ICON(model) {
  const info = state.models[model] || {};
  return info.icon || '◯';
}

// ─── Chat Actions ──────────────────────────────────────────
function newChat() {
  state.chatId = null;
  state.history = [];
  state.uploadedFile = null;
  document.getElementById('messages').innerHTML = `<div class="welcome-screen" id="welcome">
    <div class="welcome-logo">♍</div>
    <h1 class="welcome-title">Virgo AI Engine</h1>
    <p class="welcome-sub">Next-generation AI search. Multi-model. Streaming. Intelligent.</p>
    <div class="suggestions-grid" id="suggestions-grid"></div>
  </div>`;
  loadSuggestions();
  document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
  document.getElementById('file-preview-area').innerHTML = '';
  document.getElementById('user-input').value = '';
}

async function loadChat(chatId) {
  try {
    const r = await fetch(`/api/chats/${chatId}`);
    const data = await r.json();
    state.chatId = chatId;
    state.history = [];

    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';

    data.messages.forEach(m => {
      renderMessage(m.role, m.content, m.id);
      state.history.push({ role: m.role, content: m.content });
    });

    document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
    const el = document.getElementById(`chat-${chatId}`);
    if (el) el.classList.add('active');

    msgs.scrollTop = msgs.scrollHeight;
  } catch (e) { showToast('Failed to load chat'); }
}

async function deleteChat(chatId, e) {
  e.stopPropagation();
  await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
  const el = document.getElementById(`chat-${chatId}`);
  if (el) el.remove();
  if (state.chatId === chatId) newChat();
}

// ─── Send Message ──────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text || state.streaming) return;

  input.value = '';
  autoResize(input);
  hideWelcome();

  renderMessage('user', text);
  state.history.push({ role: 'user', content: text });

  const thinkingId = renderThinking();
  state.streaming = true;
  toggleSendStop(true);

  state.controller = new AbortController();

  try {
    const body = {
      message: text,
      model: state.model,
      chat_id: state.chatId,
      web_search: state.webSearch,
      history: state.history.slice(-20),
      file: state.uploadedFile,
    };

    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: state.controller.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = '';
    let msgEl = null;
    let firstChunk = true;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.type === 'meta') {
            state.chatId = ev.chat_id;
          } else if (ev.type === 'search_results') {
            removeThinking(thinkingId);
            renderSearchResults(ev.results);
            msgEl = createAssistantMsg(ev.model || state.model);
            firstChunk = false;
          } else if (ev.type === 'chunk') {
            if (firstChunk) {
              removeThinking(thinkingId);
              msgEl = createAssistantMsg(state.model);
              firstChunk = false;
            }
            assistantText += ev.text;
            updateAssistantMsg(msgEl, assistantText);
          } else if (ev.type === 'done') {
            if (firstChunk) { removeThinking(thinkingId); }
            updateTokenBadge(msgEl, ev.tokens);
            state.history.push({ role: 'assistant', content: assistantText });
            // Auto title on first message
            if (state.history.length === 2) {
              autoTitle();
              await loadChats();
            }
            // Mark chat active
            setTimeout(() => {
              document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
              const el = document.getElementById(`chat-${state.chatId}`);
              if (el) el.classList.add('active');
            }, 200);
          } else if (ev.type === 'error') {
            removeThinking(thinkingId);
            renderMessage('assistant', `⚠️ **Error:** ${ev.message}`);
          }
        } catch (e) { /* skip */ }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      removeThinking(thinkingId);
      renderMessage('assistant', `⚠️ **Connection error.** Please try again.`);
    }
  } finally {
    state.streaming = false;
    toggleSendStop(false);
    state.uploadedFile = null;
    document.getElementById('file-preview-area').innerHTML = '';
    scrollBottom();
  }
}

function stopGeneration() {
  if (state.controller) state.controller.abort();
}

// ─── Render Helpers ────────────────────────────────────────
function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

function renderMessage(role, content, id = null) {
  const msgs = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  if (id) wrap.dataset.msgId = id;

  const name = role === 'user' ? 'You' : 'Virgo';
  const avatar = role === 'user' ? 'U' : '♍';
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  wrap.innerHTML = `
    <div class="msg ${role}">
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-name">${name}</span>
          <span class="msg-time">${timeStr}</span>
        </div>
        <div class="msg-content">${role === 'assistant' ? renderMarkdown(content) : escHtml(content)}</div>
        ${role === 'assistant' ? '<div class="token-badge" id="tb-"></div>' : ''}
        <div class="msg-actions">
          <button class="msg-action-btn" onclick="copyMsg(this)" title="Copy">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            Copy
          </button>
        </div>
      </div>
    </div>`;

  msgs.appendChild(wrap);
  if (role === 'assistant') {
    wrap.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    addCodeHeaders(wrap);
  }
  scrollBottom();
  return wrap;
}

function createAssistantMsg(model) {
  const msgs = document.getElementById('messages');
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const modelInfo = state.models[model] || {};
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  const uid = Date.now();
  wrap.innerHTML = `
    <div class="msg assistant">
      <div class="msg-avatar">♍</div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-name">Virgo</span>
          <span class="msg-time">${timeStr}</span>
          <span class="msg-badge">${modelInfo.label || model}</span>
        </div>
        <div class="msg-content" id="mc-${uid}"></div>
        <div class="token-badge" id="tb-${uid}"></div>
        <div class="msg-actions">
          <button class="msg-action-btn" onclick="copyMsg(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            Copy
          </button>
        </div>
      </div>
    </div>`;
  msgs.appendChild(wrap);
  wrap._uid = uid;
  return wrap;
}

function updateAssistantMsg(wrap, text) {
  if (!wrap) return;
  const el = wrap.querySelector(`#mc-${wrap._uid}`);
  if (el) {
    el.innerHTML = renderMarkdown(text);
    el.querySelectorAll('pre code').forEach(c => hljs.highlightElement(c));
    addCodeHeaders(wrap);
    scrollBottom();
  }
}

function updateTokenBadge(wrap, tokens) {
  if (!wrap || !tokens) return;
  const el = wrap.querySelector(`#tb-${wrap._uid}`);
  if (el) el.textContent = `${tokens.toLocaleString()} tokens`;
}

function renderThinking() {
  const msgs = document.getElementById('messages');
  const id = 'think-' + Date.now();
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  wrap.id = id;
  wrap.innerHTML = `
    <div class="msg assistant">
      <div class="msg-avatar">♍</div>
      <div class="msg-body">
        <div class="msg-header"><span class="msg-name">Virgo</span></div>
        <div class="msg-content">
          <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
          </div>
        </div>
      </div>
    </div>`;
  msgs.appendChild(wrap);
  scrollBottom();
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderSearchResults(results) {
  const msgs = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  const items = results.map(r => `
    <div class="search-result-item">
      <div class="search-result-title">${escHtml(r.title)}</div>
      <div class="search-result-snippet">${escHtml((r.snippet||'').slice(0,120))}…</div>
      <div class="search-result-url">${escHtml(r.url)}</div>
    </div>`).join('');
  wrap.innerHTML = `
    <div class="msg assistant">
      <div class="msg-avatar" style="font-size:18px;">🔍</div>
      <div class="msg-body">
        <div class="search-widget">
          <div class="search-widget-title">🌐 Web Search Results</div>
          ${items}
        </div>
      </div>
    </div>`;
  msgs.appendChild(wrap);
  scrollBottom();
}

function addCodeHeaders(container) {
  container.querySelectorAll('pre:not(.has-header)').forEach(pre => {
    pre.classList.add('has-header');
    const code = pre.querySelector('code');
    const lang = (code?.className || '').replace('language-', '') || 'code';
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `<span>${lang}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy</button>`;
    pre.prepend(header);
  });
}

// ─── Markdown ──────────────────────────────────────────────
marked.setOptions({
  gfm: true,
  breaks: true,
  highlight: (code, lang) => {
    try {
      return lang ? hljs.highlight(code, { language: lang }).value : hljs.highlightAuto(code).value;
    } catch { return code; }
  }
});

function renderMarkdown(text) {
  return marked.parse(text || '');
}

// ─── Utilities ─────────────────────────────────────────────
function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scrollBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function showToast(msg, duration = 2500) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

function toggleSendStop(streaming) {
  document.getElementById('send-btn').disabled = streaming;
  const stop = document.getElementById('stop-btn');
  stop.classList.toggle('visible', streaming);
}

function changeModel(model) {
  state.model = model;
}

function toggleSearch() {
  state.webSearch = !state.webSearch;
  document.getElementById('search-toggle').classList.toggle('active', state.webSearch);
  showToast(state.webSearch ? '🔍 Web search enabled' : '🔍 Web search disabled', 1500);
}

async function autoTitle() {
  if (!state.chatId) return;
  try {
    const r = await fetch(`/api/chats/${state.chatId}/autotitle`, { method: 'POST' });
    const { title } = await r.json();
    const el = document.querySelector(`#chat-${state.chatId} .chat-item-title`);
    if (el) el.textContent = title;
  } catch (e) {}
}

function copyMsg(btn) {
  const content = btn.closest('.msg-body').querySelector('.msg-content');
  navigator.clipboard.writeText(content.innerText);
  const orig = btn.innerHTML;
  btn.innerHTML = '✓ Copied';
  setTimeout(() => { btn.innerHTML = orig; }, 1500);
}

function copyCode(btn) {
  const code = btn.closest('pre').querySelector('code');
  navigator.clipboard.writeText(code.innerText);
  const orig = btn.textContent;
  btn.textContent = 'Copied!';
  setTimeout(() => { btn.textContent = orig; }, 1500);
}

// ─── File Upload ───────────────────────────────────────────
async function handleFileUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await r.json();
    state.uploadedFile = data;
    const preview = document.getElementById('file-preview-area');
    const icon = file.type.startsWith('image/') ? '🖼️' : '📎';
    preview.innerHTML = `
      <div class="file-preview">
        ${icon} <strong>${escHtml(file.name)}</strong>
        <span style="color:var(--text-3); font-size:12px;">${(file.size/1024).toFixed(1)} KB</span>
        <button class="file-preview-remove" onclick="clearFile()">×</button>
      </div>`;
    showToast(`✓ File attached: ${file.name}`, 2000);
  } catch (e) { showToast('Upload failed'); }
  input.value = '';
}

function clearFile() {
  state.uploadedFile = null;
  document.getElementById('file-preview-area').innerHTML = '';
}

// ─── Input Setup ───────────────────────────────────────────
function setupInput() {
  const input = document.getElementById('user-input');
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener('input', () => {
    autoResize(input);
    const len = input.value.length;
    const counter = document.getElementById('char-count');
    if (len > 1000) {
      counter.textContent = `${len.toLocaleString()} / 32,000`;
    } else {
      counter.textContent = '';
    }
  });
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

// ─── Boot ──────────────────────────────────────────────────
init();
</script>
</body>
</html>"""

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          🚀 VIRGO AI ENGINE STARTING UP                 ║
║                                                          ║
║  Multi-model AI Search Engine                            ║
║  → OpenAI GPT-4o / o1                                   ║
║  → Anthropic Claude 3.5 / Opus                          ║
║  → Google Gemini 1.5 / 2.0                              ║
║  → Real-time Web Search (DuckDuckGo)                    ║
║  → Streaming Responses (SSE)                            ║
║  → Chat History (SQLite)                                ║
║  → Image/File Upload & Vision                           ║
╚══════════════════════════════════════════════════════════╝

 Open: http://localhost:5000
""")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
