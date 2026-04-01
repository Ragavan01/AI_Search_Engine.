"""
╔══════════════════════════════════════════════════════════╗
║              VIRGO AI  –  Premium Search Engine          ║
║         Built with Streamlit · Groq · Tavily             ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Virgo AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. THEME DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
THEME_VARS = {
    "dark": """
        --bg-primary:    #080C14;
        --bg-secondary:  #0F1623;
        --bg-card:       rgba(255,255,255,0.04);
        --border:        rgba(255,255,255,0.09);
        --border-focus:  rgba(99,179,237,0.55);
        --text-primary:  #F0F4FF;
        --text-secondary:#8A9BBF;
        --text-muted:    #4A5878;
        --accent-1:      #63B3ED;
        --accent-2:      #9F7AEA;
        --accent-3:      #68D391;
        --user-bubble-bg:linear-gradient(135deg,#2A3F6F,#1E2D4F);
        --user-bubble-fg:#F0F4FF;
        --ai-bubble-bg:  rgba(255,255,255,0.04);
        --shadow:        0 8px 32px rgba(0,0,0,0.4);
        --glow:          0 0 20px rgba(99,179,237,0.15);
    """,
    "light": """
        --bg-primary:    #F0F4FA;
        --bg-secondary:  #FFFFFF;
        --bg-card:       #FFFFFF;
        --border:        rgba(0,0,0,0.09);
        --border-focus:  rgba(59,130,246,0.5);
        --text-primary:  #1A202C;
        --text-secondary:#4A5568;
        --text-muted:    #A0AEC0;
        --accent-1:      #3B82F6;
        --accent-2:      #7C3AED;
        --accent-3:      #10B981;
        --user-bubble-bg:linear-gradient(135deg,#3B82F6,#2563EB);
        --user-bubble-fg:#FFFFFF;
        --ai-bubble-bg:  #FFFFFF;
        --shadow:        0 4px 20px rgba(0,0,0,0.07);
        --glow:          0 0 20px rgba(59,130,246,0.1);
    """,
}

THEME_LABELS = ["🌑 Dark", "☀️ Light", "🖥️ System"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
def make_memory(k: int):
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output",
    )

_defaults = {
    "messages":       [],
    "memory":         None,
    "theme_label":    "🌑 Dark",
    "model_key":      "llama-3.3-70b-versatile",
    "model_label":    "Llama 3.3 · 70B",
    "temperature":    0.5,
    "max_results":    5,
    "memory_window":  6,
    "show_reasoning": False,
    "total_queries":  0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.memory is None:
    st.session_state.memory = make_memory(st.session_state.memory_window)

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD THEME CSS BLOCK
# ─────────────────────────────────────────────────────────────────────────────
chosen = st.session_state.theme_label

if chosen == "🖥️ System":
    css_vars = f"""
        @media (prefers-color-scheme: dark)  {{ :root {{ {THEME_VARS["dark"]}  }} }}
        @media (prefers-color-scheme: light) {{ :root {{ {THEME_VARS["light"]} }} }}
    """
elif chosen == "☀️ Light":
    css_vars = f":root {{ {THEME_VARS['light']} }}"
else:
    css_vars = f":root {{ {THEME_VARS['dark']} }}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. INJECT GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

{css_vars}

/* ── GLOBAL ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, .stApp {{
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
.stDeployButton {{ display: none !important; }}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-secondary) !important;
}}

/* ── LOGO ── */
.v-logo {{ padding: 24px 0 18px; text-align: center; }}
.v-logo-name {{
    font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.v-logo-tag {{
    font-size: 0.65rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--text-muted); margin-top: 4px;
}}

/* ── SECTION TITLE ── */
.s-title {{
    font-size: 0.65rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: var(--text-muted);
    padding: 14px 0 8px;
}}

/* ── STAT CHIPS ── */
.stat-row {{ display: flex; gap: 8px; margin: 6px 0 4px; }}
.stat-chip {{
    flex: 1; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 6px; text-align: center;
}}
.stat-chip .sv {{ font-size: 1.25rem; font-weight: 700; color: var(--accent-1); }}
.stat-chip .sl {{
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-muted);
}}

/* ── BUTTONS ── */
.stButton > button {{
    width: 100%; background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 9px 14px !important; transition: all 0.2s !important;
}}
.stButton > button:hover {{
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(99,179,237,0.05) !important;
}}

/* ── WIDGETS ── */
.stSelectbox label, .stSlider label, .stToggle label, .stRadio label {{
    font-size: 0.8rem !important; color: var(--text-secondary) !important;
    font-family: 'Sora', sans-serif !important;
}}
.stDivider {{ border-color: var(--border) !important; }}
.stDownloadButton > button {{
    width: 100%; background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 9px 14px !important; transition: all 0.2s !important;
}}
.stDownloadButton > button:hover {{
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(99,179,237,0.05) !important;
}}

/* ── RADIO (theme) ── */
div[data-testid="stRadio"] > div {{
    flex-direction: row !important; gap: 6px !important;
    flex-wrap: wrap !important;
}}
div[data-testid="stRadio"] label {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 5px 11px !important;
    font-size: 0.77rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin: 0 !important;
}}
div[data-testid="stRadio"] label:hover {{
    border-color: var(--accent-1) !important;
}}

/* ── TOP BAR ── */
.top-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 15px 30px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 50;
}}
.top-bar-left {{
    display: flex; align-items: center; gap: 10px;
    font-size: 1rem; font-weight: 600;
}}
.pulse-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-3);
    box-shadow: 0 0 6px var(--accent-3);
    animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.55; transform:scale(0.8); }}
}}
.top-bar-right {{
    font-size: 0.75rem; color: var(--text-muted);
    display: flex; gap: 14px;
}}

/* ── WELCOME SCREEN ── */
.welcome {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 70px 20px; text-align: center; gap: 14px;
}}
.welcome-icon {{ font-size: 3.2rem; }}
.welcome-h {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.welcome-p {{
    font-size: 0.95rem; color: var(--text-secondary);
    max-width: 440px; line-height: 1.7; margin-top: 4px;
}}
.chips {{
    display: flex; flex-wrap: wrap;
    gap: 9px; justify-content: center; margin-top: 10px;
}}
.chip {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 999px; padding: 8px 16px;
    font-size: 0.8rem; color: var(--text-secondary);
    cursor: default;
}}

/* ── CHAT AREA ── */
.chat-scroll {{ padding: 18px 0 12px; }}

/* ── MESSAGE ROW ── */
.msg-row {{
    display: flex; gap: 12px; padding: 7px 28px;
    animation: fadeUp 0.28s ease;
}}
.msg-row.user {{ flex-direction: row-reverse; }}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(8px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ── AVATARS ── */
.av {{
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0; margin-top: 3px;
}}
.av.ai   {{ background: linear-gradient(135deg, var(--accent-1), var(--accent-2)); color:#fff; }}
.av.user {{ background: linear-gradient(135deg, var(--accent-2), var(--accent-1)); color:#fff; }}

/* ── BUBBLES ── */
.bub {{
    max-width: 70%; border-radius: 18px;
    padding: 12px 17px; font-size: 0.9rem; line-height: 1.72;
    border: 1px solid var(--border); box-shadow: var(--shadow);
}}
.bub.ai   {{
    background: var(--ai-bubble-bg); color: var(--text-primary);
    border-radius: 4px 18px 18px 18px;
}}
.bub.user {{
    background: var(--user-bubble-bg); color: var(--user-bubble-fg);
    border: none; border-radius: 18px 4px 18px 18px;
}}
.bub-meta {{
    font-size: 0.65rem; color: var(--text-muted);
    margin-top: 5px; display: flex; align-items: center; gap: 6px;
}}
.bub.user .bub-meta {{
    color: rgba(255,255,255,0.45); justify-content: flex-end;
}}
.bub code {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    background: rgba(0,0,0,0.2); padding: 1px 5px; border-radius: 4px;
}}
.bub pre {{
    background: rgba(0,0,0,0.25); border: 1px solid var(--border);
    border-radius: 9px; padding: 11px 13px;
    overflow-x: auto; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}}

/* ── TYPING INDICATOR ── */
.typing-row {{ display: flex; gap: 12px; padding: 7px 28px; }}
.typing-bub {{
    background: var(--ai-bubble-bg); border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px; display: flex; align-items: center; gap: 5px;
}}
.t-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent-1); animation: tbounce 1.2s infinite;
}}
.t-dot:nth-child(2) {{ animation-delay: 0.18s; }}
.t-dot:nth-child(3) {{ animation-delay: 0.36s; }}
@keyframes tbounce {{
    0%,80%,100% {{ transform:translateY(0);   opacity:0.35; }}
    40%          {{ transform:translateY(-6px); opacity:1; }}
}}

/* ── INPUT AREA ── */
.input-wrap {{
    padding: 14px 28px 20px;
    background: var(--bg-primary);
    border-top: 1px solid var(--border);
}}
.stChatInput > div {{
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: var(--shadow) !important;
    max-width: 800px !important; margin: 0 auto !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
.stChatInput > div:focus-within {{
    border-color: var(--border-focus) !important;
    box-shadow: var(--glow) !important;
}}
.stChatInput textarea {{
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: var(--accent-1) !important;
    padding: 13px 15px !important;
}}
.stChatInput textarea::placeholder {{ color: var(--text-muted) !important; }}
.stChatInput button {{ color: var(--accent-1) !important; }}

/* ── STATUS WIDGET ── */
[data-testid="stStatus"] {{
    border-radius: 11px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}

/* ── FOOTER ── */
.v-footer {{
    text-align: center; font-size: 0.64rem;
    color: var(--text-muted); margin-top: 18px; padding-bottom: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. API KEY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
try:
    GROQ_KEY   = st.secrets["GROQ_API_KEY"]
    TAVILY_KEY = st.secrets["TAVILY_API_KEY"]
except KeyError as missing_key:
    st.error(
        f"⚠️ Missing secret: **{missing_key}**  \n"
        "Go to **Settings → Secrets** and add `GROQ_API_KEY` and `TAVILY_API_KEY`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="v-logo">
        <div class="v-logo-name">🌌 Virgo AI</div>
        <div class="v-logo-tag">Real-Time Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme ──
    st.markdown('<div class="s-title">Appearance</div>', unsafe_allow_html=True)
    new_theme = st.radio("Theme", THEME_LABELS,
                         index=THEME_LABELS.index(st.session_state.theme_label),
                         label_visibility="collapsed", horizontal=True)
    if new_theme != st.session_state.theme_label:
        st.session_state.theme_label = new_theme
        st.rerun()

    st.divider()

    # ── Model ──
    st.markdown('<div class="s-title">Model</div>', unsafe_allow_html=True)
    MODELS = {
        "Llama 3.3 · 70B  ✦ Best"    : "llama-3.3-70b-versatile",
        "Llama 3.1 · 8B  ⚡ Fastest"  : "llama-3.1-8b-instant",
        "Gemma 2 · 9B  ⚖️ Balanced"   : "gemma2-9b-it",
        "Mixtral · 8×7B  🎨 Creative" : "mixtral-8x7b-32768",
    }
    model_label = st.selectbox("Model", list(MODELS.keys()),
                               label_visibility="collapsed")
    st.session_state.model_key   = MODELS[model_label]
    st.session_state.model_label = model_label.split("·")[0].strip()

    st.session_state.temperature = st.slider(
        "Creativity / Temperature",
        0.0, 1.0, st.session_state.temperature, 0.05,
        help="0 = precise & factual  ·  1 = creative & varied"
    )

    st.divider()

    # ── Search ──
    st.markdown('<div class="s-title">Search</div>', unsafe_allow_html=True)
    st.session_state.max_results = st.slider(
        "Web Results per Query",
        1, 10, st.session_state.max_results, 1)
    st.session_state.memory_window = st.slider(
        "Memory Window (turns)",
        2, 20, st.session_state.memory_window, 1,
        help="Number of past exchanges the agent remembers")
    st.session_state.show_reasoning = st.toggle(
        "Show Agent Reasoning", st.session_state.show_reasoning)

    st.divider()

    # ── Stats ──
    st.markdown('<div class="s-title">Session</div>', unsafe_allow_html=True)
    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    n_ai   = len(st.session_state.messages) - n_user
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip"><div class="sv">{n_user}</div><div class="sl">Queries</div></div>
        <div class="stat-chip"><div class="sv">{n_ai}</div><div class="sl">Replies</div></div>
        <div class="stat-chip"><div class="sv">{st.session_state.total_queries}</div><div class="sl">Total</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Actions ──
    st.markdown('<div class="s-title">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.messages      = []
        st.session_state.memory        = make_memory(st.session_state.memory_window)
        st.session_state.total_queries = 0
        st.rerun()

    if st.session_state.messages:
        lines = ["# Virgo AI — Conversation Export\n\n"]
        for m in st.session_state.messages:
            who = "**You**" if m["role"] == "user" else "**Virgo AI**"
            lines.append(f"{who}  _{m.get('time','')}_\n\n{m['content']}\n\n---\n\n")
        st.download_button(
            "⬇️  Export Chat (.md)",
            data="".join(lines),
            file_name=f"virgo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    st.markdown(
        '<div class="v-footer">Virgo AI v2.0 · Groq + Tavily</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8. ReAct PROMPT  (local — zero dependency on LangSmith / hub.pull)
# ─────────────────────────────────────────────────────────────────────────────
REACT_TEMPLATE = """You are Virgo AI — a premium, professional research assistant.
Always search the web for current information before answering factual or time-sensitive questions.
Format responses clearly with markdown (bold, bullet points, code blocks) when it improves readability.
Be concise, accurate, and comprehensive.

Available tools:
{tools}

You MUST follow this EXACT format:

Question: the input question you must answer
Thought: think step-by-step about what to do
Action: the action to take — must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought / Action / Action Input / Observation as needed)
Thought: I now know the final answer
Final Answer: the complete and well-formatted final answer

Begin!

Conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    template=REACT_TEMPLATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. AGENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_agent_executor():
    """Build a fresh AgentExecutor honoring current sidebar settings."""
    llm = ChatGroq(
        api_key=GROQ_KEY,
        model=st.session_state.model_key,
        temperature=st.session_state.temperature,
    )
    tools = [TavilySearchResults(
        api_key=TAVILY_KEY,
        max_results=st.session_state.max_results,
    )]
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=st.session_state.show_reasoning,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. MESSAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_message(msg: dict):
    is_user = msg["role"] == "user"
    row_cls = "msg-row user" if is_user else "msg-row"
    av_cls  = "av user"     if is_user else "av ai"
    bub_cls = "bub user"    if is_user else "bub ai"
    av_icon = "U"           if is_user else "V"
    ts      = msg.get("time", "")
    model_tag = (
        f'<span style="color:var(--accent-1)">⚡ {msg["model"]}</span> · '
        if not is_user and msg.get("model") else ""
    )
    st.markdown(f"""
    <div class="{row_cls}">
        <div class="{av_cls}">{av_icon}</div>
        <div>
            <div class="{bub_cls}">{msg["content"]}</div>
            <div class="bub-meta">{model_tag}{ts}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN — TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <span class="pulse-dot"></span>
        Virgo AI
        <span style="color:var(--text-muted);font-weight:400;font-size:0.82rem;">
            · {st.session_state.model_label}
        </span>
    </div>
    <div class="top-bar-right">
        <span>🌐 {st.session_state.max_results} results</span>
        <span>🧠 {st.session_state.memory_window}-turn memory</span>
        <span>🌡️ {st.session_state.temperature}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 12. CHAT DISPLAY — Welcome or History
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🌌</div>
        <div class="welcome-h">Welcome to Virgo AI</div>
        <div class="welcome-p">
            A premium real-time intelligence engine — powered by Groq and live web search.
            Ask anything and get accurate, up-to-date answers instantly.
        </div>
        <div class="chips">
            <div class="chip">🔬 Latest AI research</div>
            <div class="chip">📈 Stock market today</div>
            <div class="chip">🌍 World news</div>
            <div class="chip">💡 Explain quantum computing</div>
            <div class="chip">🚀 Latest SpaceX launch</div>
            <div class="chip">🎬 Top movies 2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 13. INPUT + AGENT RESPONSE
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask Virgo anything…")

if user_input and user_input.strip():
    now = datetime.now().strftime("%H:%M")

    # Save & render user message
    user_msg = {"role": "user", "content": user_input.strip(), "time": now}
    st.session_state.messages.append(user_msg)
    st.session_state.total_queries += 1
    render_message(user_msg)

    # Typing animation placeholder
    typing_ph = st.empty()
    typing_ph.markdown("""
    <div class="typing-row">
        <div class="av ai">V</div>
        <div class="typing-bub">
            <div class="t-dot"></div>
            <div class="t-dot"></div>
            <div class="t-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent invocation
    try:
        with st.status("🔍 Virgo is searching…",
                       expanded=st.session_state.show_reasoning):
            executor = build_agent_executor()
            result   = executor.invoke({"input": user_input.strip()})
            answer   = result.get("output", "I was unable to generate a response.")

        typing_ph.empty()

        ai_msg = {
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%H:%M"),
            "model": st.session_state.model_label,
        }
        st.session_state.messages.append(ai_msg)
        render_message(ai_msg)

    except ValueError as ve:
        typing_ph.empty()
        err = "⚠️ The model's output couldn't be parsed. Please rephrase and try again."
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "time": datetime.now().strftime("%H:%M")}
        )
        st.warning(err)

    except Exception as exc:
        typing_ph.empty()
        err = f"❌ Error: `{exc}`"
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "time": datetime.now().strftime("%H:%M")}
        )
        st.error(err)

    st.rerun()
