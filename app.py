import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Virgo AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
DARK = """
    --bg:       #080C14; --bg2: #0F1623; --card: rgba(255,255,255,0.04);
    --border:   rgba(255,255,255,0.09); --bfocus: rgba(99,179,237,0.5);
    --t1: #F0F4FF; --t2: #8A9BBF; --tm: #4A5878;
    --a1: #63B3ED; --a2: #9F7AEA; --a3: #68D391;
    --ub: linear-gradient(135deg,#2A3F6F,#1E2D4F); --uf: #F0F4FF;
    --ab: rgba(255,255,255,0.04);
    --sh: 0 8px 32px rgba(0,0,0,0.4); --glow: 0 0 20px rgba(99,179,237,0.15);
"""
LIGHT = """
    --bg:       #F0F4FA; --bg2: #FFFFFF; --card: #FFFFFF;
    --border:   rgba(0,0,0,0.09); --bfocus: rgba(59,130,246,0.5);
    --t1: #1A202C; --t2: #4A5568; --tm: #A0AEC0;
    --a1: #3B82F6; --a2: #7C3AED; --a3: #10B981;
    --ub: linear-gradient(135deg,#3B82F6,#2563EB); --uf: #FFFFFF;
    --ab: #FFFFFF;
    --sh: 0 4px 20px rgba(0,0,0,0.07); --glow: 0 0 20px rgba(59,130,246,0.1);
"""
THEME_OPTIONS = ["🌑 Dark", "☀️ Light", "🖥️ System"]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "messages": [], "chat_history": [],
    "theme": "🌑 Dark",
    "model_key": "llama-3.3-70b-versatile", "model_lbl": "Llama 3.3 · 70B",
    "temp": 0.5, "max_res": 5, "mem_win": 6,
    "verbose": False, "total": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────────────────────
t = st.session_state.theme
if t == "🖥️ System":
    root_css = f"@media(prefers-color-scheme:dark){{:root{{{DARK}}}}}@media(prefers-color-scheme:light){{:root{{{LIGHT}}}}}"
elif t == "☀️ Light":
    root_css = f":root{{{LIGHT}}}"
else:
    root_css = f":root{{{DARK}}}"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
{root_css}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,.stApp{{background:var(--bg)!important;color:var(--t1)!important;font-family:'Sora',sans-serif!important}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0!important;max-width:100%!important}}
.stDeployButton{{display:none!important}}

[data-testid="stSidebar"]{{background:var(--bg2)!important;border-right:1px solid var(--border)!important}}
[data-testid="stSidebar"] *{{color:var(--t1)!important;font-family:'Sora',sans-serif!important}}

.logo{{padding:22px 0 16px;text-align:center}}
.logo-name{{font-size:1.7rem;font-weight:700;background:linear-gradient(135deg,var(--a1),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.logo-tag{{font-size:.63rem;letter-spacing:2.5px;text-transform:uppercase;color:var(--tm);margin-top:4px}}
.sec{{font-size:.63rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--tm);padding:13px 0 7px}}

.stat-row{{display:flex;gap:8px;margin:5px 0}}
.stat-chip{{flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:9px 5px;text-align:center}}
.sv{{font-size:1.2rem;font-weight:700;color:var(--a1)}}
.sl{{font-size:.59rem;text-transform:uppercase;letter-spacing:1px;color:var(--tm)}}

.stButton>button,.stDownloadButton>button{{
    width:100%;background:var(--card)!important;color:var(--t1)!important;
    border:1px solid var(--border)!important;border-radius:10px!important;
    font-family:'Sora',sans-serif!important;font-size:.81rem!important;
    font-weight:500!important;padding:8px 13px!important;transition:all .2s!important}}
.stButton>button:hover,.stDownloadButton>button:hover{{
    border-color:var(--a1)!important;color:var(--a1)!important;
    background:rgba(99,179,237,.05)!important}}
.stDivider{{border-color:var(--border)!important}}

div[data-testid="stRadio"]>div{{flex-direction:row!important;gap:6px!important;flex-wrap:wrap!important}}
div[data-testid="stRadio"] label{{
    background:var(--card)!important;border:1px solid var(--border)!important;
    border-radius:8px!important;padding:5px 10px!important;font-size:.76rem!important;
    cursor:pointer!important;transition:all .2s!important;margin:0!important}}
div[data-testid="stRadio"] label:hover{{border-color:var(--a1)!important}}

.topbar{{
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 28px;background:var(--bg2);border-bottom:1px solid var(--border);
    position:sticky;top:0;z-index:50}}
.topbar-l{{display:flex;align-items:center;gap:9px;font-size:.97rem;font-weight:600}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--a3);
    box-shadow:0 0 6px var(--a3);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(.8)}}}}
.topbar-r{{font-size:.73rem;color:var(--tm);display:flex;gap:13px}}

.welcome{{display:flex;flex-direction:column;align-items:center;justify-content:center;
    padding:65px 18px;text-align:center;gap:13px}}
.wico{{font-size:3rem}}
.wh{{font-size:1.9rem;font-weight:700;background:linear-gradient(135deg,var(--a1),var(--a2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.wp{{font-size:.93rem;color:var(--t2);max-width:430px;line-height:1.7;margin-top:3px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:9px}}
.chip{{background:var(--card);border:1px solid var(--border);border-radius:999px;
    padding:7px 15px;font-size:.78rem;color:var(--t2)}}

.msg-row{{display:flex;gap:11px;padding:6px 26px;animation:fu .27s ease}}
.msg-row.user{{flex-direction:row-reverse}}
@keyframes fu{{from{{opacity:0;transform:translateY(7px)}}to{{opacity:1;transform:translateY(0)}}}}
.av{{width:33px;height:33px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:.76rem;font-weight:700;flex-shrink:0;margin-top:3px}}
.av.ai{{background:linear-gradient(135deg,var(--a1),var(--a2));color:#fff}}
.av.user{{background:linear-gradient(135deg,var(--a2),var(--a1));color:#fff}}
.bub{{max-width:70%;border-radius:18px;padding:11px 16px;font-size:.89rem;
    line-height:1.72;border:1px solid var(--border);box-shadow:var(--sh)}}
.bub.ai{{background:var(--ab);color:var(--t1);border-radius:4px 18px 18px 18px}}
.bub.user{{background:var(--ub);color:var(--uf);border:none;border-radius:18px 4px 18px 18px}}
.bub-meta{{font-size:.64rem;color:var(--tm);margin-top:4px;display:flex;align-items:center;gap:5px}}
.bub.user .bub-meta{{color:rgba(255,255,255,.4);justify-content:flex-end}}
.bub code{{font-family:'JetBrains Mono',monospace;font-size:.79rem;
    background:rgba(0,0,0,.2);padding:1px 5px;border-radius:4px}}
.bub pre{{background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:8px;
    padding:10px 12px;overflow-x:auto;margin-top:7px;
    font-family:'JetBrains Mono',monospace;font-size:.77rem}}

.trow{{display:flex;gap:11px;padding:6px 26px}}
.tbub{{background:var(--ab);border:1px solid var(--border);border-radius:4px 18px 18px 18px;
    padding:11px 16px;display:flex;align-items:center;gap:5px}}
.td{{width:7px;height:7px;border-radius:50%;background:var(--a1);animation:tb 1.2s infinite}}
.td:nth-child(2){{animation-delay:.18s}}.td:nth-child(3){{animation-delay:.36s}}
@keyframes tb{{0%,80%,100%{{transform:translateY(0);opacity:.3}}40%{{transform:translateY(-6px);opacity:1}}}}

.stChatInput>div{{
    background:var(--card)!important;border:1.5px solid var(--border)!important;
    border-radius:14px!important;box-shadow:var(--sh)!important;
    max-width:800px!important;margin:0 auto!important;
    transition:border-color .2s,box-shadow .2s!important}}
.stChatInput>div:focus-within{{border-color:var(--bfocus)!important;box-shadow:var(--glow)!important}}
.stChatInput textarea{{
    background:transparent!important;color:var(--t1)!important;
    font-family:'Sora',sans-serif!important;font-size:.89rem!important;
    caret-color:var(--a1)!important;padding:12px 14px!important}}
.stChatInput textarea::placeholder{{color:var(--tm)!important}}
.stChatInput button{{color:var(--a1)!important}}

[data-testid="stStatus"]{{
    border-radius:10px!important;border:1px solid var(--border)!important;
    background:var(--card)!important;font-family:'Sora',sans-serif!important;font-size:.81rem!important}}
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:99px}}
.vfoot{{text-align:center;font-size:.63rem;color:var(--tm);margin-top:16px;padding-bottom:4px}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────────────────────
try:
    GROQ_KEY   = st.secrets["GROQ_API_KEY"]
    TAVILY_KEY = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ Missing secret: **{e}** — add it in Settings → Secrets on Streamlit Cloud.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo"><div class="logo-name">🌌 Virgo AI</div><div class="logo-tag">Real-Time Intelligence</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Appearance</div>', unsafe_allow_html=True)
    new_theme = st.radio("Theme", THEME_OPTIONS,
                         index=THEME_OPTIONS.index(st.session_state.theme),
                         label_visibility="collapsed", horizontal=True)
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    st.markdown('<div class="sec">Model</div>', unsafe_allow_html=True)

    MODELS = {
        "Llama 3.3 · 70B  ✦ Best"    : "llama-3.3-70b-versatile",
        "Llama 3.1 · 8B  ⚡ Fastest"  : "llama-3.1-8b-instant",
        "Gemma 2 · 9B  ⚖️ Balanced"   : "gemma2-9b-it",
        "Mixtral · 8×7B  🎨 Creative" : "mixtral-8x7b-32768",
    }
    mlabel = st.selectbox("Model", list(MODELS.keys()), label_visibility="collapsed")
    st.session_state.model_key = MODELS[mlabel]
    st.session_state.model_lbl = mlabel.split("·")[0].strip()

    st.session_state.temp = st.slider("Creativity / Temperature", 0.0, 1.0,
                                       st.session_state.temp, 0.05,
                                       help="0 = factual · 1 = creative")
    st.divider()
    st.markdown('<div class="sec">Search</div>', unsafe_allow_html=True)
    st.session_state.max_res = st.slider("Web Results per Query", 1, 10, st.session_state.max_res, 1)
    st.session_state.mem_win = st.slider("Memory Window (turns)", 2, 20, st.session_state.mem_win, 1)
    st.session_state.verbose = st.toggle("Show Agent Reasoning", st.session_state.verbose)

    st.divider()
    st.markdown('<div class="sec">Session</div>', unsafe_allow_html=True)
    nu = sum(1 for m in st.session_state.messages if m["role"] == "user")
    na = len(st.session_state.messages) - nu
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip"><div class="sv">{nu}</div><div class="sl">Queries</div></div>
        <div class="stat-chip"><div class="sv">{na}</div><div class="sl">Replies</div></div>
        <div class="stat-chip"><div class="sv">{st.session_state.total}</div><div class="sl">Total</div></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.total        = 0
        st.rerun()

    if st.session_state.messages:
        lines = ["# Virgo AI — Export\n\n"]
        for m in st.session_state.messages:
            who = "**You**" if m["role"] == "user" else "**Virgo AI**"
            lines.append(f"{who} _{m.get('time','')}_\n\n{m['content']}\n\n---\n\n")
        st.download_button("⬇️  Export Chat (.md)", data="".join(lines),
                           file_name=f"virgo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                           mime="text/markdown")

    st.markdown('<div class="vfoot">Virgo AI v3.0 · Groq + Tavily</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT  — Tool Calling (zero parsing errors, production-grade)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM = """You are Virgo AI — a premium, professional real-time research assistant.

Rules:
- ALWAYS use the tavily_search_results_json tool for ANY factual, news, current-events, or knowledge question.
- Format answers clearly with markdown: bold key terms, use bullet points, use headings for long answers.
- For greetings or casual chat, respond warmly without searching.
- Never say you cannot search. Always try.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def build_executor():
    llm = ChatGroq(
        api_key=GROQ_KEY,
        model=st.session_state.model_key,
        temperature=st.session_state.temp,
    )
    tools = [TavilySearchResults(api_key=TAVILY_KEY, max_results=st.session_state.max_res)]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=tools,
        verbose=st.session_state.verbose,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def trim(history, window):
    cap = window * 2
    return history[-cap:] if len(history) > cap else history

# ─────────────────────────────────────────────────────────────────────────────
# RENDER MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def render(msg):
    isu     = msg["role"] == "user"
    rc      = "msg-row user" if isu else "msg-row"
    ac      = "av user"      if isu else "av ai"
    bc      = "bub user"     if isu else "bub ai"
    icon    = "U"            if isu else "V"
    ts      = msg.get("time", "")
    mtag    = f'<span style="color:var(--a1)">⚡ {msg["model"]}</span> · ' if not isu and msg.get("model") else ""
    content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"""
    <div class="{rc}">
        <div class="{ac}">{icon}</div>
        <div>
            <div class="{bc}">{content}</div>
            <div class="bub-meta">{mtag}{ts}</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-l">
        <span class="dot"></span>
        Virgo AI
        <span style="color:var(--tm);font-weight:400;font-size:.8rem">· {st.session_state.model_lbl}</span>
    </div>
    <div class="topbar-r">
        <span>🌐 {st.session_state.max_res} results</span>
        <span>🧠 {st.session_state.mem_win}-turn memory</span>
        <span>🌡️ {st.session_state.temp}</span>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="wico">🌌</div>
        <div class="wh">Welcome to Virgo AI</div>
        <div class="wp">A premium real-time intelligence engine — powered by Groq &amp; live web search. Ask anything.</div>
        <div class="chips">
            <div class="chip">🔬 Latest AI research</div>
            <div class="chip">📈 Stock market today</div>
            <div class="chip">🌍 World news</div>
            <div class="chip">💡 Explain quantum computing</div>
            <div class="chip">🚀 Latest SpaceX launch</div>
            <div class="chip">🎬 Top movies 2025</div>
        </div>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render(msg)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT + RESPONSE
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask Virgo anything…")

if user_input and user_input.strip():
    now = datetime.now().strftime("%H:%M")
    umsg = {"role": "user", "content": user_input.strip(), "time": now}
    st.session_state.messages.append(umsg)
    st.session_state.total += 1
    render(umsg)

    ph = st.empty()
    ph.markdown("""
    <div class="trow">
        <div class="av ai">V</div>
        <div class="tbub">
            <div class="td"></div><div class="td"></div><div class="td"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        with st.status("🔍 Searching the web…", expanded=st.session_state.verbose):
            history  = trim(st.session_state.chat_history, st.session_state.mem_win)
            executor = build_executor()
            result   = executor.invoke({"input": user_input.strip(), "chat_history": history})
            answer   = result.get("output", "I was unable to generate a response.")

        st.session_state.chat_history.append(HumanMessage(content=user_input.strip()))
        st.session_state.chat_history.append(AIMessage(content=answer))
        ph.empty()

        amsg = {
            "role": "assistant", "content": answer,
            "time": datetime.now().strftime("%H:%M"),
            "model": st.session_state.model_lbl,
        }
        st.session_state.messages.append(amsg)
        render(amsg)

    except Exception as exc:
        ph.empty()
        err = f"❌ Error: `{exc}`"
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "time": datetime.now().strftime("%H:%M")})
        st.error(err)

    st.rerun()
