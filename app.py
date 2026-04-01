import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory
# 1. Premium UI Setup
st.set_page_config(page_title="Virgo AI Engine", page_icon="🌌", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .premium-title { font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="premium-title">Virgo AI</p>', unsafe_allow_html=True)

# 2. API Keys
groq_api_key = st.secrets["GROQ_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]

# 3. Memory & History
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am Virgo AI. How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 4. The Brain (Simplified Agent)
MODEL_NAME = "llama-3.3-70b-versatile"
MEMORY_WINDOW = 5
INITIAL_MESSAGE = "I am Virgo AI. How can I help you today?"
PROMPT_TEMPLATE = "hwchase17/react"

# 5. Chat Logic
if user_input := st.chat_input("Ask Virgo AI..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # We use a try/except block so the whole app doesn't crash if search fails
            try:
                response = agent_executor.invoke({"input": user_input})
                final_text = response["output"]
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error("I hit a small snag. Try asking again!")
