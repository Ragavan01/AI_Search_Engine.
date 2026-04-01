import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory

# 1. GLOBAL SETTINGS & STYLING (Premium UX)
st.set_page_config(page_title="Virgo AI | Global Intelligence", page_icon="🌌", layout="wide")

st.markdown("""
<style>
    /* Dark Nebula Theme */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #1e293b 0%, #020617 100%);
        color: #f1f5f9;
    }
    
    /* Neon Header */
    .virgo-title {
        font-family: 'Inter', sans-serif;
        font-size: 55px;
        font-weight: 900;
        background: linear-gradient(135deg, #22d3ee 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -2px;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chat Styling */
    .stChatMessage {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(30, 41, 59, 0.3) !important;
        margin: 12px 0px;
    }
</style>
""", unsafe_allow_html=True)

# 2. VIRGO COMMAND CENTER (Sidebar)
with st.sidebar:
    st.markdown("## ⚙️ Virgo Command")
    st.divider()
    
    # Python 3.11 Optimized selection
    model_id = st.selectbox("Intelligence Core", 
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], 
        help="Select the neural architecture for Virgo AI.")
    
    human_feel = st.slider("Human Resonance", 0.0, 1.0, 0.8, 
        help="Higher values make responses more conversational and creative.")
    
    st.divider()
    if st.button("✨ Purge Neural Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    
    st.caption("Virgo Engine v3.5.1 | Python 3.11 Optimized")

# 3. BRAIN & MEMORY INITIALIZATION
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=10, 
        memory_key="chat_history", 
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Virgo AI Online. I have full access to global information. How can I assist you?"}
    ]

# Setup Agent Architecture
try:
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"], 
        model_name=model_id, 
        temperature=human_feel
    )
    search = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"])
    tools = [search]
    prompt_template = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=st.session_state.memory, 
        handle_parsing_errors=True,
        verbose=False
    )
except Exception as e:
    st.error(f"Hardware/API Link Failure: {e}")
    st.stop()

# 4. CHAT INTERFACE
st.markdown('<p class="virgo-title">Virgo AI</p>', unsafe_allow_html=True)

# Display History
for msg in st.session_state.messages:
    avatar = "🌌" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# User Interaction
if user_query := st.chat_input("Ask Virgo anything..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🌌"):
        with st.status("🌌 Scanning Global Networks...", expanded=False) as status:
            try:
                # Optimized Identity Injection
                identity_prompt = f"User: {user_query}. (Note: You are Virgo AI. Be human-like, expert, and professional.)"
                
                # Execute AI Logic
                result = agent_executor.invoke({"input": identity_prompt})
                full_text = result["output"]
                
                status.update(label="Analysis Complete", state="complete")
                
                # Gemini-style word streaming effect
                placeholder = st.empty()
                streamed_text = ""
                for word in full_text.split(" "):
                    streamed_text += word + " "
                    placeholder.markdown(streamed_text + "▌")
                    time.sleep(0.04) # Speed of human-like typing
                placeholder.markdown(streamed_text)
                
                st.session_state.messages.append({"role": "assistant", "content": full_text})
                
            except Exception as e:
                status.update(label="Core Failure", state="error")
                st.error(f"I encountered a system snag: {str(e)}")
