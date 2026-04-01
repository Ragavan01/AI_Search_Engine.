import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory

# 1. PREMIUM PAGE SETUP
st.set_page_config(page_title="Virgo AI | Global Intelligence", page_icon="🌌", layout="wide")

# 2. CYBER-GLASS UI DESIGN (CSS)
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    .main-title {
        font-size: 50px;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px !important;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# 3. CORE SYSTEM IDENTITY
with st.sidebar:
    st.markdown("### 🌌 Virgo Control Center")
    st.divider()
    model_name = st.selectbox("Intelligence Core", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    creativity = st.slider("Human Resonance (Temp)", 0.0, 1.0, 0.7)
    
    if st.button("Purge Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    st.caption("Virgo AI v3.0 | Global Search Active")

# 4. INITIALIZE AGENT & MEMORY
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System online. I am Virgo AI. How can I assist your global research today?"}]

# Setup Tools & LLM
llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model=model_name, temperature=creativity)
search = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"])
tools = [search]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, handle_parsing_errors=True)

# 5. PREMIUM CHAT INTERFACE
st.markdown('<p class="main-title">Virgo AI Engine</p>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌌" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("Ask anything about the world..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🌌"):
        with st.status("🌌 Accessing Global Data...", expanded=False) as status:
            try:
                # Direct instruction for Gemini-like personality
                system_instruction = f"User: {prompt_input}. (Note: You are Virgo AI. Provide comprehensive, human-like, world-class information.)"
                
                response = agent_executor.invoke({"input": system_instruction})
                full_response = response["output"]
                
                status.update(label="Analysis Complete", state="complete")
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                status.update(label="Core Snag", state="error")
                st.error(f"I encountered a small hiccup: {str(e)}")
