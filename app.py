import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory

# 1. ELITE PAGE CONFIGURATION
st.set_page_config(
    page_title="Virgo AI Engine",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PREMIUM CSS INJECTION (The "Look & Feel")
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0e1117);
        color: #e0e0e0;
    }
    
    /* Neon Gradient Title */
    .premium-header {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        margin-bottom: 0rem;
    }

    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 26, 37, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chat Bubble Customization */
    .stChatMessage {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: rgba(255, 255, 255, 0.02) !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR ENGINE CONTROLS
with st.sidebar:
    st.markdown("## 🌌 Virgo Control Center")
    st.divider()
    
    # Customization Sliders
    st.markdown("### 🛠️ Engine Tuning")
    creativity = st.slider("Creativity (Temp)", 0.0, 1.0, 0.7, 0.1)
    memory_k = st.number_input("Memory Depth (Messages)", 1, 20, 5)
    
    st.divider()
    
    # Model Selection
    target_model = st.selectbox("Neural Architecture", 
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"])
    
    if st.button("🗑️ Purge System Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.caption("Virgo Core: v2.5.0 Premium")

# 4. INITIALIZE CORE LOGIC
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
except KeyError:
    st.error("Missing Security Credentials in Secrets.")
    st.stop()

# Persistent Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=memory_k, 
        memory_key="chat_history", 
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Virgo AI Systems Online. Awaiting directive."}]

# 5. UI HEADER
st.markdown('<h1 class="premium-header">Virgo AI</h1>', unsafe_allow_html=True)
st.markdown("#### Real-Time Intelligence Engine")

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🌌" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# 6. THE BRAIN (Agent Construction)
llm = ChatGroq(api_key=groq_api_key, model=target_model, temperature=creativity)
search = TavilySearchResults(api_key=tavily_api_key)
tools = [search]

# Pulling a professional ReAct prompt from LangChain Hub
prompt_template = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True, 
    handle_parsing_errors=True
)

# 7. EXECUTION LAYER
if user_query := st.chat_input("Enter complex query..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🌌"):
        with st.status("🌌 Virgo Thinking...", expanded=False) as status:
            try:
                # Custom System Identity Instruction
                full_input = f"User asks: {user_query}. (Note: You are Virgo AI. If asked for your name, identify as Virgo AI.)"
                
                response = agent_executor.invoke({"input": full_input})
                output_text = response["output"]
                
                status.update(label="Analysis Complete", state="complete")
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                status.update(label="System Error", state="error")
                st.error(f"Internal Core Error: {str(e)}")
