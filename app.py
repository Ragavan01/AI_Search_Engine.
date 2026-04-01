import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory

# -----------------------------------------
# 1. PREMIUM PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(page_title="Virgo AI Engine", page_icon="🌌", layout="wide")

# -----------------------------------------
# 2. CUSTOM CSS FOR HIGH-END UI (Glassmorphism & Dark Theme)
# -----------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sleek Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161A25;
        border-right: 1px solid #2B3040;
    }
    
    /* Custom Chat Message Bubbles */
    .stChatMessage {
        background-color: #1E2330;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2B3040;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Glowing Title */
    .premium-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    /* Subtitle styling */
    .premium-subtitle {
        color: #8B949E;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 3. ADVANCED SIDEBAR CONTROLS
# -----------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")
    
    # Customization: Let the user control the AI's creativity
    temperature = st.slider("Brain Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Higher values make the AI more creative, lower values make it more focused and factual.")
    
    # Customization: Model Selector
    selected_model = st.selectbox("Select Core Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"])
    
    st.divider()
    
    # Feature: Clear Memory Button
    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.caption("System Status: Online 🟢")

# -----------------------------------------
# 4. MAIN UI HEADER
# -----------------------------------------
st.markdown('<p class="premium-title">Virgo AI</p>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Advanced Real-Time Search & Intelligence</p>', unsafe_allow_html=True)

# -----------------------------------------
# 5. INITIALIZE SECRETS & MEMORY
# -----------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
except KeyError:
    st.error("API Keys missing! Please check Streamlit Advanced Settings.")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome. I am Virgo AI. How can I assist you with your research today?"}]

# -----------------------------------------
# 6. DISPLAY CHAT HISTORY WITH CUSTOM AVATARS
# -----------------------------------------
for message in st.session_state.messages:
    avatar = "🌌" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# -----------------------------------------
# 7. AI AGENT SETUP
# -----------------------------------------
# The AI now updates instantly based on sidebar slider & selectbox!
llm = ChatGroq(api_key=groq_api_key, model=selected_model, temperature=temperature, streaming=True)
search_tool = TavilySearchResults(api_key=tavily_api_key)

system_prompt = """You are Virgo AI, a highly advanced, premium AI search engine. 
You are intelligent, extremely helpful, and speak with a professional, sleek tone. 
If a user asks 'What is your name?' or 'Who are you?', you must answer 'I am Virgo AI, your premium search engine.'
Use your search tool to find real-time information when needed, but answer general conversational questions from your own knowledge naturally."""

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    agent_kwargs={"system_message": system_prompt},
    verbose=True
)

# -----------------------------------------
# 8. USER INPUT & RESPONSE GENERATION
# -----------------------------------------
if prompt := st.chat_input("Enter your query here..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # AI Response
    with st.chat_message("assistant", avatar="🌌"):
        with st.spinner("Processing data..."):
            try:
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"System Error: {e}")
