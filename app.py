import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------
# 1. ELITE UI CONFIGURATION (Dark Mode & Glassmorphism)
# ---------------------------------------------------------
st.set_page_config(page_title="Virgo AI Engine", page_icon="🌌", layout="wide")

st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background: radial-gradient(circle at top right, #0F172A, #020617);
        color: #F8FAFC;
    }

    /* Neon Gradient Title */
    .premium-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(to right, #22D3EE, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Glass Effect Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chat Bubbles */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Virgo Control")
    st.divider()

    temp = st.slider("Neural Creativity", 0.0, 1.0, 0.7, 0.1)
    model_choice = st.selectbox(
        "Core Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )

    st.divider()
    if st.button("🗑️ Clear Neural Memory", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Virgo AI Online. How may I assist your research today?"}
        ]
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=False,  # FIX: must be False for ReAct string prompt
            input_key="input",
            output_key="output"
        )
        st.rerun()

    st.caption("System Status: Optimal 🟢")

# ---------------------------------------------------------
# 3. API KEY VALIDATION
# ---------------------------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"Missing API Key in Streamlit Secrets: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
# FIX: return_messages=False for ReAct string-based prompt compatibility
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output"
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Virgo AI Online. How may I assist your research today?"}
    ]

# ---------------------------------------------------------
# 5. HEADER
# ---------------------------------------------------------
st.markdown('<h1 class="premium-title">Virgo AI</h1>', unsafe_allow_html=True)
st.markdown("##### Premium Real-Time Intelligence Engine")

# ---------------------------------------------------------
# 6. BRAIN INITIALIZATION (Rebuilt every render with current settings)
# ---------------------------------------------------------
# FIX: Use a local ReAct prompt instead of hub.pull() to avoid
#      network dependency on LangSmith and hub authentication errors.
REACT_PROMPT_TEMPLATE = """You are Virgo AI, a premium professional intelligence assistant. 
Answer questions using the provided tools when real-time or external data is needed.
Be concise, accurate, and high-end in your responses.

You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    template=REACT_PROMPT_TEMPLATE,
)

@st.cache_resource(show_spinner=False)
def build_tools(tavily_key: str):
    """Cache the tool list so it isn't rebuilt on every rerender."""
    return [TavilySearchResults(api_key=tavily_key, max_results=5)]

tools = build_tools(tavily_api_key)

# FIX: LLM is NOT cached — it must respect the sidebar sliders (temp / model)
llm = ChatGroq(api_key=groq_api_key, model=model_choice, temperature=temp)

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# FIX: Pass memory correctly and set output_keys so memory knows what to save
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,          # FIX: prevent infinite loops
    return_intermediate_steps=False,
)

# ---------------------------------------------------------
# 7. CHAT INTERFACE
# ---------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🌌" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Enter your query..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🌌"):
        with st.status("🌌 Virgo Analyzing Data...", expanded=False) as status:
            try:
                # FIX: Pass only "input" key — memory variables are injected
                #      automatically by AgentExecutor via the memory object.
                result = agent_executor.invoke({"input": user_input})
                response_text = result.get("output", "I could not generate a response.")

                status.update(label="✅ Analysis Complete", state="complete")
                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

            except ValueError as ve:
                # Handles LLM output parsing failures gracefully
                status.update(label="⚠️ Parsing Error", state="error")
                st.warning(f"The model returned an unexpected format. Try rephrasing.\n\n`{ve}`")

            except Exception as e:
                status.update(label="❌ Core Failure", state="error")
                st.error(f"System Error: {str(e)}")
