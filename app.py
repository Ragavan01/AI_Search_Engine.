import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Virgo AI", page_icon="🌌", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stSidebar { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR / SETTINGS ---
with st.sidebar:
    st.title("🌌 Virgo AI Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    st.divider()
    model_option = st.selectbox("Choose Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"])
    temp = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.5)
    max_results = st.slider("Web Search Depth", 1, 10, 5)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HEADER ---
st.title("Virgo AI: Real-Time Intelligence")
st.caption("Powered by Groq & Tavily Search")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CORE LOGIC ---
if prompt := st.chat_input("Ask Virgo anything..."):
    if not groq_api_key or not tavily_api_key:
        st.error("Please provide both API keys in the sidebar.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. Initialize Tools & LLM
            search = TavilySearchResults(api_wrapper_kwargs={'tavily_api_key': tavily_api_key}, k=max_results)
            llm = ChatGroq(api_key=groq_api_key, model_name=model_option, temperature=temp)

            # 2. Perform Web Search (The "Search Engine" part)
            with st.spinner("Searching the live web..."):
                search_data = search.run(prompt)
            
            # 3. Construct the RAG Prompt
            context = "\n".join([f"Source: {d['url']}\nContent: {d['content']}" for d in search_data])
            
            messages = [
                SystemMessage(content=f"You are Virgo AI, a high-end research assistant. Use the following web context to answer the user accurately. Cite your sources using [Source Name](URL). \n\nContext: {context}"),
                *[HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            ]

            # 4. Stream the Response
            for chunk in llm.stream(messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
