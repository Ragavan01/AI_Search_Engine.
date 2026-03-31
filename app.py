import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. Page Configuration
st.set_page_config(page_title="AI Search Hub", page_icon="🌐", layout="wide")

# 2. Access Keys from Secrets
groq_key = st.secrets["GROQ_API_KEY"]
tavily_key = st.secrets["TAVILY_API_KEY"]

# 3. Initialize AI Tools
search_tool = TavilySearchResults(api_key=tavily_key, k=5)
llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile", temperature=0)

# 4. The Main UI
st.title("🔍 AI Real-Time Search")
query = st.text_input("What do you want to find?", placeholder="Search the web...")

if query:
    with st.spinner("Searching the web..."):
        search_context = search_tool.run(query)
        prompt = f"Context: {search_context}\n\nQuestion: {query}\n\nAnswer in detail with bullet points:"
        response = llm.invoke(prompt)
        st.markdown("### 🤖 AI Research Summary")
        st.write(response.content)