import streamlit as st
import os
from groq import Groq
from tavily import TavilyClient
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Virgo AI Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium glassmorphism design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --primary-glow: rgba(99, 102, 241, 0.4);
        --secondary-glow: rgba(168, 85, 247, 0.4);
        --accent-color: #6366f1;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --glass-bg: rgba(17, 24, 39, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --shadow-glow: 0 8px 32px 0 rgba(99, 102, 241, 0.15);
    }
    
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background-image: 
            radial-gradient(circle at 20% 50%, var(--primary-glow) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, var(--secondary-glow) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.3) 0%, transparent 50%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Header styling */
    .virgo-header {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        margin-bottom: 2rem;
    }
    
    .virgo-logo {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { 
            filter: drop-shadow(0 0 20px var(--primary-glow));
        }
        to { 
            filter: drop-shadow(0 0 40px var(--secondary-glow));
        }
    }
    
    .virgo-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .virgo-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 300;
        font-family: 'Space Mono', monospace;
    }
    
    /* Glass container */
    .glass-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 900px;
        box-shadow: var(--shadow-glow);
        transition: all 0.3s ease;
    }
    
    .glass-container:hover {
        box-shadow: 0 8px 40px 0 rgba(99, 102, 241, 0.25);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Chat messages */
    .chat-message {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, #14b8a6, #06b6d4);
    }
    
    .message-content {
        flex: 1;
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .message-content p {
        margin: 0 0 0.5rem 0;
    }
    
    .message-content p:last-child {
        margin-bottom: 0;
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-family: 'Space Mono', monospace;
    }
    
    /* Search indicator */
    .search-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        color: #60a5fa;
        font-family: 'Space Mono', monospace;
    }
    
    .search-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #60a5fa;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    
    /* Stats card */
    .stats-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .stats-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 0.25rem;
    }
    
    .stats-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-family: 'Space Mono', monospace;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .virgo-title {
            font-size: 2rem;
        }
        
        .virgo-logo {
            font-size: 3rem;
        }
        
        .glass-container {
            padding: 1.25rem;
            margin: 0.5rem;
            border-radius: 16px;
        }
        
        .chat-message {
            gap: 0.75rem;
        }
        
        .avatar {
            width: 36px;
            height: 36px;
            font-size: 1rem;
        }
        
        .message-content {
            padding: 0.875rem 1rem;
            font-size: 0.95rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0

# Initialize API clients
def init_clients():
    groq_api_key = os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY', '')
    tavily_api_key = os.getenv('TAVILY_API_KEY') or st.secrets.get('TAVILY_API_KEY', '')
    
    groq_client = None
    tavily_client = None
    
    if groq_api_key:
        try:
            groq_client = Groq(api_key=groq_api_key)
        except Exception as e:
            st.error(f"Error initializing Groq: {e}")
    
    if tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=tavily_api_key)
        except Exception as e:
            st.error(f"Error initializing Tavily: {e}")
    
    return groq_client, tavily_client

groq_client, tavily_client = init_clients()

# Header
st.markdown("""
<div class="virgo-header">
    <div class="virgo-logo">🔮</div>
    <h1 class="virgo-title">Virgo AI Engine</h1>
    <p class="virgo-subtitle">Next-generation intelligent search & conversation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    model_choice = st.selectbox(
        "AI Model",
        ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"],
        help="Choose your preferred AI model"
    )
    
    enable_search = st.checkbox("Enable Web Search", value=True, help="Augment responses with real-time web search")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1, help="Higher values = more creative responses")
    max_tokens = st.slider("Response Length", 512, 4096, 2048, 256)
    
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-value">{st.session_state.conversation_count}</div>
        <div class="stats-label">Conversations</div>
    </div>
    <div class="stats-card">
        <div class="stats-value">{st.session_state.search_count}</div>
        <div class="stats-label">Web Searches</div>
    </div>
    <div class="stats-card">
        <div class="stats-value">{len(st.session_state.messages)}</div>
        <div class="stats-label">Messages</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.session_state.search_count = 0
        st.rerun()

# Main chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        has_search = message.get("has_search", False)
        
        avatar_class = "user-avatar" if role == "user" else "ai-avatar"
        avatar_emoji = "👤" if role == "user" else "🔮"
        
        message_html = f"""
        <div class="chat-message">
            <div class="avatar {avatar_class}">{avatar_emoji}</div>
            <div class="message-content">
                {"<div class='search-indicator'><div class='search-pulse'></div>Web Search Active</div>" if has_search else ""}
                {content}
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """
        st.markdown(message_html, unsafe_allow_html=True)

# Input area
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Ask me anything... I can search the web and provide intelligent responses",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send", use_container_width=True)

# Process user input
if send_button and user_input:
    if not groq_client:
        st.error("⚠️ Please configure your GROQ_API_KEY in Streamlit secrets or environment variables")
    else:
        # Add user message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Prepare context from conversation history
        context_messages = []
        for msg in st.session_state.messages[-10:]:  # Keep last 10 messages for context
            context_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Web search if enabled
        search_results = ""
        has_search = False
        
        if enable_search and tavily_client:
            try:
                with st.spinner("🔍 Searching the web..."):
                    search_response = tavily_client.search(
                        query=user_input,
                        search_depth="advanced",
                        max_results=5
                    )
                    
                    if search_response.get('results'):
                        has_search = True
                        st.session_state.search_count += 1
                        
                        search_results = "\n\nWeb Search Results:\n"
                        for idx, result in enumerate(search_response['results'][:3], 1):
                            search_results += f"\n{idx}. {result.get('title', 'No title')}\n"
                            search_results += f"   {result.get('content', 'No content')}\n"
                            search_results += f"   Source: {result.get('url', 'No URL')}\n"
            except Exception as e:
                st.warning(f"Search failed: {e}")
        
        # Generate AI response
        try:
            with st.spinner("🤔 Thinking..."):
                # Build the final prompt
                system_prompt = """You are Virgo AI Engine, a sophisticated and emotionally intelligent AI assistant. 
You communicate with warmth, empathy, and deep understanding. Your responses are:
- Thoughtful and comprehensive
- Emotionally aware and human-like
- Clear and well-structured
- Backed by accurate information
- Conversational yet professional

When web search results are provided, integrate them naturally into your response and cite sources."""

                if search_results:
                    context_messages[-1]["content"] += search_results
                
                # Generate response
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *context_messages
                    ],
                    model=model_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                ai_response = chat_completion.choices[0].message.content
                
                # Add AI response
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": timestamp,
                    "has_search": has_search
                })
                
                st.session_state.conversation_count += 1
                
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
        
        st.rerun()

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: var(--text-secondary); font-size: 0.85rem; font-family: "Space Mono", monospace;'>
    Powered by Groq AI • Enhanced with Tavily Search • Built with ❤️
</div>
""", unsafe_allow_html=True)
