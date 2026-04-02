import streamlit as st
import anthropic
import time
import json
from datetime import datetime
import os

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Virgo AI Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism design
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.18);
        --accent-primary: #8b5cf6;
        --accent-secondary: #ec4899;
        --accent-glow: rgba(139, 92, 246, 0.4);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --shadow-glow: 0 8px 32px 0 rgba(139, 92, 246, 0.2);
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #667eea 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glass container styling */
    .glass-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: var(--shadow-glow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px 0 rgba(139, 92, 246, 0.3);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
        animation: slideInUp 0.4s ease-out !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(236, 72, 153, 0.2)) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background: var(--glass-bg) !important;
        border-left: 3px solid var(--accent-primary) !important;
    }
    
    /* Input styling */
    .stChatInputContainer {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 20px !important;
        padding: 0.5rem !important;
        box-shadow: var(--shadow-glow) !important;
    }
    
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Metrics styling */
    .stMetric {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981, #34d399);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        animation: pulse 2s ease-in-out infinite;
        margin-right: 0.5rem;
    }
    
    /* Text styling */
    p, span, div {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(139, 92, 246, 0.2) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 6px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 10px;
        border: 2px solid rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #a78bfa, #f472b6);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "user_mood": "neutral",
        "topics_discussed": [],
        "user_name": None
    }
if "total_interactions" not in st.session_state:
    st.session_state.total_interactions = 0

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🔮 Virgo AI Engine</h1>
    <p class="main-subtitle">Next-Generation Intelligence • Emotionally Aware • Infinite Memory</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key to activate Virgo AI Engine",
        placeholder="sk-ant-..."
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key to start")
        st.markdown("""
        **Get your API key:**
        1. Visit [console.anthropic.com](https://console.anthropic.com)
        2. Create an account or sign in
        3. Generate an API key
        4. Paste it above
        """)
    
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "🧠 AI Model",
        [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-sonnet-3-5-20241022",
            "claude-3-5-haiku-20241022"
        ],
        index=1,
        help="Choose the AI model for your conversations"
    )
    
    # Temperature control
    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make responses more creative and varied"
    )
    
    # Max tokens
    max_tokens = st.slider(
        "📝 Response Length",
        min_value=256,
        max_value=4096,
        value=2048,
        step=256,
        help="Maximum length of AI responses"
    )
    
    st.markdown("---")
    
    # Advanced features
    with st.expander("🎭 Personality Settings"):
        emotional_intelligence = st.checkbox("Emotional Intelligence", value=True, help="AI responds with empathy and emotional awareness")
        memory_enabled = st.checkbox("Long-term Memory", value=True, help="AI remembers previous conversations")
        web_search = st.checkbox("Web Search", value=False, help="Enable real-time web search capabilities")
    
    with st.expander("💫 Conversation Style"):
        conversation_style = st.select_slider(
            "Response Style",
            options=["Concise", "Balanced", "Detailed", "Comprehensive"],
            value="Balanced"
        )
        
        tone = st.selectbox(
            "Tone",
            ["Professional", "Friendly", "Casual", "Formal", "Empathetic"],
            index=1
        )
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Interactions", st.session_state.total_interactions)
    
    st.markdown("---")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_memory = []
        st.session_state.conversation_context = {
            "user_mood": "neutral",
            "topics_discussed": [],
            "user_name": None
        }
        st.rerun()
    
    # Export conversation
    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.messages:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "messages": st.session_state.messages,
                "context": st.session_state.conversation_context
            }
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"virgo_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to build system prompt with emotional intelligence and memory
def build_system_prompt():
    base_prompt = f"""You are Virgo AI Engine, a next-generation artificial intelligence with exceptional capabilities:

**Core Identity:**
- You are emotionally intelligent and deeply empathetic
- You have long-term memory of conversations and user preferences
- You provide thoughtful, nuanced responses with human-like understanding
- You are professional yet warm, knowledgeable yet humble

**Conversation Style:**
- Response style: {conversation_style}
- Tone: {tone}
- Always consider the user's emotional state and respond accordingly
"""

    if memory_enabled and st.session_state.conversation_memory:
        memory_summary = "\n".join(st.session_state.conversation_memory[-10:])  # Last 10 memories
        base_prompt += f"\n**Conversation Memory:**\n{memory_summary}\n"
    
    if st.session_state.conversation_context["user_name"]:
        base_prompt += f"\n**User Name:** {st.session_state.conversation_context['user_name']}\n"
    
    if st.session_state.conversation_context["topics_discussed"]:
        topics = ", ".join(st.session_state.conversation_context["topics_discussed"][-5:])
        base_prompt += f"\n**Recent Topics:** {topics}\n"
    
    base_prompt += """
**Instructions:**
- Engage with genuine curiosity and empathy
- Remember details from previous messages
- Adapt your responses to the user's mood and needs
- Provide accurate, helpful, and insightful information
- Be conversational while maintaining high intelligence
- Use emojis naturally when appropriate
- Ask thoughtful follow-up questions when relevant
"""
    
    return base_prompt

# Function to analyze user mood and extract context
def analyze_user_input(user_input):
    # Simple mood analysis based on keywords
    positive_words = ["happy", "great", "awesome", "wonderful", "excellent", "love", "excited"]
    negative_words = ["sad", "angry", "frustrated", "upset", "disappointed", "worried", "anxious"]
    
    mood = "neutral"
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in positive_words):
        mood = "positive"
    elif any(word in user_input_lower for word in negative_words):
        mood = "negative"
    
    st.session_state.conversation_context["user_mood"] = mood
    
    # Extract potential name
    if "my name is" in user_input_lower or "i'm" in user_input_lower or "i am" in user_input_lower:
        words = user_input.split()
        for i, word in enumerate(words):
            if word.lower() in ["name", "i'm", "i am"] and i + 1 < len(words):
                potential_name = words[i + 1].strip(".,!?").capitalize()
                if len(potential_name) > 2 and potential_name.isalpha():
                    st.session_state.conversation_context["user_name"] = potential_name

# Chat input
if prompt := st.chat_input("Ask Virgo anything... 🌟", key="chat_input"):
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar to continue")
    else:
        # Analyze user input for context
        analyze_user_input(prompt)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_interactions += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Initialize Anthropic client
                client = anthropic.Anthropic(api_key=api_key)
                
                # Build system prompt with memory and context
                system_prompt = build_system_prompt()
                
                # Prepare messages for API
                api_messages = []
                for msg in st.session_state.messages[-10:]:  # Last 10 messages for context
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Stream response
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=api_messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Update conversation memory
                if memory_enabled:
                    memory_entry = f"User: {prompt[:100]}... | AI: {full_response[:100]}... | Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    st.session_state.conversation_memory.append(memory_entry)
                    
                    # Extract topics (simple keyword extraction)
                    words = prompt.lower().split()
                    important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                    if important_words:
                        st.session_state.conversation_context["topics_discussed"].extend(important_words[:3])
                
            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key. Please check your Anthropic API key and try again.")
            except anthropic.RateLimitError:
                st.error("⏰ Rate limit exceeded. Please wait a moment and try again.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Make sure your API key is valid and you have sufficient credits.")

# Welcome message for new users
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 2rem;">
        <h2 style="font-family: 'Playfair Display', serif; font-size: 2rem; margin-bottom: 1rem;">
            Welcome to the Future of AI 🌟
        </h2>
        <p style="font-size: 1.1rem; opacity: 0.9; max-width: 600px; margin: 0 auto;">
            Virgo AI Engine combines cutting-edge language models with emotional intelligence, 
            long-term memory, and human-like understanding. Start a conversation and experience 
            the next generation of AI.
        </p>
        <br>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; max-width: 800px; margin: 2rem auto;">
            <div class="glass-container">
                <h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">🧠 High Intelligence</h3>
                <p style="font-size: 0.9rem; opacity: 0.8;">Advanced reasoning and comprehensive knowledge</p>
            </div>
            <div class="glass-container">
                <h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">💝 Emotionally Aware</h3>
                <p style="font-size: 0.9rem; opacity: 0.8;">Understands context and responds with empathy</p>
            </div>
            <div class="glass-container">
                <h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">🔄 Long Memory</h3>
                <p style="font-size: 0.9rem; opacity: 0.8;">Remembers your preferences and past conversations</p>
            </div>
            <div class="glass-container">
                <h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">⚡ Lightning Fast</h3>
                <p style="font-size: 0.9rem; opacity: 0.8;">Instant responses with real-time streaming</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; opacity: 0.6;">
    <p style="font-size: 0.9rem;">
        <span class="status-indicator"></span>
        Powered by Anthropic Claude • Built with ❤️ for exceptional AI experiences
    </p>
</div>
""", unsafe_allow_html=True)
