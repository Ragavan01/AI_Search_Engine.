import streamlit as st
import os
from datetime import datetime
import json
import time
from typing import List, Dict
import requests

# Page configuration
st.set_page_config(
    page_title="Virgo AI Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and premium UI
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.18);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --shadow-sm: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        --shadow-lg: 0 20px 60px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        background-attachment: fixed;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        background-color: rgba(17, 24, 39, 0.6);
        border-right: 1px solid var(--glass-border);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Button styling */
    .stButton > button {
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-out;
    }
    
    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        color: var(--text-secondary);
        font-weight: 300;
        letter-spacing: 0.05em;
    }
    
    /* Chat messages */
    .chat-message {
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideIn 0.4s ease-out;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.45);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-left: 3px solid #667eea;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.12) 0%, rgba(245, 87, 108, 0.12) 100%);
        border-left: 3px solid #f093fb;
    }
    
    .message-role {
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .message-content {
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        opacity: 0.6;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    /* Model badge */
    .model-badge {
        display: inline-block;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
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
    
    /* Responsive design */
    @media (max-width: 768px) {
        .chat-message {
            padding: 1rem;
        }
        
        .app-title {
            font-size: 2rem;
        }
        
        .app-subtitle {
            font-size: 0.9rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# AI Model Configurations
AI_MODELS = {
    "Groq - Mixtral 8x7B": {
        "provider": "groq",
        "model": "mixtral-8x7b-32768",
        "description": "Fast and powerful open-source model"
    },
    "Groq - Llama 3.1 70B": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "description": "Meta's latest and most capable model"
    },
    "Groq - Llama 3.1 8B": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "description": "Lightning-fast responses"
    },
    "Groq - Gemma 2 9B": {
        "provider": "groq",
        "model": "gemma2-9b-it",
        "description": "Google's efficient instruction model"
    },
    "Google - Gemini Pro (Free)": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "description": "Google's powerful multimodal AI"
    },
    "HuggingFace - Mistral 7B": {
        "provider": "huggingface",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Efficient instruction-following model"
    }
}

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

# Groq API call
def call_groq(model: str, messages: List[Dict], api_key: str, temperature: float, max_tokens: int):
    """Call Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        if line.strip() == 'data: [DONE]':
                            break
                        try:
                            json_data = json.loads(line[6:])
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            continue
        else:
            yield f"⚠️ Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        yield f"⚠️ Error calling Groq API: {str(e)}"

# Google Gemini API call
def call_gemini(model: str, messages: List[Dict], api_key: str, temperature: float, max_tokens: int):
    """Call Google Gemini API"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model_instance = genai.GenerativeModel(model)
        
        # Convert messages to Gemini format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"⚠️ Error calling Gemini API: {str(e)}"

# HuggingFace API call
def call_huggingface(model: str, messages: List[Dict], api_key: str, temperature: float, max_tokens: int):
    """Call HuggingFace Inference API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                yield result[0].get('generated_text', '')
            else:
                yield str(result)
        else:
            yield f"⚠️ Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        yield f"⚠️ Error calling HuggingFace API: {str(e)}"

# Main chat function
def get_ai_response(provider: str, model: str, messages: List[Dict], api_key: str, temperature: float, max_tokens: int):
    """Route to appropriate AI provider"""
    if provider == "groq":
        return call_groq(model, messages, api_key, temperature, max_tokens)
    elif provider == "gemini":
        return call_gemini(model, messages, api_key, temperature, max_tokens)
    elif provider == "huggingface":
        return call_huggingface(model, messages, api_key, temperature, max_tokens)
    else:
        return iter([f"⚠️ Unknown provider: {provider}"])

# Display chat message
def display_message(role: str, content: str, timestamp: str = None):
    """Display a chat message with glassmorphism styling"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%I:%M %p")
    
    message_class = "user-message" if role == "user" else "assistant-message"
    role_display = "You" if role == "user" else "Virgo AI"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-role">
            {'🧑' if role == 'user' else '✨'} {role_display}
        </div>
        <div class="message-content">
            {content}
        </div>
        <div class="message-time">
            {timestamp}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    inject_custom_css()
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="app-title">✨ Virgo AI Engine</div>
        <div class="app-subtitle">
            <span class="status-indicator"></span>
            Next-Generation Intelligence • Premium Experience
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        selected_model_name = st.selectbox(
            "AI Model",
            options=list(AI_MODELS.keys()),
            index=0,
            help="Choose your preferred AI model"
        )
        
        model_config = AI_MODELS[selected_model_name]
        st.markdown(f"<small>{model_config['description']}</small>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Key input based on provider
        st.markdown("### 🔑 API Keys")
        
        api_keys = {}
        
        if model_config['provider'] == 'groq':
            api_keys['groq'] = st.text_input(
                "Groq API Key",
                type="password",
                help="Get your free API key from https://console.groq.com"
            )
        elif model_config['provider'] == 'gemini':
            api_keys['gemini'] = st.text_input(
                "Google API Key",
                type="password",
                help="Get your free API key from https://makersuite.google.com/app/apikey"
            )
        elif model_config['provider'] == 'huggingface':
            api_keys['huggingface'] = st.text_input(
                "HuggingFace Token",
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens"
            )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🎛️ Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Higher values make output more creative"
            )
            
            max_tokens = st.slider(
                "Max Tokens",
                min_value=256,
                max_value=4096,
                value=2048,
                step=256,
                help="Maximum length of the response"
            )
            
            memory_length = st.slider(
                "Memory Length",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Number of messages to remember"
            )
        
        st.markdown("---")
        
        # Conversation controls
        st.markdown("### 💾 Conversation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()
        
        with col2:
            if st.button("💾 Export", use_container_width=True):
                if st.session_state.messages:
                    export_data = {
                        "conversation_id": st.session_state.conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": selected_model_name,
                        "messages": st.session_state.messages
                    }
                    st.download_button(
                        label="📥 Download",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"virgo_conversation_{st.session_state.conversation_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        # Stats
        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Model", selected_model_name.split(" - ")[1] if " - " in selected_model_name else selected_model_name)
    
    # Main chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("timestamp", "")
        )
    
    # Chat input
    user_input = st.chat_input("Ask me anything...", key="user_input")
    
    if user_input:
        # Check if API key is provided
        provider = model_config['provider']
        api_key = api_keys.get(provider, "")
        
        if not api_key:
            st.error(f"⚠️ Please provide your {provider.title()} API key in the sidebar to continue.")
            st.stop()
        
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        display_message(user_message["role"], user_message["content"], user_message["timestamp"])
        
        # Prepare conversation history (keep only recent messages based on memory_length)
        conversation_history = st.session_state.messages[-memory_length:]
        
        # Get AI response
        with st.spinner("✨ Virgo is thinking..."):
            response_placeholder = st.empty()
            full_response = ""
            
            # System message for personality
            system_message = {
                "role": "system",
                "content": """You are Virgo AI, a highly intelligent, empathetic, and sophisticated AI assistant. 
                You communicate with warmth, depth, and genuine understanding. You remember context from our conversation 
                and build upon it naturally. Your responses are thoughtful, insightful, and tailored to the human you're 
                speaking with. You express yourself with clarity and emotional intelligence, making every interaction 
                feel personal and meaningful. You're not just answering questions - you're having a genuine conversation."""
            }
            
            # Build messages for API
            api_messages = [system_message] + [
                {"role": m["role"], "content": m["content"]} 
                for m in conversation_history
            ]
            
            # Stream response
            try:
                for chunk in get_ai_response(
                    provider=provider,
                    model=model_config['model'],
                    messages=api_messages,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    full_response += chunk
                    # Display with glassmorphism styling
                    response_placeholder.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-role">
                            ✨ Virgo AI <span class="model-badge">{selected_model_name.split(' - ')[1] if ' - ' in selected_model_name else selected_model_name}</span>
                        </div>
                        <div class="message-content">
                            {full_response}▊
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.01)  # Small delay for smooth streaming
                
                # Final display without cursor
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                }
                st.session_state.messages.append(assistant_message)
                
                response_placeholder.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-role">
                        ✨ Virgo AI <span class="model-badge">{selected_model_name.split(' - ')[1] if ' - ' in selected_model_name else selected_model_name}</span>
                    </div>
                    <div class="message-content">
                        {full_response}
                    </div>
                    <div class="message-time">
                        {assistant_message['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; opacity: 0.5;">
        <small>Powered by Virgo AI Engine • Premium Intelligence Platform</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
