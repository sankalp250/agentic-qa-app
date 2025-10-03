"""
Streamlit Frontend for AI Q&A Bot
A web interface for the LangGraph-powered Q&A bot using Groq API
"""

import streamlit as st
import os
from typing import List, Dict
from agent.simple_agent import SimpleQAAgent


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False


def check_api_key() -> bool:
    """Check if API key is configured"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    return groq_api_key is not None and groq_api_key != "your_groq_api_key_here"


def setup_sidebar():
    """Setup the sidebar with configuration options"""
    st.sidebar.title("🔧 Configuration")
    
    # API Key Status
    if check_api_key():
        st.sidebar.success("✅ Groq API Key Configured")
        st.session_state.api_key_configured = True
    else:
        st.sidebar.error("❌ Groq API Key Not Found")
        st.sidebar.info("""
        **Setup Instructions:**
        1. Get a free API key from [Groq Console](https://console.groq.com/)
        2. Create a `.env` file in the project root
        3. Add: `GROQ_API_KEY=your_actual_api_key`
        4. Restart the Streamlit app
        """)
        st.session_state.api_key_configured = False
    
    # Model Selection
    st.sidebar.subheader("🤖 Model Settings")
    model_options = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        model_options,
        index=0,
        help="Select the Groq model for generating responses"
    )
    
    # Clear Chat Button
    st.sidebar.subheader("💬 Chat Controls")
    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()
    
    # Show message count
    message_count = len(st.session_state.messages)
    st.sidebar.caption(f"Messages in chat: {message_count}")
    
    return selected_model


def initialize_agent(model_name: str) -> SimpleQAAgent:
    """Initialize the Q&A agent with the selected model"""
    try:
        if st.session_state.agent is None:
            st.session_state.agent = SimpleQAAgent(model_name=model_name)
        return st.session_state.agent
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        st.stop()


def display_chat_messages():
    """Display all chat messages in the main area"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_input(agent: SimpleQAAgent):
    """Handle user input and generate response"""
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.ask(prompt)
                    st.write(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })


def show_example_questions():
    """Show example questions that users can click"""
    st.subheader("💡 Try These Example Questions:")
    
    examples = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms",
        "How does machine learning work?",
        "What are the benefits of renewable energy?",
        "Explain the difference between Python lists and tuples"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                # Add the example question to chat
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="AI Q&A Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get configuration
    selected_model = setup_sidebar()
    
    # Main content area
    st.title("🤖 AI Q&A Bot")
    st.markdown("*Powered by LangGraph + Groq API*")
    
    # Check if API key is configured
    if not st.session_state.api_key_configured:
        st.warning("⚠️ Please configure your Groq API key to start chatting.")
        st.info("""
        **Quick Setup:**
        1. Get a free API key from [Groq Console](https://console.groq.com/)
        2. Create a `.env` file in the project root directory
        3. Add your API key: `GROQ_API_KEY=your_actual_api_key`
        4. Restart this Streamlit app
        """)
        
        # Show example of .env file
        st.code("GROQ_API_KEY=gsk_your_actual_groq_api_key_here", language="bash")
        
        st.stop()
    
    # Initialize the agent
    agent = initialize_agent(selected_model)
    
    # Show example questions if no messages yet
    if len(st.session_state.messages) == 0:
        show_example_questions()
        st.markdown("---")
    
    # Display existing chat messages
    display_chat_messages()
    
    # Handle new user input
    handle_user_input(agent)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>
                Built with ❤️ using LangGraph, Streamlit, and Groq API | 
                <a href='https://github.com/langchain-ai/langgraph' target='_blank'>LangGraph</a> | 
                <a href='https://console.groq.com/' target='_blank'>Groq</a>
            </small>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
