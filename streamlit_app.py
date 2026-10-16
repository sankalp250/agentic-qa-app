"""
Streamlit Frontend for the Proactive AI Agent
"""
import os
import streamlit as st
from datetime import datetime

# --- Sync Streamlit Secrets to os.environ ---
if hasattr(st, "secrets"):
    for key in ["GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL"]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = str(st.secrets[key])

from agent.proactive_agent import ProactiveAgent, AVAILABLE_GROQ_MODELS, DEFAULT_MODEL

# --- Page Configuration ---
st.set_page_config(
    page_title="Proactive AI Agent",
    page_icon="🧠",
    layout="wide",
)

# --- Agent Initialization ---
@st.cache_resource
def get_agent(topic, model_name):
    """Loads and caches the ProactiveAgent for a topic and model."""
    try:
        agent = ProactiveAgent(topic=topic, model_name=model_name)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}", icon="🚨")
        return None

# --- Session State ---
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Proactive AI Agent")
    st.markdown("This agent autonomously researches a topic to build a persistent knowledge base.")
    
    st.divider()
    st.subheader("Configuration")
    agent_topic = st.text_input(
        "Agent Research Topic", 
        "rag",
        help="The topic the agent will research (e.g., 'rag', 'multimodal AI', 'quantum computing')."
    )

    selected_model = st.selectbox(
        "Groq LLM Model",
        AVAILABLE_GROQ_MODELS,
        index=0,
        help="Active chat models available on Groq API."
    )

    st.subheader("API Keys (Optional Override)")
    groq_input = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    tavily_input = st.text_input("Tavily API Key", value=os.environ.get("TAVILY_API_KEY", ""), type="password")
    if groq_input:
        os.environ["GROQ_API_KEY"] = groq_input
    if tavily_input:
        os.environ["TAVILY_API_KEY"] = tavily_input

    # Get cached agent instance
    agent = None
    if os.environ.get("GROQ_API_KEY") and os.environ.get("TAVILY_API_KEY"):
        agent = get_agent(agent_topic, selected_model)

    st.divider()
    st.subheader("Knowledge Base Control")
    
    current_doc_count = agent.get_document_count() if agent else 0
    st.metric(label="Indexed Knowledge Chunks", value=current_doc_count)
    
    # Button to trigger the learning cycle
    if st.button("🔄 Update Knowledge Base Now", type="primary", use_container_width=True):
        if not agent:
            st.error("Please configure your GROQ_API_KEY and TAVILY_API_KEY first.")
        else:
            with st.spinner(f"Agent is researching '{agent_topic}', analyzing web sources, and updating knowledge base..."):
                try:
                    stats = agent.run_proactive_cycle()
                    st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.messages = []  # Reset chat history with fresh context
                    
                    if stats["total_chunks"] > 0:
                        st.success(f"Knowledge base updated! Added {stats['new_chunks']} chunks (Total: {stats['total_chunks']}).")
                    else:
                        st.warning("Learning cycle finished, but 0 chunks were added. Check Tavily search results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during learning cycle: {e}")
    
    if st.session_state.last_update_time:
        st.caption(f"🕒 Last updated: {st.session_state.last_update_time}")
    elif current_doc_count > 0:
        st.caption(f"💾 Using persistent knowledge base ({current_doc_count} chunks)")
    else:
        st.warning("Knowledge base is empty. Click the update button to start learning.")

# --- Main Chat Interface ---
st.title("Ask the AI Agent")
st.markdown(f"The agent is an expert on the topic: **'{agent_topic}'** *(Model: {selected_model})*")

if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    st.warning("⚠️ API keys missing: Please ensure GROQ_API_KEY and TAVILY_API_KEY are provided in the sidebar or .env file.")
    st.stop()

doc_count = agent.get_document_count() if agent else 0

if doc_count > 0:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the topic..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Searching knowledge base and reasoning with {selected_model}..."):
                try:
                    response = agent.ask(prompt)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error getting response: {e}")
                    response = f"Sorry, I encountered an error: {e}"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("💡 **Knowledge base is currently empty.** Click **'Update Knowledge Base Now'** in the sidebar to research this topic and build the knowledge base!")
