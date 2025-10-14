"""
Streamlit Frontend for the Proactive AI Agent
"""
import streamlit as st
from datetime import datetime
from agent.proactive_agent import ProactiveAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Proactive AI Agent",
    page_icon="🧠",
    layout="wide",
)

# --- Agent Initialization ---
# This function is now simpler. It just creates the agent.
# We will run the learning cycle separately.
@st.cache_resource
def get_agent(topic):
    """
    Loads the ProactiveAgent and caches it.
    Does NOT run the learning cycle here.
    """
    try:
        agent = ProactiveAgent(topic=topic)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}", icon="🚨")
        return None

# --- Main Application Logic ---

# Initialize session state for tracking updates
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Proactive AI Agent")
    st.markdown("This agent autonomously researches a topic to build its knowledge base.")
    
    st.subheader("Configuration")
    agent_topic = st.text_input(
        "Agent Research Topic", 
        "Latest advancements in multimodal AI models",
        help="The topic the agent will research."
    )

    st.subheader("Knowledge Base Control")
    # Button to trigger the learning cycle
    if st.button("🔄 Update Knowledge Base Now", type="primary"):
        with st.spinner("Agent is learning... This may take a moment."):
            try:
                agent = get_agent(agent_topic)
                if agent:
                    agent.run_proactive_cycle()
                    st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Clear chat history after updating knowledge
                    st.session_state.messages = [] 
                    st.success("Knowledge base updated!")
                else:
                    st.error("Failed to initialize agent. Please check your API keys.")
            except Exception as e:
                st.error(f"Error during learning cycle: {e}")
                st.info("This might be due to network connectivity issues. Please try again later.")
    
    if st.session_state.last_update_time:
        st.success(f"Knowledge last updated at: {st.session_state.last_update_time}")
    else:
        st.warning("Knowledge base is empty. Click the update button to start the agent's learning process.")

# --- Main Chat Interface ---
st.title("Ask the AI Agent")
st.markdown(f"The agent is an expert on the topic: **'{agent_topic}'**")

# Only initialize agent if knowledge base has been updated
if st.session_state.last_update_time:
    # Get the cached agent instance
    agent = get_agent(agent_topic)
    
    if agent:
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
                with st.spinner("Searching knowledge base..."):
                    try:
                        response = agent.ask(prompt)
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error getting response: {e}")
                        response = "Sorry, I encountered an error while processing your question."
                
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Agent could not be initialized. Please check the logs in your terminal.")
else:
    st.info("Please click the 'Update Knowledge Base Now' button in the sidebar to enable the agent.")