# 🧠 Proactive AI Agent: A "Living" Knowledge Base

[![Streamlit App](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agentic-app-app-9sqfnhaptnfzcrwmuvjxhx.streamlit.app/)

A modern agentic app that proactively researches a topic, summarizes new content, filters for novelty, and assimilates it into a local vector store. Built with LangGraph, LangChain, and Groq for fast, cyclical learning beyond simple Q&A.

> **🌐 [Try the Live Demo](https://agentic-app-app-9sqfnhaptnfzcrwmuvjxhx.streamlit.app/)** - No installation required!

## ✨ Features
- **Autonomous learning cycle**: Forage → Summarize → Critique → Assimilate
- **Web foraging**: Uses Tavily for recent, relevant sources
- **Novelty filter**: RAG check to avoid duplicates and keep quality high
- **Local knowledge base**: FAISS vector store grows over time
- **Fast LLM**: Groq API for low-latency reasoning and summaries
- **Two UIs**: Streamlit web app and a scheduled CLI runner

## 🏛️ Architecture at a glance
- **LangGraph** orchestrates the workflow with a state machine
- **HuggingFaceEmbeddings** generate vector representations
- **FAISS/ChromaDB** indexes and searches local documents (auto-fallback)
- **Groq Chat model** handles summarization and critique
- **Tavily API** provides web search capabilities

## 🚀 Quickstart

### 1) Clone and install
```bash
# Clone the repository
git clone https://github.com/sankalp250/agentic-qa-app.git
cd agentic-qa-app

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Environment variables
Create a `.env` in the project root with:
```bash
GROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### 3) Run options
- **Web app (recommended):**
```bash
# On Windows:
.\venv\Scripts\streamlit.exe run streamlit_app.py
# On macOS/Linux:
streamlit run streamlit_app.py
```
- **Scheduled CLI service:** runs a learning cycle now, then every 24h
```bash
# On Windows:
.\venv\Scripts\python.exe run_proactive_agent.py
# On macOS/Linux:
python run_proactive_agent.py
```
- **Single run (no scheduler):**
```bash
# On Windows:
.\venv\Scripts\python.exe run_proactive_agent.py --run_once
# On macOS/Linux:
python run_proactive_agent.py --run_once
```

## 📂 Project structure
```text
agentic-qa-app/
├─ agent/
│  ├─ tools/
│  │  └─ web_search.py       # Tavily-backed web search tool
│  ├─ __init__.py
│  ├─ proactive_agent.py     # Core agent + LangGraph workflow
│  └─ simple_agent.py        # Original basic Q&A agent (legacy)
├─ .env                      # Your API keys (not committed)
├─ requirements.txt
├─ run_proactive_agent.py    # CLI entrypoint with APScheduler
├─ streamlit_app.py          # Streamlit UI
└─ README.md
```

### **API Setup**
1. Get a free Groq API key: https://console.groq.com/
2. Get a free Tavily API key: https://tavily.com/
3. Create `.env` file with your keys:
   ```bash
   GROQ_API_KEY="your_groq_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

## 📜 License
MIT License. Feel free to use, modify, and build upon it.