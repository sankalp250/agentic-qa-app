# 🧠 Proactive AI Agent: A "Living" Knowledge Base

A modern agentic app that proactively researches a topic, summarizes new content, filters for novelty, and assimilates it into a local FAISS vector store. Built with LangGraph, LangChain, and Groq for fast, cyclical learning beyond simple Q&A.

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
- **FAISS** indexes and searches local documents
- **Groq Chat model** handles summarization and critique

## 🚀 Quickstart

### 1) Clone and install
```bash
# from a terminal
cd D:\agentic-qa-app
# if you need a venv (optional if already present)
python -m venv venv
.\venv\Scripts\pip.exe install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt
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
.\venv\Scripts\streamlit.exe run streamlit_app.py
```
- **Scheduled CLI service:** runs a learning cycle now, then every 24h
```bash
.\venv\Scripts\python.exe run_proactive_agent.py
```
- **Single run (no scheduler):**
```bash
.\venv\Scripts\python.exe run_proactive_agent.py --run_once
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

## 🧩 Troubleshooting
- **Import "faiss" could not be resolved**
  - Install the Windows wheel (import name remains `faiss`):
    ```powershell
    .\venv\Scripts\pip.exe install faiss-cpu==1.8.0.post1
    ```
  - Ensure your IDE interpreter is `D:\agentic-qa-app\venv\Scripts\python.exe`.
  - GPU users (optional): `pip install faiss-gpu` (requires CUDA).

- **Import "apscheduler.schedulers.background" could not be resolved**
  - Install in the active venv:
    ```powershell
    .\venv\Scripts\pip.exe install APScheduler==3.10.4
    ```
  - Point IDE to the same venv and reload.

- **Dependency resolution errors with LangChain/LangGraph**
  - This repo pins compatible versions in `requirements.txt`:
    - `langchain==0.3.7`, `langchain-community==0.3.7`, `langgraph==0.6.8`
  - If issues persist, upgrade pip and reinstall:
    ```powershell
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    .\venv\Scripts\pip.exe install -r requirements.txt --upgrade
    ```

## 📜 License
MIT License. Feel free to use, modify, and build upon it.