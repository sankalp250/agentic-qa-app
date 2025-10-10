🧠 Proactive AI Agent: A "Living" Knowledge Base
This project is not just a Q&A bot; it's a sophisticated, autonomous AI agent designed to become a world-class expert on any topic you assign. It proactively researches subjects online, builds its own "living" knowledge base, and intelligently keeps it up-to-date.

Powered by a robust LangGraph state machine, high-speed inference from the Groq API, and a local FAISS vector store, this agent demonstrates a modern, cyclical AI architecture that goes far beyond simple request-response interactions.

✨ Key Features
🧠 Autonomous Learning: The agent runs on its own to find, analyze, and assimilate the latest information on a topic.

🌐 Proactive Web Foraging: Uses the Tavily search API to discover the most recent articles, research, and news from the web.

🤔 Intelligent Critique System: Before saving new information, the agent performs a RAG query on its own knowledge base to determine if the information is truly novel and valuable, preventing redundancy and ensuring high-quality data.

💾 Persistent Knowledge Base: Builds and maintains a local FAISS vector store, allowing its expertise to grow and persist over time.

⚡ High-Speed LLM: Leverages the Groq API for near-instant reasoning, summarization, and analysis tasks.

🔧 Advanced Agentic Architecture: Built with LangGraph to create a reliable, stateful, and cyclical workflow (Forage -> Summarize -> Critique -> Assimilate).

🖥️ Interactive Web Interface: A polished Streamlit application allows anyone to interact with the agent, query its knowledge, and command it to learn.

🏛️ Architecture: The Agent's Learning Cycle
The core of this project is the agent's autonomous learning cycle, managed by a LangGraph state machine. This cycle defines its "thought process" and actions.

Forage 🔎
The agent begins by formulating a targeted search query based on its core topic and the time since its last update. It then uses a web search tool to find the most recent and relevant online sources.

Summarize 📝
It fetches the full content from the discovered URLs and uses the LLM to generate concise, factual summaries. This step is crucial for quickly understanding the essence of each new document.

Critique 🤔
This is the agent's most intelligent step. For each summary, it performs a similarity search against its own existing knowledge base. It asks itself: "Is this information truly new, or is it just a rehash of what I already know?" This self-reflection prevents data duplication and ensures the knowledge base remains high-quality.

Assimilate 💾
If the information is deemed novel and valuable, the agent chunks the full original content, creates vector embeddings, and permanently adds them to its FAISS vector store. Its knowledge has now officially grown.

This entire cycle can be triggered on-demand through the web UI, demonstrating a complete, end-to-end agentic workflow.

🚀 Getting Started
1. Installation
Clone the repository and install the required dependencies:

Bash

git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
2. Get API Keys
You will need two free API keys for this project to function:

Groq API Key: For LLM inference. Get it from the Groq Console.

Tavily API Key: For the advanced web search tool. Get it from Tavily AI.

3. Configure Your Environment
Create a .env file in the root directory of the project. Copy the contents of .env.example and add your API keys:

Code snippet

GROQ_API_KEY="your_actual_groq_api_key_here"
TAVILY_API_KEY="your_actual_tavily_api_key_here"
🖥️ Usage: The Web Interface
The best way to interact with the agent is through the Streamlit web application.

To start the app, run:

Bash

streamlit run streamlit_app.py
How It Works:
Set a Topic: Use the sidebar to define the research topic you want the agent to become an expert on.

Build the Knowledge Base: Click the "🔄 Update Knowledge Base Now" button. This will command the agent to perform its full, real-time learning cycle. You will see its progress as it works.

Ask Questions: Once the knowledge base is built, you can ask the agent complex questions about the topic. It will use its own research—not its pre-trained knowledge—to provide accurate, up-to-date answers.

📂 Project Structure
The project has been refactored to support the advanced agentic architecture:

agentic-qa-app/
├── agent/
│   ├── tools/
│   │   └── web_search.py   # The agent's web search tool
│   ├── __init__.py
│   ├── proactive_agent.py  # The core logic for the advanced agent
│   └── simple_agent.py     # The original basic Q&A agent (archived)
├── .env                    # Your API keys
├── main.py                 # Original CLI script (archived)
├── requirements.txt
├── run_proactive_agent.py  # Script to run the agent as a continuous service
└── streamlit_app.py        # The main web interface
└── README.md               # This file
License
This project is licensed under the MIT License. Feel free to use, modify, and build upon it.