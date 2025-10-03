# AI Q&A Bot

A simple command-line AI Q&A bot built with LangGraph and powered by Groq's free API.

## Live Demo

- Open the deployed app: [agentic-qa-app (Streamlit)](https://agentic-app-app-git-init-joqxdmqmxevtdt4byeczek.streamlit.app/)

## Features

- 🌐 **Web Interface**: Beautiful Streamlit frontend
- 🤖 **Interactive CLI**: Command-line chat mode
- ❓ **Single Question Mode**: Quick CLI queries
- 🔧 **LangGraph Architecture**: Robust conversation flow
- ⚡ **Groq API**: Fast inference with free tier
- 🎨 **Dual Interface**: Choose between web or CLI

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Create an API key

### 3. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

## Usage

### 🌐 Web Interface (Recommended)

```bash
# Option 1: Direct streamlit command
streamlit run streamlit_app.py

# Option 2: Using the helper script
python run_web.py
```

Opens a beautiful web interface in your browser with:
- Chat-style conversation
- Model selection
- Example questions
- Message history
- Easy setup guidance

### 🤖 Command Line Interface

#### Interactive Chat Mode
```bash
python main.py
```

#### Single Question Mode
```bash
python main.py -q "What is artificial intelligence?"
```

### Command Line Options

```bash
python main.py --help
```

Available options:
- `-q, --question`: Ask a single question
- `-m, --model`: Choose Groq model (default: llama3-8b-8192)
- `--version`: Show version information

### Available Models

- `llama-3.1-8b-instant` (default - fast and reliable)
- `llama3-70b-8192` (larger model, better for complex tasks)
- `mixtral-8x7b-32768` (good for reasoning tasks)
- `gemma2-9b-it` (efficient alternative)

> **Note**: Model availability may change. If you encounter model decommissioning errors, try a different model from the list above.

## Examples

### Web Interface
```bash
# Start the web app
streamlit run streamlit_app.py
# Then open your browser to http://localhost:8501
```

### CLI Interface
```bash
# Start interactive chat
python main.py

# Ask about programming
python main.py -q "Explain the difference between Python lists and tuples"

# Use a different model
python main.py -m llama3-70b-8192 -q "Write a short poem about AI"
```

## Project Structure

```
agentic-qa-app/
├── agent/
│   ├── __init__.py
│   └── simple_agent.py     # Main LangGraph agent
├── main.py                 # CLI interface
├── streamlit_app.py        # Web interface
├── run_web.py              # Helper script to start web app
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
├── .gitignore             # Git configuration
└── README.md              # This file
```

## Architecture

The bot uses LangGraph to create a simple but robust conversation flow:

1. **Process Question**: Validates and prepares user input
2. **Generate Answer**: Uses Groq API to generate responses
3. **Handle Errors**: Gracefully manages API errors and edge cases

## Contributing

This is a simple starter project. Feel free to enhance it with:

- Web search capabilities
- Memory/conversation history
- Multiple AI providers
- More sophisticated prompting
- Chat history persistence

## License

MIT License - feel free to use and modify as needed.