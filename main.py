#!/usr/bin/env python3
"""
AI Q&A Bot - Main Entry Point
A simple command-line Q&A bot using LangGraph and Groq API
"""

import sys
import argparse
from agent.simple_agent import SimpleQAAgent


def run_single_question(question: str):
    """Run a single question and exit"""
    try:
        agent = SimpleQAAgent()
        answer = agent.ask(question)
        print(f"Bot Answer: {answer}")
        
    except ValueError as e:
        print(f"[ERROR] Configuration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Get a free API key from https://console.groq.com/")
        print("2. Create a .env file in the project root")
        print("3. Add your API key: GROQ_API_KEY=your_actual_api_key")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_interactive_chat():
    """Run interactive chat mode"""
    try:
        agent = SimpleQAAgent()
        agent.chat()
        
    except ValueError as e:
        print(f"[ERROR] Configuration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Get a free API key from https://console.groq.com/")
        print("2. Create a .env file in the project root")
        print("3. Add your API key: GROQ_API_KEY=your_actual_api_key")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="AI Q&A Bot - Ask questions and get AI-powered answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Start interactive chat
  python main.py -q "What is artificial intelligence?"  # Ask a single question
  python main.py --question "Explain quantum computing"  # Ask a single question

Models available:
  - llama-3.1-8b-instant (default)
  - llama3-70b-8192
  - mixtral-8x7b-32768
  - gemma2-9b-it
        """
    )
    
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Ask a single question and exit"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Groq model to use (default: llama-3.1-8b-instant)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="AI Q&A Bot v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Handle single question mode
    if args.question:
        run_single_question(args.question)
    else:
        # Run interactive chat
        run_interactive_chat()


if __name__ == "__main__":
    main()
