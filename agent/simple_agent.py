"""
Simple AI Q&A Agent using LangGraph and Groq API
"""
import os
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State for the Q&A agent"""
    messages: list
    question: str
    answer: str
    error: str


class SimpleQAAgent:
    """Simple Q&A Agent using LangGraph"""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the Q&A Agent
        
        Args:
            model_name: The Groq model to use (default: llama3-8b-8192)
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please add your Groq API key to the .env file"
            )
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_question", self._process_question)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("process_question")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_question",
            self._should_generate_answer,
            {
                "generate": "generate_answer",
                "error": "handle_error"
            }
        )
        
        # End nodes
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _process_question(self, state: AgentState) -> AgentState:
        """Process and validate the user question"""
        question = state.get("question", "").strip()
        
        if not question:
            state["error"] = "Please provide a valid question."
            return state
        
        if len(question) < 3:
            state["error"] = "Question is too short. Please provide a more detailed question."
            return state
        
        # Add the question to messages
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append(HumanMessage(content=question))
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate an answer using the LLM"""
        try:
            # Create a system prompt for better responses
            system_prompt = """You are a helpful AI assistant. Provide clear, accurate, and helpful answers to user questions. 
            If you're not sure about something, acknowledge the uncertainty. Be concise but thorough."""
            
            messages = [
                HumanMessage(content=system_prompt),
                *state["messages"]
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Store the answer
            state["answer"] = response.content
            state["messages"].append(AIMessage(content=response.content))
            
        except Exception as e:
            state["error"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors gracefully"""
        error_msg = state.get("error", "Unknown error occurred")
        state["answer"] = f"Sorry, I encountered an error: {error_msg}"
        return state
    
    def _should_generate_answer(self, state: AgentState) -> str:
        """Determine if we should generate an answer or handle an error"""
        if state.get("error"):
            return "error"
        return "generate"
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: The user's question
            
        Returns:
            The AI's answer
        """
        # Initialize state
        initial_state = {
            "question": question,
            "messages": [],
            "answer": "",
            "error": ""
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state.get("answer", "Sorry, I couldn't generate an answer.")
    
    def chat(self):
        """Start an interactive chat session"""
        print("AI Q&A Bot (powered by LangGraph + Groq)")
        print("=" * 50)
        print("Ask me anything! Type 'quit', 'exit', or 'bye' to stop.")
        print()
        
        while True:
            try:
                # Get user input
                question = input("You: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nGoodbye! Thanks for chatting!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                print("\nBot: ", end="")
                answer = self.ask(question)
                print(answer)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n[ERROR] Error: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the Q&A bot"""
    try:
        # Create the agent
        agent = SimpleQAAgent()
        
        # Start the chat
        agent.chat()
        
    except ValueError as e:
        print(f"[ERROR] Configuration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Get a free API key from https://console.groq.com/")
        print("2. Create a .env file in the project root")
        print("3. Add your API key: GROQ_API_KEY=your_actual_api_key")
        
    except Exception as e:
        print(f"[ERROR] Unexpected Error: {e}")


if __name__ == "__main__":
    main()
