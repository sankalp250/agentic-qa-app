"""
Main entry point for running the Proactive, "Living" Knowledge Base Agent.
"""
import time
import argparse
from apscheduler.schedulers.background import BackgroundScheduler
from agent.proactive_agent import ProactiveAgent
import ast

# --- Global Agent Instance ---
# We define it globally so the scheduler and the main loop can access the same agent.
agent = None

def run_learning_cycle():
    """Function to be called by the scheduler to run the agent's learning cycle."""
    if agent:
        try:
            print("\n--- Scheduler triggered: Starting proactive learning cycle ---")
            agent.run_proactive_cycle()
            print("--- Proactive learning cycle finished. Next run scheduled. ---")
        except Exception as e:
            print(f"❌ Error during scheduled learning cycle: {e}")
    else:
        print("Scheduler skipped: Agent not initialized yet.")

def start_interactive_session(agent_instance: ProactiveAgent):
    """Starts an interactive Q&A session with the agent."""
    print("\n✅ Agent is ready. You can now ask questions about its knowledge base.")
    print("   Type 'quit', 'exit', or 'bye' to stop.")
    print("-" * 50)

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! 👋")
                break
            if not question:
                continue

            print("\nBot is thinking...")
            answer = agent_instance.ask(question)
            print(f"\nBot: {answer}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    """Main function to initialize and run the agent."""
    global agent

    parser = argparse.ArgumentParser(
        description="Run the Proactive AI Agent.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Latest advancements in multimodal AI models",
        help="The topic for the agent to research."
    )
    parser.add_argument(
        "--run_once",
        action="store_true",
        help="Run the learning cycle once and then exit (for non-interactive use)."
    )
    args = parser.parse_args()

    print("--- Initializing Proactive Agent ---")
    try:
        agent = ProactiveAgent(topic=args.topic)
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # --- Run the first learning cycle immediately ---
    print("\n--- Running initial learning cycle to build knowledge base... ---")
    agent.run_proactive_cycle()

    if args.run_once:
        print("\n--- 'run_once' flag set. Exiting after initial cycle. ---")
        return

    # --- Schedule future runs ---
    scheduler = BackgroundScheduler()
    # Schedule the job to run every 24 hours
    scheduler.add_job(run_learning_cycle, 'interval', hours=24)
    scheduler.start()
    print("\n✅ Agent learning cycle has been scheduled to run every 24 hours.")
    
    try:
        # --- Start the interactive Q&A session ---
        start_interactive_session(agent)
    finally:
        # --- Clean shutdown ---
        print("\n--- Shutting down scheduler... ---")
        scheduler.shutdown()

if __name__ == "__main__":
    main()