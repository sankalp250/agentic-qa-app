#!/usr/bin/env python3
"""
Simple script to run the Streamlit web interface
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    try:
        # Change to the app directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run streamlit
        print("🚀 Starting AI Q&A Bot Web Interface...")
        print("📱 Your browser should open automatically")
        print("🌐 If not, go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the web server. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting the web app: {e}")
        print("💡 Try running: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
