"""
Web Search Tool using Tavily API
"""
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def get_web_search_tool():
    """
    Initializes and returns the Tavily web search tool.

    This tool is configured to search for recent news and articles,
    making it ideal for our proactive agent.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in environment variables. "
            "Please add it to the .env file."
        )

    # max_results=5 ensures we get a good number of diverse sources.
    # search_depth="advanced" allows the tool to follow links for better results.
    web_search_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        name="web_search"
    )
    return web_search_tool

# Example of how to use the tool (optional, for testing)
if __name__ == '__main__':
    print("Testing the web search tool...")
    search_tool = get_web_search_tool()
    results = search_tool.invoke({
        "query": "latest news on LangGraph library"
    })
    print("Search Results:")
    for result in results:
        print(f"- {result['content']} (Source: {result['url']})")