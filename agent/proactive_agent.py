"""
Proactive AI Agent with a "Living" Knowledge Base
"""
import os
try:
    import faiss  # optional; may be unavailable on Windows
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False
import json
from datetime import datetime, timedelta
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools.web_search import get_web_search_tool
import chromadb
from chromadb.config import Settings

# Load environment variables from .env file
load_dotenv()

# --- 1. Agent State Definition ---
class ProactiveAgentState(TypedDict):
    """
    Represents the state of our proactive agent.
    """
    topic: str
    last_run_time: datetime
    # Messages have a special annotation that tells LangGraph how to aggregate them
    messages: Annotated[list, add_messages]
    # These fields will hold the data as it flows through the graph
    new_articles: List[dict]
    valuable_info: List[str]


# --- 2. The Proactive Agent Core ---
class ProactiveAgent:
    """
    An agent that proactively researches a topic, updates its own knowledge base,
    and can answer questions based on its accumulated knowledge.
    """
    def __init__(self, topic: str, model_name: str = "llama-3.1-8b-instant"):
        self.topic = topic
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        # LLM for reasoning and generation
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=0.7
        )
        # Tool for web search
        self.tools = [get_web_search_tool()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

        # Vector Store for the knowledge base
        self.vector_store = self._initialize_vector_store()

        # LangGraph state machine
        self.graph = self._build_graph()

    def _initialize_vector_store(self):
        """Initializes an in-memory vector store. Prefer FAISS; fallback to Chroma."""
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Try FAISS first if available
        if HAS_FAISS:
            try:
                embedding_size = 384
                index = faiss.IndexFlatL2(embedding_size)
                docstore = InMemoryDocstore({})
                vector_store = FAISS(
                    embedding_function=embedding_model,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id={}
                )
                print("✅ Vector store initialized with FAISS.")
                return vector_store
            except Exception as e:
                print(f"⚠️ FAISS initialization failed, falling back to Chroma: {e}")

        # Fallback: Chroma (pure Python)
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        vector_store = Chroma(
            embedding_function=embedding_model,
            client=client,
            collection_name="agent_kb"
        )
        print("✅ Vector store initialized with Chroma.")
        return vector_store

    def _build_graph(self):
        """Builds the LangGraph state machine for the agent's workflow."""
        workflow = StateGraph(ProactiveAgentState)

        workflow.add_node("forage", self._forage_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("assimilate", self._assimilate_node)
        workflow.add_node("web_search_tool", self.tool_node)

        workflow.set_entry_point("forage")
        workflow.add_edge("forage", "web_search_tool")
        workflow.add_edge("web_search_tool", "summarize")
        workflow.add_edge("summarize", "critique")
        workflow.add_edge("critique", "assimilate")
        workflow.add_edge("assimilate", END)

        print("✅ Agent graph built successfully.")
        return workflow.compile()

    # --- Node Implementations ---

    def _forage_node(self, state: ProactiveAgentState):
        """Generates a search query based on the topic and time."""
        print("--- foraging for new information ---")
        time_since_last_run = datetime.now() - state['last_run_time']
        query = (
            f"Find the latest news, articles, and research on '{state['topic']}' "
            f"published in the last {time_since_last_run.days + 1} days."
        )
        # We create a fake AIMessage to trigger the tool call
        message = AIMessage(
            content="",
            tool_calls=[{
                "id": "forage_search_1",
                "name": "web_search",
                "args": {"query": query}
            }]
        )
        return {"messages": [message]}

    def _summarize_node(self, state: ProactiveAgentState):
        """Summarizes the content of new articles found."""
        print("--- summarizing new articles ---")
        tool_message = state['messages'][-1]
        
        # content is a stringified list of dicts
        raw_articles = json.loads(tool_message.content) 

        if not raw_articles:
            print("No new articles found to summarize.")
            return {"new_articles": []}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        loader = WebBaseLoader(
            [article['url'] for article in raw_articles],
            requests_per_second=2,
            header_template=headers
        )
        docs = loader.load()

        processed_articles = []
        for i, doc in enumerate(docs):
            prompt = f"Please provide a concise, factual summary of the following text:\n\n{doc.page_content[:4000]}"
            summary_response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Store everything we need for the next steps
            processed_articles.append({
                "title": raw_articles[i]['title'],
                "url": raw_articles[i]['url'],
                "summary": summary_response.content,
                "full_content": doc.page_content
            })

        return {"new_articles": processed_articles}

    def _critique_node(self, state: ProactiveAgentState):
        """Critiques summaries to see if they are novel and valuable."""
        print("--- critiquing and filtering information ---")
        valuable_info = []
        for article in state['new_articles']:
            summary = article['summary']
            similar_docs = self.vector_store.similarity_search(summary, k=1)
            
            context = "No similar information found in knowledge base."
            if similar_docs:
                context = f"Found similar information in knowledge base:\n{similar_docs[0].page_content}"

            prompt = f"""
            You are a data curation expert. Your job is to determine if new information is valuable.
            
            Current Knowledge Base Context:
            {context}

            New Information Summary:
            {summary}

            Based on the context from the knowledge base, is this new summary providing significant new information, a novel perspective, or a critical update?
            Answer with a single word: YES or NO.
            """
            
            critique_response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = critique_response.content.strip().upper()

            print(f"Critique for article '{article['title']}': {decision}")
            if "YES" in decision:
                # We store the original, full content, not just the summary
                valuable_info.append(article['full_content'])

        return {"valuable_info": valuable_info}

    def _assimilate_node(self, state: ProactiveAgentState):
        """Adds valuable new information to the knowledge base."""
        valuable_info = state.get('valuable_info', [])
        print(f"--- assimilating {len(valuable_info)} new documents ---")
        if valuable_info:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.create_documents(valuable_info)
            self.vector_store.add_documents(chunks)
            # Report count for both FAISS and Chroma
            total_docs = None
            try:
                if hasattr(self.vector_store, "index") and hasattr(self.vector_store.index, "ntotal"):
                    total_docs = self.vector_store.index.ntotal
                elif hasattr(self.vector_store, "_collection") and self.vector_store._collection is not None:
                    total_docs = self.vector_store._collection.count()
            except Exception:
                pass
            if total_docs is not None:
                print(f"✅ Knowledge base updated. Total documents: {total_docs}")
            else:
                print("✅ Knowledge base updated.")
        else:
            print("✅ No new valuable information to add.")
            
        return {}

    # --- Public Methods ---
    def run_proactive_cycle(self):
        """Executes one full cycle of the agent's proactive learning."""
        print(f"\n🚀 Starting proactive cycle for topic: '{self.topic}'")
        initial_state = {
            "topic": self.topic,
            "last_run_time": datetime.now() - timedelta(days=1),
            "messages": [],
        }
        # Run the graph from start to finish
        self.graph.invoke(initial_state)
        print("✨ Proactive cycle completed.")

    def ask(self, question: str) -> str:
        """Asks a question to the agent's knowledge base."""
        print(f"\n❓ Answering question: '{question}'")
        try:
            has_faiss_index = hasattr(self.vector_store, "index") and hasattr(self.vector_store.index, "ntotal")
            if has_faiss_index and self.vector_store.index.ntotal == 0:
                return "My knowledge base is empty. Please run a proactive cycle first."
            if hasattr(self.vector_store, "_collection") and self.vector_store._collection is not None:
                if self.vector_store._collection.count() == 0:
                    return "My knowledge base is empty. Please run a proactive cycle first."
        except Exception:
            # If we cannot determine, proceed to try answering
            pass

        # Retrieve relevant documents from the knowledge base (RAG)
        relevant_docs = self.vector_store.similarity_search(question, k=4) # Increased to 4 for more context
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # A much stricter, more directive prompt
        prompt = f"""
You are a specialized research assistant. Your sole purpose is to synthesize and answer questions based *only* on the provided text.
You are forbidden from using any external knowledge. Do not mention your own knowledge cutoff date.

Here is the context retrieved from your knowledge base:
---
{context}
---

Based exclusively on the text provided above, answer the following question:
Question: {question}

Answer:
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content