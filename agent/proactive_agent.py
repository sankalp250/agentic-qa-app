"""
Proactive AI Agent with a "Living" Knowledge Base
"""
import os
import json
import ast
from datetime import datetime, timedelta
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

# Try importing FAISS; optional
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# Automatically sync Streamlit secrets to environment if available
load_dotenv()
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

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
from langgraph.prebuilt import ToolNode

from agent.tools.web_search import get_web_search_tool

# Supported and active Groq models with auto-fallback
AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b"
]
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# --- 1. Agent State Definition ---
class ProactiveAgentState(TypedDict):
    """Represents the state of our proactive agent."""
    topic: str
    last_run_time: datetime
    messages: Annotated[list, add_messages]
    new_articles: List[dict]
    valuable_info: List[str]


# --- 2. The Proactive Agent Core ---
class ProactiveAgent:
    """
    An agent that proactively researches a topic, updates its own knowledge base,
    and can answer questions based on its accumulated knowledge and session context.
    """
    def __init__(self, topic: str, model_name: str = None):
        self.topic = topic
        self.model_name = model_name or DEFAULT_MODEL
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables or Streamlit secrets."
            )

        # Web search tool
        self.search_tool = get_web_search_tool()
        self.tools = [self.search_tool]
        self.tool_node = ToolNode(self.tools)

        # Vector Store for the knowledge base
        self.vector_store_type = "unknown"
        self.vector_store = self._initialize_vector_store()

        # LangGraph state machine
        self.graph = self._build_graph()

    def _invoke_llm(self, messages, max_tokens: int = 4096):
        """
        Invokes Groq LLM with high token ceiling (to avoid truncation)
        and automatic fallback across active models.
        """
        models_to_try = [self.model_name] + [m for m in AVAILABLE_GROQ_MODELS if m != self.model_name]
        last_error = None
        for model in models_to_try:
            try:
                llm = ChatGroq(
                    groq_api_key=self.groq_api_key,
                    model_name=model,
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                response = llm.invoke(messages)
                self.model_name = model
                return response
            except Exception as e:
                err_str = str(e).lower()
                if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                    print(f"⚠️ Model '{model}' unavailable, trying fallback...")
                    last_error = e
                    continue
                raise e
        if last_error:
            raise last_error

    def _initialize_vector_store(self):
        """Initializes a persistent vector store. Prefers Chroma persistent, fallbacks to FAISS."""
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        try:
            import chromadb
            persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
            os.makedirs(persist_directory, exist_ok=True)
            
            safe_collection_name = "".join(c if c.isalnum() else "_" for c in self.topic.lower())[:50]
            if not safe_collection_name:
                safe_collection_name = "agent_kb"
            safe_collection_name = f"kb_{safe_collection_name}".strip("_")[:63]
            if len(safe_collection_name) < 3:
                safe_collection_name = "kb_general"

            client = chromadb.PersistentClient(path=persist_directory)
            vector_store = Chroma(
                embedding_function=embedding_model,
                client=client,
                collection_name=safe_collection_name
            )
            self.vector_store_type = "chroma"
            print(f"✓ Vector store initialized with persistent Chroma at {persist_directory}")
            return vector_store
        except Exception as e:
            print(f"⚠️ Chroma persistent initialization failed, trying FAISS or in-memory: {e}")

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
                self.vector_store_type = "faiss"
                print("✓ Vector store initialized with FAISS.")
                return vector_store
            except Exception as e:
                print(f"⚠️ FAISS initialization failed: {e}")

        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.Client(Settings(anonymized_telemetry=False))
            vector_store = Chroma(
                embedding_function=embedding_model,
                client=client,
                collection_name="agent_kb"
            )
            self.vector_store_type = "chroma_ephemeral"
            print("✓ Vector store initialized with ephemeral Chroma.")
            return vector_store
        except Exception as e:
            raise Exception(f"Failed to initialize any vector store: {e}")

    def get_document_count(self) -> int:
        """Returns the number of documents/chunks in the vector store."""
        try:
            if hasattr(self.vector_store, "index") and hasattr(self.vector_store.index, "ntotal"):
                return int(self.vector_store.index.ntotal)
            if hasattr(self.vector_store, "_collection") and self.vector_store._collection is not None:
                return int(self.vector_store._collection.count())
        except Exception:
            pass
        return 0

    def _build_graph(self):
        """Builds the LangGraph state machine for the agent's workflow."""
        workflow = StateGraph(ProactiveAgentState)

        workflow.add_node("forage", self._forage_node)
        workflow.add_node("web_search_tool", self.tool_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("assimilate", self._assimilate_node)

        workflow.set_entry_point("forage")
        workflow.add_edge("forage", "web_search_tool")
        workflow.add_edge("web_search_tool", "summarize")
        workflow.add_edge("summarize", "critique")
        workflow.add_edge("critique", "assimilate")
        workflow.add_edge("assimilate", END)

        print("✓ Agent graph built successfully.")
        return workflow.compile()

    # --- Node Implementations ---

    def _forage_node(self, state: ProactiveAgentState):
        """Generates a targeted, informative search query based on the topic."""
        topic = state.get("topic", self.topic)
        print(f"--- foraging for new information on '{topic}' ---")
        query = f"Key concepts, architecture, guide, and recent developments in {topic}"

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
        
        raw_articles = []
        content = getattr(tool_message, "content", "")
        
        if isinstance(content, list):
            raw_articles = content
        elif isinstance(content, dict):
            raw_articles = content.get("results", [content])
        elif isinstance(content, str):
            try:
                raw_articles = json.loads(content)
            except Exception:
                try:
                    raw_articles = ast.literal_eval(content)
                except Exception:
                    pass
                    
        if isinstance(raw_articles, dict) and "results" in raw_articles:
            raw_articles = raw_articles["results"]

        if not raw_articles or not isinstance(raw_articles, list):
            print("Tool node output empty, falling back to direct web search invoke...")
            try:
                query = f"Key concepts, architecture, guide, and recent developments in {state['topic']}"
                direct_results = self.search_tool.invoke({"query": query})
                if isinstance(direct_results, list):
                    raw_articles = direct_results
                elif isinstance(direct_results, dict) and "results" in direct_results:
                    raw_articles = direct_results["results"]
            except Exception as e:
                print(f"Fallback search invocation failed: {e}")

        if not raw_articles or not isinstance(raw_articles, list):
            print("No articles found to summarize.")
            return {"new_articles": []}

        print(f"Processing {len(raw_articles)} raw search articles...")

        processed_articles = []
        for i, article in enumerate(raw_articles):
            if not isinstance(article, dict):
                continue
            
            title = article.get("title", f"Article {i+1}")
            url = article.get("url") or article.get("link") or article.get("href") or ""
            snippet = article.get("content") or article.get("snippet") or article.get("description") or ""
            
            article_content = snippet
            if url:
                try:
                    loader = WebBaseLoader(
                        [url],
                        requests_per_second=2,
                        header_template={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        }
                    )
                    docs = loader.load()
                    if docs and docs[0].page_content.strip():
                        article_content = docs[0].page_content[:4000]
                except Exception:
                    pass

            if not article_content or len(article_content.strip()) < 30:
                continue

            try:
                prompt = (
                    f"Please provide a clear, factual, and informative summary of the following content "
                    f"concerning '{state['topic']}':\n\n{article_content[:3500]}"
                )
                summary_response = self._invoke_llm([HumanMessage(content=prompt)], max_tokens=1500)
                summary_text = summary_response.content.strip()
            except Exception as e:
                print(f"LLM summary failed for article {i}: {e}")
                summary_text = snippet[:600]

            processed_articles.append({
                "title": title,
                "url": url,
                "summary": summary_text,
                "full_content": f"Title: {title}\nURL: {url}\n\nContent:\n{article_content}"
            })

        print(f"Successfully processed {len(processed_articles)} articles.")
        return {"new_articles": processed_articles}

    def _critique_node(self, state: ProactiveAgentState):
        """Critiques summaries to see if they are novel and valuable."""
        print("--- critiquing and filtering information ---")
        valuable_info = []
        current_docs_count = self.get_document_count()

        for article in state.get('new_articles', []):
            summary = article['summary']
            
            if current_docs_count == 0:
                print(f"Knowledge base is empty: Automatically assimilating '{article['title']}'")
                valuable_info.append(article['full_content'])
                continue

            similar_docs = self.vector_store.similarity_search(summary, k=1)
            context = "No similar information found in knowledge base."
            if similar_docs:
                context = f"Existing Knowledge Base Context:\n{similar_docs[0].page_content[:500]}"

            prompt = f"""
            You are a research curator evaluating whether new information adds value to the knowledge base.
            
            {context}

            New Information Summary:
            {summary}

            Does this new summary provide useful, novel, or complementary knowledge about '{state['topic']}'?
            Answer YES if it provides useful facts, details, or perspective.
            Answer NO only if it is entirely redundant, duplicate, or irrelevant.
            Reply with single word: YES or NO.
            """
            
            try:
                critique_response = self._invoke_llm([HumanMessage(content=prompt)], max_tokens=20)
                decision = critique_response.content.strip().upper()
            except Exception:
                decision = "YES"

            print(f"Critique for article '{article['title']}': {decision}")
            if "YES" in decision or "NO" not in decision:
                valuable_info.append(article['full_content'])

        return {"valuable_info": valuable_info}

    def _assimilate_node(self, state: ProactiveAgentState):
        """Adds valuable new information to the knowledge base."""
        valuable_info = state.get('valuable_info', [])
        print(f"--- assimilating {len(valuable_info)} new documents ---")
        if valuable_info:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = text_splitter.create_documents(valuable_info)
            self.vector_store.add_documents(chunks)
            print(f"✓ Added {len(chunks)} chunks to vector store. Total documents: {self.get_document_count()}")
        else:
            print("No new information met assimilation criteria.")
            
        return {"valuable_info": []}

    # --- Public Methods ---
    def run_proactive_cycle(self) -> dict:
        """Executes one full cycle of the agent's proactive learning and returns stats."""
        print(f"\n🚀 Starting proactive cycle for topic: '{self.topic}'")
        initial_count = self.get_document_count()
        initial_state = {
            "topic": self.topic,
            "last_run_time": datetime.now() - timedelta(days=7),
            "messages": [],
            "new_articles": [],
            "valuable_info": []
        }
        
        self.graph.invoke(initial_state)
        final_count = self.get_document_count()
        new_chunks = max(0, final_count - initial_count)
        print(f"✓ Proactive cycle completed. New chunks added: {new_chunks}. Total chunks: {final_count}")
        return {
            "topic": self.topic,
            "initial_chunks": initial_count,
            "new_chunks": new_chunks,
            "total_chunks": final_count
        }

    def ask(self, question: str, chat_history: List[dict] = None) -> str:
        """
        Answers a user question utilizing both indexed RAG knowledge chunks
        and conversational session memory, with a high 4096 token limit to prevent truncation.
        """
        print(f"\n❓ Answering question: '{question}' (model: {self.model_name})")
        doc_count = self.get_document_count()
        
        # 1. Retrieve knowledge base context
        context = ""
        has_kb_docs = False
        if doc_count > 0:
            relevant_docs = self.vector_store.similarity_search(question, k=4)
            if relevant_docs:
                has_kb_docs = True
                context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # 2. Format conversation history (session memory)
        history_formatted = ""
        if chat_history:
            recent_msgs = chat_history[-6:]
            history_lines = []
            for m in recent_msgs:
                role = "User" if m.get("role") == "user" else "Assistant"
                content_preview = m.get("content", "")
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "..."
                history_lines.append(f"{role}: {content_preview}")
            history_formatted = "\n".join(history_lines)

        # 3. Hybrid RAG prompt: ground in KB when available, answer comprehensively even if outside
        prompt = f"""
You are an expert AI research assistant.

Current Research Topic: {self.topic}

Conversation History (Session Memory):
{history_formatted if history_formatted else "None (Beginning of conversation)"}

Researched Knowledge Base Context:
---
{context if has_kb_docs else "No specific research chunks indexed yet for this question."}
---

User Question: {question}

Instructions:
1. If the Researched Knowledge Base Context above contains relevant facts, prioritize and cite them directly.
2. If the user refers back to previous questions in the Conversation History (e.g., comparing topics or follow-ups), maintain context and continuity naturally.
3. If the user asks about a topic outside the current knowledge base chunks, provide a thorough, helpful answer using your broad knowledge, while noting that it is general technical knowledge.
4. Structure your response with clean Markdown headings, bullet points, and code snippets where appropriate. Complete all thoughts and tables fully without stopping abruptly.
"""
        response = self._invoke_llm([HumanMessage(content=prompt)], max_tokens=4096)
        return response.content
