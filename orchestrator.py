# agents/orchestrator.py
"""
Step 7: LangGraph Multi-Agent Orchestrator
Wires together Retrieval → Validation → Response agents
using a typed state graph with conditional routing.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from loguru import logger

from core.config import settings
from core.vector_store import get_vector_store, BaseVectorStore
from core.hybrid_search import HybridSearcher
from agents.retrieval_agent import RetrievalAgent
from agents.validation_agent import ValidationAgent, CrossEncoderReranker
from agents.response_agent import ResponseAgent, AgentResponse


# ──────────────────────────────────────────────
# State Schema
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between agents in the LangGraph.
    Each agent reads from and writes to this state.
    """
    # Input
    query: str
    user_roles: List[str]
    conversation_history: List[Dict[str, str]]
    session_id: str

    # Retrieval Agent output
    retrieval_output: Dict[str, Any]

    # Validation Agent output
    validation_output: Dict[str, Any]

    # Response Agent output
    final_response: Optional[AgentResponse]

    # Control flags
    error: Optional[str]
    steps_completed: Annotated[List[str], operator.add]   # Trace each step


# ──────────────────────────────────────────────
# Node Functions (one per agent)
# ──────────────────────────────────────────────

def retrieval_node(state: AgentState, retrieval_agent: RetrievalAgent) -> AgentState:
    """LangGraph node: runs the RetrievalAgent."""
    logger.info("[Graph] Retrieval node executing")
    try:
        output = retrieval_agent.retrieve(
            query=state["query"],
            user_roles=state.get("user_roles", ["all"]),
            top_k=settings.top_k_results,
        )
        return {
            **state,
            "retrieval_output": output,
            "steps_completed": ["retrieval"],
        }
    except Exception as e:
        logger.error(f"Retrieval node error: {e}")
        return {**state, "error": str(e), "steps_completed": ["retrieval_failed"]}


def validation_node(state: AgentState, validation_agent: ValidationAgent) -> AgentState:
    """LangGraph node: runs the ValidationAgent."""
    logger.info("[Graph] Validation node executing")
    try:
        output = validation_agent.validate(
            query=state["query"],
            retrieval_output=state["retrieval_output"],
            top_k=settings.rerank_top_k,
        )
        return {
            **state,
            "validation_output": output,
            "steps_completed": ["validation"],
        }
    except Exception as e:
        logger.error(f"Validation node error: {e}")
        return {**state, "error": str(e), "steps_completed": ["validation_failed"]}


def response_node(state: AgentState, response_agent: ResponseAgent) -> AgentState:
    """LangGraph node: runs the ResponseAgent."""
    logger.info("[Graph] Response node executing")
    try:
        response = response_agent.generate(
            query=state["query"],
            validation_output=state["validation_output"],
            conversation_history=state.get("conversation_history", []),
        )
        return {
            **state,
            "final_response": response,
            "steps_completed": ["response"],
        }
    except Exception as e:
        logger.error(f"Response node error: {e}")
        return {**state, "error": str(e), "steps_completed": ["response_failed"]}


def should_continue_after_retrieval(state: AgentState) -> str:
    """Conditional edge: stop if retrieval errored or returned nothing."""
    if state.get("error"):
        return "error"
    retrieval = state.get("retrieval_output", {})
    if not retrieval.get("results"):
        return "no_results"
    return "validate"


def should_continue_after_validation(state: AgentState) -> str:
    """Conditional edge: stop if validation errored."""
    if state.get("error"):
        return "error"
    return "respond"


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Builds and runs the LangGraph multi-agent pipeline.

    Pipeline:
      START
        ↓
      retrieval_node  ──(error / no_results)──→ END
        ↓ (validate)
      validation_node ──(error)───────────────→ END
        ↓ (respond)
      response_node
        ↓
      END
    """

    def __init__(self, vector_store: Optional[BaseVectorStore] = None):
        # Initialize components
        self.vector_store = vector_store or get_vector_store()
        self.searcher = HybridSearcher(self.vector_store)
        self.retrieval_agent = RetrievalAgent(self.searcher, use_query_expansion=True)
        self.validation_agent = ValidationAgent(CrossEncoderReranker(use_llm=False))
        self.response_agent = ResponseAgent()

        self._graph = None
        self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph state machine."""
        try:
            from langgraph.graph import StateGraph, END

            # Bind agent instances to node functions via closures
            ra = self.retrieval_agent
            va = self.validation_agent
            resp_a = self.response_agent

            graph = StateGraph(AgentState)

            # Add nodes
            graph.add_node("retrieval",  lambda s: retrieval_node(s, ra))
            graph.add_node("validation", lambda s: validation_node(s, va))
            graph.add_node("response",   lambda s: response_node(s, resp_a))

            # Entry point
            graph.set_entry_point("retrieval")

            # Conditional routing after retrieval
            graph.add_conditional_edges(
                "retrieval",
                should_continue_after_retrieval,
                {
                    "validate":   "validation",
                    "no_results": "response",   # Response agent handles empty case
                    "error":      END,
                },
            )

            # Conditional routing after validation
            graph.add_conditional_edges(
                "validation",
                should_continue_after_validation,
                {
                    "respond": "response",
                    "error":   END,
                },
            )

            # Response always goes to END
            graph.add_edge("response", END)

            self._graph = graph.compile()
            logger.info("LangGraph compiled successfully")

        except ImportError:
            logger.warning(
                "langgraph not installed. Using sequential fallback mode."
            )
            self._graph = None

    def run(
        self,
        query: str,
        user_roles: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Run the full multi-agent pipeline for a query.

        Args:
            query: User's natural language question
            user_roles: User's access roles for RBAC
            conversation_history: Prior turns [{"role": "user"/"assistant", "content": "..."}]
            session_id: Session identifier for memory

        Returns:
            Dict with answer, citations, confidence, sources, trace
        """
        initial_state: AgentState = {
            "query": query,
            "user_roles": user_roles or ["all"],
            "conversation_history": conversation_history or [],
            "session_id": session_id,
            "retrieval_output": {},
            "validation_output": {},
            "final_response": None,
            "error": None,
            "steps_completed": [],
        }

        if self._graph:
            # Run via LangGraph
            logger.info(f"Running LangGraph pipeline for: '{query[:60]}'")
            final_state = self._graph.invoke(initial_state)
        else:
            # Sequential fallback (no langgraph)
            logger.info("Running sequential fallback pipeline")
            final_state = self._run_sequential(initial_state)

        # Build response dict
        response = final_state.get("final_response")
        if response:
            result = response.to_dict()
        else:
            result = {
                "answer": "An error occurred while processing your request.",
                "citations": [],
                "confidence": 0.0,
                "query": query,
                "sources": [],
            }

        result["steps_completed"] = final_state.get("steps_completed", [])
        result["error"] = final_state.get("error")
        return result

    def _run_sequential(self, state: AgentState) -> AgentState:
        """Fallback: run agents sequentially without LangGraph."""
        state = retrieval_node(state, self.retrieval_agent)
        if state.get("error") or not state.get("retrieval_output", {}).get("results"):
            return response_node(state, self.response_agent)

        state = validation_node(state, self.validation_agent)
        if state.get("error"):
            return state

        state = response_node(state, self.response_agent)
        return state

    def add_documents(self, chunks) -> None:
        """Add new document chunks and rebuild BM25 index."""
        self.vector_store.add_chunks(chunks)
        self.searcher.rebuild_bm25()
        logger.info(f"Added {len(chunks)} chunks. BM25 index rebuilt.")
