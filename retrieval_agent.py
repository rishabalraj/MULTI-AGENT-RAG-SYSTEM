# agents/retrieval_agent.py
"""
Step 4: Retrieval Agent
First agent in the pipeline. Receives user query, expands it,
runs hybrid search, and returns raw candidate chunks.
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from core.config import settings
from core.hybrid_search import HybridSearcher, SearchResult


# ──────────────────────────────────────────────
# Query Expander (optional but improves recall)
# ──────────────────────────────────────────────

class QueryExpander:
    """
    Uses an LLM to generate alternative phrasings of the query.
    Increases recall by retrieving documents that match different
    ways of expressing the same information need.
    """

    EXPANSION_PROMPT = """You are a search query optimizer.
Given a user's question, generate 2 alternative search queries that capture
the same intent using different vocabulary. Output ONLY the 2 queries,
one per line. No numbering, no explanation.

User question: {query}
Alternative queries:"""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            if settings.openai_api_key:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=settings.llm_model,
                    temperature=0,
                    api_key=settings.openai_api_key,
                )
            else:
                from langchain_community.llms import Ollama
                self._llm = Ollama(
                    model=settings.ollama_model, base_url=settings.ollama_base_url
                )
        return self._llm

    def expand(self, query: str) -> List[str]:
        """Return original + expanded queries."""
        try:
            llm = self._get_llm()
            prompt = self.EXPANSION_PROMPT.format(query=query)
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            expansions = [q.strip() for q in content.strip().split("\n") if q.strip()]
            logger.debug(f"Query expansions: {expansions}")
            return [query] + expansions[:2]   # Original + up to 2 alternatives
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return [query]


# ──────────────────────────────────────────────
# Retrieval Agent
# ──────────────────────────────────────────────

class RetrievalAgent:
    """
    Agent 1/3: Retrieves relevant document chunks.

    Workflow:
    1. Optionally expand the query into multiple phrasings
    2. Run hybrid search for each phrasing
    3. Deduplicate & merge results
    4. Return top-K candidates for the Validation Agent
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        use_query_expansion: bool = True,
    ):
        self.searcher = searcher
        self.expander = QueryExpander() if use_query_expansion else None

    def retrieve(
        self,
        query: str,
        user_roles: Optional[List[str]] = None,
        top_k: int = settings.top_k_results,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's natural language question
            user_roles: RBAC roles for this user (e.g. ["engineering", "all"])
            top_k: Max number of chunks to return

        Returns:
            Dict with keys:
              - original_query: str
              - expanded_queries: List[str]
              - results: List[SearchResult]
              - num_results: int
        """
        logger.info(f"RetrievalAgent: processing query = '{query[:80]}'")

        # Step 1: Query expansion
        if self.expander:
            queries = self.expander.expand(query)
        else:
            queries = [query]

        # Step 2: Run hybrid search for each query
        all_results: Dict[str, SearchResult] = {}

        for q in queries:
            results = self.searcher.search(
                query=q,
                top_k=top_k,
                filter_roles=user_roles,
            )
            for r in results:
                cid = r.chunk.chunk_id
                if cid not in all_results or r.hybrid_score > all_results[cid].hybrid_score:
                    all_results[cid] = r

        # Step 3: Sort by hybrid score, return top_k
        merged = sorted(all_results.values(), key=lambda x: x.hybrid_score, reverse=True)
        final = merged[:top_k]

        logger.info(
            f"RetrievalAgent: retrieved {len(final)} chunks "
            f"from {len(queries)} query variant(s)"
        )

        return {
            "original_query": query,
            "expanded_queries": queries,
            "results": final,
            "num_results": len(final),
        }
