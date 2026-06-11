# agents/validation_agent.py
"""
Step 5: Validation Agent
Second agent in the pipeline. Takes raw retrieval results,
cross-encodes them against the query for precise relevance scoring,
filters low-quality chunks, and returns a ranked shortlist.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.config import settings
from core.hybrid_search import SearchResult


# ──────────────────────────────────────────────
# Cross-Encoder Reranker
# ──────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks (query, passage) pairs using an LLM-based relevance scorer.

    Two modes:
    1. Local cross-encoder model (sentence-transformers) — fast, free
    2. LLM-based scoring — more accurate but slower
    """

    RELEVANCE_PROMPT = """Score the relevance of the following passage to the query.
Return ONLY a number between 0 and 10. 10 = perfectly answers the query, 0 = irrelevant.

Query: {query}
Passage: {passage}
Score:"""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._model = None
        self._llm = None

    def _get_model(self):
        """Load cross-encoder model (local)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Loaded local cross-encoder model")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to hybrid score ordering."
                )
        return self._model

    def _get_llm(self):
        """Load LLM for relevance scoring."""
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

    def rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Return results sorted by cross-encoder relevance score."""
        if not results:
            return []

        if self.use_llm:
            return self._llm_rerank(query, results)
        else:
            return self._cross_encoder_rerank(query, results)

    def _cross_encoder_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        model = self._get_model()
        if model is None:
            return results   # Fallback: return as-is

        pairs = [(query, r.chunk.text) for r in results]
        scores = model.predict(pairs)

        for result, score in zip(results, scores):
            result.hybrid_score = float(score)

        return sorted(results, key=lambda x: x.hybrid_score, reverse=True)

    def _llm_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Score each (query, passage) with the LLM — accurate but slow."""
        llm = self._get_llm()
        scored = []

        for result in results:
            try:
                prompt = self.RELEVANCE_PROMPT.format(
                    query=query, passage=result.chunk.text[:500]
                )
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                score = float(content.strip().split()[0])
                result.hybrid_score = score / 10.0   # Normalize to 0–1
            except Exception as e:
                logger.debug(f"LLM rerank scoring error: {e}")
                # Keep original score

            scored.append(result)

        return sorted(scored, key=lambda x: x.hybrid_score, reverse=True)


# ──────────────────────────────────────────────
# Validation Agent
# ──────────────────────────────────────────────

class ValidationAgent:
    """
    Agent 2/3: Validates and re-ranks retrieval results.

    Workflow:
    1. Cross-encoder reranking for precise relevance
    2. Duplicate / near-duplicate removal
    3. Relevance threshold filtering
    4. Source diversity enforcement
    5. Return validated, ranked shortlist
    """

    def __init__(self, reranker: Optional[CrossEncoderReranker] = None):
        self.reranker = reranker or CrossEncoderReranker(use_llm=False)
        self.relevance_threshold = 0.1    # Drop anything below this score
        self.max_per_source = 3           # Max chunks from same source

    def validate(
        self,
        query: str,
        retrieval_output: Dict[str, Any],
        top_k: int = settings.rerank_top_k,
    ) -> Dict[str, Any]:
        """
        Validate and rerank retrieval results.

        Args:
            query: Original user query
            retrieval_output: Output dict from RetrievalAgent.retrieve()
            top_k: Final number of results to return

        Returns:
            Dict with:
              - validated_results: List[SearchResult]  (top_k)
              - removed_count: int
              - sources: List[str]
        """
        raw_results: List[SearchResult] = retrieval_output.get("results", [])
        logger.info(f"ValidationAgent: validating {len(raw_results)} candidates")

        if not raw_results:
            return {
                "validated_results": [],
                "removed_count": 0,
                "sources": [],
            }

        # Step 1: Cross-encoder reranking
        reranked = self.reranker.rerank(query, raw_results)

        # Step 2: Relevance threshold filter
        filtered = [r for r in reranked if r.hybrid_score >= self.relevance_threshold]
        removed_by_threshold = len(reranked) - len(filtered)

        # Step 3: Deduplicate near-identical chunks
        deduplicated = self._deduplicate(filtered)
        removed_by_dedup = len(filtered) - len(deduplicated)

        # Step 4: Source diversity — limit chunks per source
        diversified = self._enforce_source_diversity(deduplicated)

        # Step 5: Final top-k
        final = diversified[:top_k]

        sources = list({r.chunk.source for r in final})

        logger.info(
            f"ValidationAgent: {len(raw_results)} → {len(final)} results "
            f"(removed: {removed_by_threshold} low-relevance, "
            f"{removed_by_dedup} duplicates)"
        )

        return {
            "validated_results": final,
            "removed_count": removed_by_threshold + removed_by_dedup,
            "sources": sources,
        }

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove near-duplicate chunks (same leading 100 chars)."""
        seen_prefixes = set()
        unique = []
        for r in results:
            prefix = r.chunk.text[:100].strip().lower()
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique.append(r)
        return unique

    def _enforce_source_diversity(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Ensure no single source dominates the result set."""
        source_counts: Dict[str, int] = {}
        diverse = []
        for r in results:
            src = r.chunk.source
            count = source_counts.get(src, 0)
            if count < self.max_per_source:
                diverse.append(r)
                source_counts[src] = count + 1
        return diverse
