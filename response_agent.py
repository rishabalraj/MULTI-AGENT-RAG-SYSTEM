# agents/response_agent.py
"""
Step 6: Response Agent
Third and final agent. Takes validated chunks, builds a context-aware
prompt, calls the LLM, and returns a structured answer with citations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.config import settings
from core.hybrid_search import SearchResult


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    page_number: Optional[int]
    section: Optional[str]
    snippet: str    # Short excerpt from the chunk

    def __str__(self) -> str:
        parts = [f"[{self.source}"]
        if self.page_number:
            parts.append(f", p.{self.page_number}")
        if self.section:
            parts.append(f", §{self.section}")
        parts.append("]")
        return "".join(parts)


@dataclass
class AgentResponse:
    answer: str
    citations: List[Citation]
    confidence: float         # 0.0 – 1.0, estimated from result scores
    query: str
    sources: List[str]        # Unique source document names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "source": c.source,
                    "page_number": c.page_number,
                    "section": c.section,
                    "snippet": c.snippet,
                    "citation_str": str(c),
                }
                for c in self.citations
            ],
            "confidence": round(self.confidence, 3),
            "query": self.query,
            "sources": self.sources,
        }


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

class PromptBuilder:
    """Constructs the LLM prompt from validated search results."""

    SYSTEM_PROMPT = """You are an expert enterprise knowledge assistant. 
Your role is to provide accurate, helpful answers based ONLY on the provided context documents.

Rules:
1. Answer ONLY using information from the provided context.
2. If the context doesn't contain enough information, say so clearly.
3. Always cite your sources using [SOURCE_N] notation in your answer.
4. Be concise and professional.
5. If multiple sources disagree, note the discrepancy.
6. Never make up information or hallucinate facts."""

    USER_TEMPLATE = """Context Documents:
{context}

---
Conversation History:
{history}

---
User Question: {query}

Please provide a comprehensive answer based on the context above. 
Use [SOURCE_N] citations inline where you reference specific information."""

    def build(
        self,
        query: str,
        validated_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, str]:
        """
        Build (system_prompt, user_prompt) for the LLM.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format context with source references
        context_parts = []
        for i, result in enumerate(validated_results, start=1):
            chunk = result.chunk
            source_label = f"[SOURCE_{i}]"
            meta = f"Source: {chunk.source}"
            if chunk.page_number:
                meta += f", Page {chunk.page_number}"
            if chunk.section:
                meta += f", Section: {chunk.section}"

            context_parts.append(
                f"{source_label} {meta}\n{chunk.text}\n"
            )

        context = "\n---\n".join(context_parts)

        # Format conversation history
        history_str = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-6:]:   # Last 3 exchanges
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)
        else:
            history_str = "No previous conversation."

        user_prompt = self.USER_TEMPLATE.format(
            context=context,
            history=history_str,
            query=query,
        )

        return self.SYSTEM_PROMPT, user_prompt


# ──────────────────────────────────────────────
# Response Agent
# ──────────────────────────────────────────────

class ResponseAgent:
    """
    Agent 3/3: Generates the final answer with citations.

    Workflow:
    1. Build structured prompt with context documents
    2. Call LLM to generate the answer
    3. Extract citations from the response
    4. Compute confidence score from retrieval scores
    5. Return structured AgentResponse
    """

    def __init__(self):
        self._llm = None
        self.prompt_builder = PromptBuilder()

    def _get_llm(self):
        if self._llm is None:
            if settings.openai_api_key:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.max_tokens,
                    api_key=settings.openai_api_key,
                )
            else:
                from langchain_community.llms import Ollama
                self._llm = Ollama(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                )
        return self._llm

    def generate(
        self,
        query: str,
        validation_output: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """
        Generate a final answer with citations.

        Args:
            query: Original user query
            validation_output: Dict from ValidationAgent.validate()
            conversation_history: List of {"role": "user"/"assistant", "content": "..."}

        Returns:
            AgentResponse with answer, citations, confidence
        """
        validated_results: List[SearchResult] = validation_output.get(
            "validated_results", []
        )
        sources: List[str] = validation_output.get("sources", [])

        logger.info(
            f"ResponseAgent: generating answer from {len(validated_results)} chunks"
        )

        # Edge case: no results
        if not validated_results:
            return AgentResponse(
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing or contact "
                    "your administrator to ingest the relevant documents."
                ),
                citations=[],
                confidence=0.0,
                query=query,
                sources=[],
            )

        # Step 1: Build prompt
        system_prompt, user_prompt = self.prompt_builder.build(
            query=query,
            validated_results=validated_results,
            conversation_history=conversation_history,
        )

        # Step 2: Call LLM
        try:
            llm = self._get_llm()
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer_text = f"Error generating response: {str(e)}"

        # Step 3: Build citations
        citations = self._build_citations(validated_results)

        # Step 4: Estimate confidence from avg hybrid score
        scores = [r.hybrid_score for r in validated_results]
        confidence = float(sum(scores) / len(scores)) if scores else 0.0
        # Normalize if cross-encoder scores are 0–1, otherwise cap at 1.0
        confidence = min(confidence, 1.0)

        logger.info(
            f"ResponseAgent: answer generated "
            f"(confidence={confidence:.2f}, sources={sources})"
        )

        return AgentResponse(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            query=query,
            sources=sources,
        )

    def _build_citations(self, results: List[SearchResult]) -> List[Citation]:
        """Extract citation objects from result chunks."""
        citations = []
        for r in results:
            chunk = r.chunk
            snippet = chunk.text[:200].replace("\n", " ").strip()
            if len(chunk.text) > 200:
                snippet += "..."

            citations.append(
                Citation(
                    source=chunk.source,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    snippet=snippet,
                )
            )
        return citations
