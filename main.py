# api/main.py
"""
Step 12: FastAPI Application Entry Point
CORS, lifespan events, middleware, and router registration.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from loguru import logger

from core.config import settings
from core.memory import get_memory_manager
from api.routes import router


# ──────────────────────────────────────────────
# Startup / Shutdown
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup: initialize DB tables and log config.
    Runs on shutdown: graceful cleanup.
    """
    logger.info("=" * 60)
    logger.info("  Enterprise Knowledge Assistant Starting Up")
    logger.info(f"  LLM Model    : {settings.llm_model}")
    logger.info(f"  Vector Store : {settings.vector_store_type}")
    logger.info(f"  Embedding    : {settings.embedding_model}")
    logger.info("=" * 60)

    # Create DB tables
    memory = get_memory_manager()
    await memory.create_tables()

    # Ensure data directories exist
    Path("./data/documents").mkdir(parents=True, exist_ok=True)
    Path("./data/uploads").mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)

    logger.info("Startup complete. Ready to serve requests.")
    yield

    logger.info("Shutting down gracefully.")


# ──────────────────────────────────────────────
# App Factory
# ──────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent Enterprise Knowledge Assistant",
        description=(
            "RAG-based enterprise assistant with hybrid search, "
            "multi-agent pipeline, RBAC, and conversation memory."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API Routes ──────────────────────────
    app.include_router(router, prefix="/api")

    # ── Static Files (frontend) ─────────────
    frontend_dir = Path("./frontend")
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()


# ──────────────────────────────────────────────
# Dev server entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
