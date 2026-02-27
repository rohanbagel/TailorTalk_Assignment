"""FastAPI application – serves the Titanic chat agent over HTTP."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agent import ask_agent, _get_df
from backend.config import get_settings


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up: eagerly load the dataset so the first request is fast."""
    _get_df()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Titanic Chat Agent API",
    description="Ask questions about the Titanic dataset in plain English.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question in natural language")
    session_id: str = Field(default="default", description="Session ID for conversation memory")


class ChatResponse(BaseModel):
    text: str = Field(..., description="The agent's textual answer")
    charts: list[str] = Field(default_factory=list, description="Base64-encoded PNG chart images")


class HealthResponse(BaseModel):
    status: str
    dataset_rows: int
    dataset_columns: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint – also confirms the dataset is loaded."""
    df = _get_df()
    return HealthResponse(
        status="healthy",
        dataset_rows=df.shape[0],
        dataset_columns=df.shape[1],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a natural-language question and receive text + optional charts."""
    try:
        result = await ask_agent(req.question, session_id=req.session_id)
        return ChatResponse(text=result["text"], charts=result["charts"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entrypoint (for `python -m backend.main`) ────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
    )
