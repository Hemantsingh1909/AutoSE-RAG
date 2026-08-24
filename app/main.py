import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.rag.hybrid_retriever import HybridRetriever
from app.engine.workflow import AgenticWorkflow
from app.hitl.session_store import HITLSessionStore
from app.eval.benchmark_data import BENCHMARK_REQUIREMENTS
from app.eval.experiment_runner import ExperimentRunner

app = FastAPI(
    title="AI Software Engineering Assistant - Research Edition (v2)",
    version="0.2.0",
    description="Research-grade Hybrid RAG + Multi-Agent Self-Refining Pipeline with Human-in-the-Loop & PhD Benchmark Suite."
)

retriever = HybridRetriever()
workflow = AgenticWorkflow(retriever)
hitl_store = HITLSessionStore()
experiment_runner = ExperimentRunner()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AnalyzeRequest(BaseModel):
    requirement: str = Field(min_length=8, description="Software engineering requirement specification")
    top_k: int = Field(default=4, ge=1, le=10)
    mode: str = Field(default="hybrid", description="'hybrid' | 'dense' | 'tfidf'")
    generate: bool = True
    require_human_approval: bool = False
    max_revisions: int = Field(default=2, ge=0, le=5)


class HumanReviewRequest(BaseModel):
    feedback: Optional[str] = Field(default="", description="Reviewer feedback or critique")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"name": app.title, "version": app.version, "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "knowledge_base_chunks": len(retriever.docs),
        "dense_embeddings_enabled": retriever.dense.index is not None,
    }


@app.post("/retrieve")
def retrieve_evidence(req: AnalyzeRequest):
    evidence = retriever.retrieve(req.requirement, top_k=req.top_k, mode=req.mode)
    return {
        "requirement": req.requirement,
        "mode": req.mode,
        "top_k": req.top_k,
        "evidence": evidence,
    }


@app.post("/api/agent/pipeline/run")
def run_agentic_pipeline(req: AnalyzeRequest):
    """
    Executes the multi-agent pipeline:
    Retriever Agent -> CodeGen Agent -> TestGen Agent -> [HITL Gateway] -> Sandbox Executor -> Critic Agent -> Refiner
    """
    result = workflow.run_pipeline(
        requirement=req.requirement,
        top_k=req.top_k,
        max_revisions=req.max_revisions,
        require_human_approval=req.require_human_approval,
    )

    # If pending human review, create HITL session
    if result.get("status") == "PENDING_APPROVAL":
        session_id = hitl_store.create_session(result)
        result["session_id"] = session_id

    return result


@app.get("/api/hitl/sessions")
def list_hitl_sessions():
    return {"sessions": hitl_store.list_sessions()}


@app.get("/api/hitl/sessions/{session_id}")
def get_hitl_session(session_id: str):
    sess = hitl_store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.post("/api/hitl/sessions/{session_id}/approve")
def approve_hitl_session(session_id: str):
    try:
        updated = hitl_store.approve_session(session_id)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/hitl/sessions/{session_id}/reject")
def reject_hitl_session(session_id: str, body: HumanReviewRequest):
    try:
        updated = hitl_store.reject_and_refine(session_id, feedback=body.feedback or "Revise implementation to meet safety criteria.")
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/eval/benchmark")
def get_benchmark_suite():
    return {"benchmark_requirements": BENCHMARK_REQUIREMENTS}


@app.post("/api/eval/run_retrieval_benchmark")
def run_retrieval_benchmark(limit: int = 15):
    """
    Experiment 1: Evaluates Sparse (TF-IDF) vs Dense (FAISS) vs Hybrid (RRF).
    """
    return experiment_runner.run_retrieval_experiment(max_items=min(limit, len(BENCHMARK_REQUIREMENTS)))


@app.post("/api/eval/run_generation_benchmark")
def run_generation_benchmark(limit: int = 15):
    """
    Experiment 2: Evaluates M0 (Raw LLM) vs M1 (Standard RAG) vs M2 (Agentic RAG) against Hidden Oracle Tests.
    """
    return experiment_runner.run_generation_experiment(max_items=min(limit, len(BENCHMARK_REQUIREMENTS)))


@app.post("/api/eval/run_comparative")
def run_comparative_benchmark(limit: int = 15):
    """
    Runs both Experiment 1 (Retrieval) and Experiment 2 (Generation & Hidden Oracle Verification).
    """
    results = experiment_runner.run_all_experiments(max_items=min(limit, len(BENCHMARK_REQUIREMENTS)))
    return results
