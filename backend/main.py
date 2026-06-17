"""
NBIS FastAPI service
====================
Loads the NBIS model, FAISS index, mapping, threshold, and SQLite DB ONCE
at startup and exposes:

    GET  /health            — liveness + system info
    POST /identify          — multipart image → identification + full DB record
    POST /identify-base64   — JSON {image: "<base64>"} → same as above
    GET  /record/{sid}      — fetch DB record by subject_id (diagnostic)

Serves the single-file frontend (index.html) at /.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from db import NBISDatabase
from model_loader import NBISSystem

# ─── Paths & config ─────────────────────────────────────────────────────────
BACKEND_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = BACKEND_DIR.parent
ARTIFACTS_DIR = Path(os.environ.get("NBIS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
DB_PATH       = Path(os.environ.get("NBIS_DB_PATH",       PROJECT_ROOT / "artifacts" / "nbis.db"))
FRONTEND_DIR  = PROJECT_ROOT / "frontend"
TOP_K_DEFAULT = int(os.environ.get("NBIS_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.environ.get("NBIS_MAX_UPLOAD_MB", "10"))

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("nbis.api")

# ─── Shared singletons ──────────────────────────────────────────────────────
nbis = NBISSystem(ARTIFACTS_DIR)
db   = NBISDatabase(DB_PATH)

_stats = {
    "total_scans": 0,
    "matches": 0,
    "no_matches": 0,
    "errors": 0,
    "failure_streak": 0,
    "started_at": time.time(),
}


# ─── Lifespan: load once at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading NBIS artifacts from %s", ARTIFACTS_DIR)
    try:
        nbis.load()
        log.info(
            "NBIS ready: %d vectors, dim=%d, threshold=%.4f",
            nbis.index.ntotal,
            nbis.prep_config.get("embedding_dim", 0),
            nbis.threshold,
        )
    except Exception as e:
        log.exception("Failed to load NBIS artifacts")
        nbis.ready = False
        nbis.load_error = str(e)

    if db.available:
        log.info("SQLite DB found at %s", DB_PATH)
    else:
        log.warning("SQLite DB NOT found at %s — responses will omit `database` block",
                    DB_PATH)
    yield
    log.info("Shutting down NBIS service.")


app = FastAPI(
    title="NBIS — Newborn Biometric Identification Service",
    description="Fingerprint identification via MobileNetV2 embeddings + FAISS + SQLite.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ────────────────────────────────────────────────────────────────
class Base64Request(BaseModel):
    image: str = Field(..., description="Base64-encoded image bytes "
                                        "(raw, or `data:image/*;base64,...`)")
    top_k: int | None = Field(None, ge=1, le=50)


# ─── Helpers ────────────────────────────────────────────────────────────────
def _record_result(result: dict) -> None:
    _stats["total_scans"] += 1
    if result["status"] == "MATCH":
        _stats["matches"] += 1
        _stats["failure_streak"] = 0
    else:
        _stats["no_matches"] += 1
        _stats["failure_streak"] += 1
        if _stats["failure_streak"] >= 5:
            log.warning("High NO_MATCH streak: %d consecutive",
                        _stats["failure_streak"])


def _decode_base64(s: str) -> bytes:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")


def _enrich_with_db(result: dict) -> None:
    """If the result is a MATCH, attach the joined DB record in-place."""
    if result.get("status") != "MATCH":
        return
    sid = result.get("subject_id")
    if not sid:
        return
    record = db.fetch_full_record(sid)
    result["database"] = record  # None if unavailable — the UI handles it


# ─── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    h = nbis.health()
    h["uptime_seconds"] = round(time.time() - _stats["started_at"], 1)
    h["status"] = "ok" if nbis.ready else "error"
    h["database_available"] = db.available
    h["database_path"] = str(DB_PATH)
    if not nbis.ready and hasattr(nbis, "load_error"):
        h["load_error"] = nbis.load_error
    return h


@app.get("/stats")
async def stats():
    return {**_stats, "uptime_seconds": round(time.time() - _stats["started_at"], 1)}


@app.get("/record/{subject_id}")
async def record(subject_id: str):
    """Diagnostic: fetch a DB record directly by subject_id."""
    rec = db.fetch_full_record(subject_id)
    if rec is None:
        raise HTTPException(status_code=404,
                            detail=f"No record for subject_id={subject_id}")
    return rec


@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = TOP_K_DEFAULT):
    if not nbis.ready:
        raise HTTPException(status_code=503, detail="NBIS not loaded.")
    t0 = time.perf_counter()
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413,
                                detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
        result = nbis.identify(data, top_k=top_k)
        _enrich_with_db(result)
    except HTTPException:
        _stats["errors"] += 1
        raise
    except Exception as e:
        _stats["errors"] += 1
        log.exception("Identification failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["filename"] = file.filename
    _record_result(result)
    return result


@app.post("/identify-base64")
async def identify_base64(req: Base64Request):
    if not nbis.ready:
        raise HTTPException(status_code=503, detail="NBIS not loaded.")
    t0 = time.perf_counter()
    try:
        data = _decode_base64(req.image)
        if not data:
            raise HTTPException(status_code=400, detail="Empty decoded image.")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413,
                                detail=f"Image exceeds {MAX_UPLOAD_MB} MB limit.")
        result = nbis.identify(data, top_k=req.top_k or TOP_K_DEFAULT)
        _enrich_with_db(result)
    except HTTPException:
        _stats["errors"] += 1
        raise
    except Exception as e:
        _stats["errors"] += 1
        log.exception("Base64 identification failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _record_result(result)
    return result


# ─── Frontend (single index.html) ───────────────────────────────────────────
if FRONTEND_DIR.exists():
    @app.get("/")
    async def root():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
