"""FastAPI service wrapping the locked SER pipeline: POST /predict, POST /predict/youtube,
GET /health, GET /model/version. Per SER_Audit_and_Production_Architecture.md section 4:
- input validation with graceful, specific errors -- never a raw 500 for a bad upload
- structured logs with per-stage latency, model version, confidence
- no raw audio persisted beyond what's needed to serve the response
- model loaded once at startup, never per-request
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from ser_api.adaptive import MODES, AdaptivePackage, analyze as analyze_adaptive, route as route_adaptive
from ser_api.config import Config
from ser_api.errors import SERInputError
from ser_api.explainability import build_timeline
from ser_api.inference import predict
from ser_api.model_package import ModelPackage
from ser_api.preprocessing import load_and_validate
from ser_api.youtube import fetch_audio as fetch_youtube_audio

logger = logging.getLogger("ser_api")

_state: dict = {}  # holds the loaded ModelPackage + config; set once at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config.load()
    logging.basicConfig(level=config.log_level.upper())
    logger.info("Loading model package...")
    _state["config"] = config
    _state["model"] = ModelPackage.load(config)

    # Adaptive routing is an optional add-on. If its artifacts aren't present the rest of the
    # service must still come up -- /predict does not depend on it.
    _state["adaptive"] = None
    if config.adaptive_artifacts_dir and config.adaptive_artifacts_dir.exists():
        try:
            _state["adaptive"] = AdaptivePackage.load(config)
        except Exception:
            logger.exception("Adaptive routing failed to load; continuing without it.")
    else:
        logger.info("Adaptive routing disabled (no artifacts_dir configured or it does not exist).")

    logger.info("Startup complete, model_version=%s", _state["model"].version_tag)
    yield
    _state.clear()


app = FastAPI(title="SER API", version="1.0", lifespan=lifespan)

# Local/dev CORS: a locally-opened HTML file (Origin: null) or a dev-server page on a
# different port needs this to call the API from the browser. Permissive on purpose --
# tighten allow_origins before any real deployment.
# SER_API_CORS_ORIGINS is a comma-separated allow-list for a public deployment (e.g. the
# frontend's own domain). Read from the environment rather than config.yaml because middleware is
# installed at import, before the config is loaded in lifespan.
_cors = [o.strip() for o in os.environ.get("SER_API_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YouTubeRequest(BaseModel):
    url: str


def _require_ready() -> tuple[ModelPackage, Config]:
    model: ModelPackage | None = _state.get("model")
    config: Config | None = _state.get("config")
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    return model, config


def _run_pipeline(y, config: Config, model: ModelPackage, request_id: str) -> dict:
    """predict + explain, shared by every audio source (upload, YouTube, ...). Returns a dict
    with the two stage timings (seconds) plus the assembled response fields that don't depend
    on how the audio got here."""
    t_start = time.monotonic()
    prediction = predict(y, config.sample_rate, model)
    t_predict = time.monotonic()

    timeline = build_timeline(y, config.sample_rate, model, prediction)
    t_explain = time.monotonic()

    logger.info(
        "predict request_id=%s emotion=%s confidence=%.3f n_windows=%d attribution_exact=%s",
        request_id, prediction.emotion, prediction.confidence,
        timeline.n_windows, timeline.attribution_exact,
    )

    warnings = []
    if not timeline.attribution_exact:
        warnings.append(
            "This recording was split into multiple windows. Per-window contribution "
            "percentages are approximate supporting evidence, not exact arithmetic -- see "
            "SER_Stage5_Explainability_Finding.md for why."
        )

    return {
        "predict_sec": t_predict - t_start,
        "explain_sec": t_explain - t_predict,
        "warnings": warnings,
        "body": {
            "request_id": request_id,
            "model_version": prediction.model_version,
            "emotion": prediction.emotion,
            "confidence": round(prediction.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prediction.probabilities.items()},
            "timeline": {
                "window_sec": timeline.window_sec,
                "n_windows": timeline.n_windows,
                "attribution_exact": timeline.attribution_exact,
                "reconstruction_error": round(timeline.reconstruction_error, 6),
                "top_window_idx": timeline.top_window_idx,
                "windows": timeline.windows,
                "explanation_text": timeline.explanation_text,
            },
        },
    }


# The model calls are seconds of blocking CPU work. Run directly inside these async endpoints,
# they froze the whole event loop: while one visitor's analysis ran, nobody else could even load
# /health, so the site looked down. They now run on a worker thread, behind a semaphore so the
# shared encoder still sees one inference at a time (the safe default on a CPU host -- raise
# SER_API_MAX_CONCURRENT only on hardware with room for it).
_inference_slots = asyncio.Semaphore(max(1, int(os.environ.get("SER_API_MAX_CONCURRENT", "1"))))


async def _off_loop(fn, *args):
    return await run_in_threadpool(fn, *args)


async def _infer(fn, *args):
    async with _inference_slots:
        return await run_in_threadpool(fn, *args)


@app.exception_handler(SERInputError)
async def handle_input_error(request: Request, exc: SERInputError):
    return JSONResponse(status_code=400, content={"error_code": exc.code, "message": exc.message})


@app.get("/health")
async def health():
    model: ModelPackage | None = _state.get("model")
    config: Config | None = _state.get("config")
    return {
        "status": "ok" if model is not None else "starting",
        "model_loaded": model is not None,
        # lets the frontend hide what this particular deployment has switched off
        "features": {
            "youtube": bool(config and config.youtube_enabled),
            "adaptive": _state.get("adaptive") is not None,
        },
        # the page trims recordings to these before uploading instead of failing after the upload
        "limits": {
            "max_duration_sec": config.max_duration_sec if config else None,
            "max_upload_mb": config.max_upload_mb if config else None,
        },
    }


@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the web page from the API itself, so one URL is the whole product (and the page can
    find the API on its own origin with no configuration)."""
    config: Config | None = _state.get("config")
    if config and config.frontend_path and config.frontend_path.is_file():
        return FileResponse(config.frontend_path, media_type="text/html")
    return JSONResponse({"service": "SER API", "docs": "/docs", "health": "/health"})


@app.get("/model/version")
async def model_version():
    model: ModelPackage | None = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    return {
        "model_version": model.version_tag,
        "encoder_model_id": model.config.encoder_model_id,
        "classifier_config": model.config.classifier_expected_config,
        "classes": model.classes,
        "window_sec": model.window_sec,
        "sample_rate": model.sample_rate,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    model, config = _require_ready()
    request_id = uuid.uuid4().hex[:8]
    t0 = time.monotonic()

    raw_bytes = await file.read()
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise SERInputError(
            "file_too_large",
            f"Upload is larger than the {config.max_upload_mb}MB limit this service accepts.",
        )

    t_read = time.monotonic()
    y, duration_sec = await _off_loop(load_and_validate, raw_bytes, config)
    t_validate = time.monotonic()

    result = await _infer(_run_pipeline, y, config, model, request_id)
    t_done = t_validate + result["predict_sec"] + result["explain_sec"]

    latency_ms = {
        "validate_ms": round((t_validate - t_read) * 1000, 1),
        "predict_ms": round(result["predict_sec"] * 1000, 1),
        "explain_ms": round(result["explain_sec"] * 1000, 1),
        "total_ms": round((t_done - t0) * 1000, 1),
    }

    body = result["body"]
    body["duration_sec"] = round(duration_sec, 2)
    body["warnings"] = result["warnings"]
    body["latency_ms"] = latency_ms
    return body


@app.post("/predict/youtube")
async def predict_youtube_endpoint(req: YouTubeRequest):
    """Same pipeline as /predict, but the audio comes from a YouTube link instead of an
    upload -- see ser_api/youtube.py for why this is scoped to YouTube only and what it does
    and doesn't do. The extracted WAV is returned back to the caller (base64) so a browser
    frontend can decode it locally for timeline segment replay, exactly like an uploaded file."""
    model, config = _require_ready()
    if not config.youtube_enabled:
        raise HTTPException(status_code=404, detail="YouTube analysis is not enabled on this server.")
    request_id = uuid.uuid4().hex[:8]
    t0 = time.monotonic()

    wav_bytes, yt_info = await _off_loop(fetch_youtube_audio, req.url, config)
    t_download = time.monotonic()

    y, duration_sec = await _off_loop(load_and_validate, wav_bytes, config)
    t_validate = time.monotonic()

    result = await _infer(_run_pipeline, y, config, model, request_id)
    t_done = t_validate + result["predict_sec"] + result["explain_sec"]

    latency_ms = {
        "download_ms": round((t_download - t0) * 1000, 1),
        "validate_ms": round((t_validate - t_download) * 1000, 1),
        "predict_ms": round(result["predict_sec"] * 1000, 1),
        "explain_ms": round(result["explain_sec"] * 1000, 1),
        "total_ms": round((t_done - t0) * 1000, 1),
    }

    warnings = result["warnings"] + [
        "This audio was downloaded from a YouTube link for personal analysis. You are "
        "responsible for respecting the source video's copyright and YouTube's Terms of Service."
    ]

    logger.info(
        "predict_youtube request_id=%s url=%s duration_sec=%.2f %s",
        request_id, req.url, duration_sec, latency_ms,
    )

    body = result["body"]
    body["source"] = {"type": "youtube", "url": req.url, "title": yt_info.get("title")}
    body["duration_sec"] = round(duration_sec, 2)
    body["warnings"] = warnings
    body["latency_ms"] = latency_ms
    body["audio_base64"] = base64.b64encode(wav_bytes).decode("ascii")
    return body


# ---------------------------------------------------------------------------------------------
# Adaptive routing (additive -- the locked classifier above is untouched by all of this)
# ---------------------------------------------------------------------------------------------
def _require_adaptive() -> AdaptivePackage:
    adaptive: AdaptivePackage | None = _state.get("adaptive")
    if adaptive is None:
        raise HTTPException(
            status_code=503,
            detail="Adaptive routing is not available on this server. Point adaptive.artifacts_dir "
                   "in config.yaml at the Router folder and restart.",
        )
    return adaptive


@app.get("/adaptive/info")
async def adaptive_info():
    """What the routing pages need to describe themselves: the threshold, the published evaluation
    numbers (read from the notebooks' result files, never from memory), and the demo clips."""
    adaptive = _require_adaptive()
    results_path = adaptive.artifacts_dir / "notebook28_results.json"
    r = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    real_path = adaptive.artifacts_dir / "notebook31_summary.json"
    is_v2 = "heavy_branch" in adaptive.spec
    real = json.loads(real_path.read_text(encoding="utf-8")) if (is_v2 and real_path.exists()) else None

    lat = r.get("green_metrics", {}).get("latency_seconds_per_clip", {})
    always = sum(lat.get(k, 0.0) for k in
                 ("audio_decode_s", "mfcc_extract_s", "asca_extract_s", "light_head_s", "router_s"))
    escalated = lat.get("emotion2vec_plus_encoder_s", 0.0) + lat.get("heavy_head_s", 0.0)

    return {
        "available": True,
        "version": "v2" if is_v2 else "v1",
        "spec": adaptive.spec_name,
        "heavy_head": adaptive.heavy_name,
        "threshold": adaptive.threshold,
        "threshold_source": (
            "Learn-then-Test on 516 real clips (notebook 31)" if is_v2
            else "Learn-then-Test certified (notebook 28)"
        ),
        "classes": adaptive.classes,
        "router_audio_sec": 3.0,
        # notebook 31's leave-one-person-out results on real voices (v2 only)
        "real_voice": real,
        # notebook 28's acted-corpus evaluation of the ORIGINAL system; the cost figures below
        # (feature extraction vs encoder time) apply to both versions
        "metrics": {
            "light_test_accuracy": r.get("light_branch", {}).get("test_accuracy"),
            "heavy_test_accuracy": r.get("heavy_branch", {}).get("test_accuracy"),
            "routed_test_accuracy": r.get("risk_control", {}).get("test_accuracy"),
            "accuracy_retention_percent": r.get("risk_control", {}).get("accuracy_retention_percent"),
            "invocation_rate": r.get("risk_control", {}).get("test_invocation"),
            "certified_error_bound": r.get("risk_control", {}).get("alpha"),
            "certified_confidence": 1 - r.get("risk_control", {}).get("delta", 0.05),
            "always_paid_sec": round(always, 4),
            "escalated_extra_sec": round(escalated, 4),
            "cost_multiple": round(escalated / always, 1) if always else None,
        },
        # Every number above is on the ACTED evaluation corpus. The product's own measured
        # real-world figure is ~32.8%; the frontend is required to keep that visible.
        "corpus_caveat": (
            "These figures are on the acted evaluation corpus. This project measured about 32.8% "
            "accuracy on real spontaneous speech -- routing does not change that."
        ),
        "latency_caveat": (
            "Measured on a Colab CPU with 3-second clips, not on this hardware and not on 30-60s "
            "recordings. Read it as compute cost, not as device speed."
        ),
        # The manifest's recorded outputs belong to the v1 system, so they are only offered as
        # "expected" values when v1 is what is being served.
        "demo_clips": [
            {
                "case": c["case"],
                "true_label": c["true_label"],
                "source": "RAVDESS" if "ravdess" in c["exported_filename"].lower() else "CREMA-D",
                **({} if is_v2 else {"expected": {
                    "light_prediction": c["light_prediction"],
                    "light_confidence": c["light_confidence"],
                    "router_score": c["router_score_p_heavy"],
                    "escalated": c["escalated"],
                    "branch_used": c["branch_used"],
                    "final_prediction": c["final_prediction"],
                    "heavy_prediction": c["heavy_prediction"],
                    "heavy_confidence": c["heavy_confidence"],
                }}),
            }
            for c in adaptive.demo_clips
        ],
    }


@app.get("/adaptive/demo/{case}/audio")
async def adaptive_demo_audio(case: str):
    adaptive = _require_adaptive()
    clip = next((c for c in adaptive.demo_clips if c["case"] == case), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"No demo clip named {case!r}.")
    path = adaptive.artifacts_dir / clip["exported_filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Demo clip file is missing: {path.name}")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/predict/adaptive")
async def predict_adaptive_endpoint(file: UploadFile = File(...)):
    """The routed decision for an uploaded clip. Returns the locked model's result unchanged,
    plus an `adaptive_routing` block showing what the cheap branch said, what the router scored,
    and -- only when it escalated -- what the expensive branch said."""
    model, config = _require_ready()
    adaptive = _require_adaptive()
    request_id = uuid.uuid4().hex[:8]
    t0 = time.monotonic()

    raw_bytes = await file.read()
    if len(raw_bytes) > config.max_upload_mb * 1024 * 1024:
        raise SERInputError(
            "file_too_large",
            f"Upload is larger than the {config.max_upload_mb}MB limit this service accepts.",
        )

    y, duration_sec = await _off_loop(load_and_validate, raw_bytes, config)
    t_validate = time.monotonic()

    routing = await _infer(route_adaptive, y, config.sample_rate, adaptive, model)
    t_routed = time.monotonic()

    logger.info(
        "predict_adaptive request_id=%s duration_sec=%.2f light=%s router_score=%.4f branch=%s",
        request_id, duration_sec, routing["light_prediction"], routing["router_score"],
        routing["branch_used"],
    )

    return {
        "request_id": request_id,
        "duration_sec": round(duration_sec, 2),
        "adaptive_routing": routing,
        "latency_ms": {
            "validate_ms": round((t_validate - t0) * 1000, 1),
            "routed_ms": round((t_routed - t_validate) * 1000, 1),
            "total_ms": round((t_routed - t0) * 1000, 1),
        },
    }


# ---------------------------------------------------------------------------------------------
# What the website calls: ONE answer per recording, from the light or the heavy model
# ---------------------------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), mode: str = Form("gie")):
    """mode=gie   -> Green Inference Engine: light model, router, heavy model only if escalated.
    mode=heavy -> heavy model on the whole clip, no routing.
    Returns the answer, all six probabilities, the routing trace (gie only) and a per-window
    timeline built from whichever model gave the answer."""
    model, config = _require_ready()
    adaptive = _require_adaptive()
    if mode not in MODES:
        raise SERInputError("invalid_mode", f"mode must be one of {', '.join(MODES)}.")
    request_id = uuid.uuid4().hex[:8]
    t0 = time.monotonic()

    raw_bytes = await file.read()
    if len(raw_bytes) > config.max_upload_mb * 1024 * 1024:
        raise SERInputError(
            "file_too_large",
            f"Upload is larger than the {config.max_upload_mb}MB limit this service accepts.",
        )

    y, duration_sec = await _off_loop(load_and_validate, raw_bytes, config)
    t_validate = time.monotonic()

    body = await _infer(analyze_adaptive, y, config.sample_rate, adaptive, model, mode, config.window_sec)
    t_done = time.monotonic()

    logger.info(
        "analyze request_id=%s mode=%s duration_sec=%.2f branch=%s emotion=%s confidence=%.3f router=%s",
        request_id, mode, duration_sec, body["branch_used"], body["emotion"], body["confidence"],
        f"{body['routing']['router_score']:.4f}" if body["routing"] else "-",
    )

    return {
        "request_id": request_id,
        "duration_sec": round(duration_sec, 2),
        **body,
        "latency_ms": {
            "validate_ms": round((t_validate - t0) * 1000, 1),
            "analyze_ms": round((t_done - t_validate) * 1000, 1),
            "total_ms": round((t_done - t0) * 1000, 1),
        },
    }


@app.post("/youtube/audio")
async def youtube_audio_endpoint(req: YouTubeRequest):
    """Download a YouTube link's audio track and hand it back (base64 WAV), without analysing it.
    The page then treats it exactly like a recording and sends it to /analyze."""
    _, config = _require_ready()
    if not config.youtube_enabled:
        raise HTTPException(status_code=404, detail="YouTube analysis is not enabled on this server.")
    wav_bytes, yt_info = await _off_loop(fetch_youtube_audio, req.url, config)
    logger.info("youtube_audio url=%s bytes=%d", req.url, len(wav_bytes))
    return {
        "source": {"type": "youtube", "url": req.url, "title": yt_info.get("title")},
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
    }
