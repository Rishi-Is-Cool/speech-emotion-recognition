# SER API — Stage 6

A small FastAPI service that wraps the locked SER pipeline (Emotion2Vec+ → StandardScaler →
LogisticRegression, notebook 19's config C, 32.8% real-voice accuracy) plus the Stage 5
explainability timeline, behind three endpoints:

- `POST /predict` — upload an audio file, get back the emotion, confidence, per-class
  probabilities, and the per-window timeline (with an honest `attribution_exact` flag).
- `POST /predict/youtube` — same thing, but for a `youtube.com`/`youtu.be` link instead of an
  upload: give it `{"url": "..."}`, it downloads just the audio, runs the identical pipeline, and
  also returns the extracted audio (base64) so a frontend can replay timeline segments exactly
  like it does for an uploaded file. Needs `ffmpeg` installed — see section 1b.
- `GET /health` — is the service up and has the model finished loading.
- `GET /model/version` — which model/classifier is currently loaded.

This is the same math as notebook 23, just running as a normal Python service instead of a
Colab notebook, so a device or web app can call it.

## 1. Get the model artifacts onto this machine

This runs locally (or on a server) — not in Colab — so the model files need to be downloaded
from Drive once, not re-downloaded per request:

1. Download `tier1b_best_head.joblib` from Drive:
   `state/notebook19_tier1b_iemocap/tier1b_best_head.joblib`
2. Put it at `ser_api/artifacts/tier1b_best_head.joblib` (create the `artifacts/` folder), or
   anywhere else — just update `classifier.artifact_path` in `ser_api/config.yaml` to point at it.
3. Leave the `encoder` section in `config.yaml` as-is — it matches what `verdict.json` printed
   throughout the project (`iic/emotion2vec_plus_base`, hub `ms`). The encoder itself downloads
   automatically from ModelScope the first time the service starts (same as every notebook did),
   so the first startup will take a minute or two.

## 1b. `ffmpeg` (only needed for `/predict/youtube`)

`yt-dlp` needs a real `ffmpeg` binary to extract a YouTube video's audio track:

- **Windows**: `winget install Gyan.FFmpeg` (or download from https://ffmpeg.org/download.html),
  then open a new terminal so PATH picks it up.
- **macOS**: `brew install ffmpeg`.
- **Linux**: `apt install ffmpeg` (or your distro's package manager).

If it can't be put on PATH, set `youtube.ffmpeg_location` in `config.yaml` to its folder instead.
`/predict` (file upload) does **not** need ffmpeg — only the YouTube endpoint does.

## 2. Install and run

```bash
cd ser_api
pip install -r requirements.txt

# from inside ser_api/
uvicorn ser_api.main:app --host 0.0.0.0 --port 8000
```

First startup loads the encoder + classifier once (can take a minute for the encoder download/
load) — after that, `/predict` calls are fast because nothing is reloaded per request.

Check it's alive:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/version
```

## 3. Call it

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/a/recording.wav"
```

Or with the included script, which also pretty-prints the timeline:

```bash
python scripts/manual_test.py /path/to/a/recording.wav
```

Example response shape:

```json
{
  "request_id": "a1b2c3d4",
  "model_version": "emotion2vec_plus_base+9f3a1c2b8e7d",
  "duration_sec": 44.0,
  "emotion": "angry",
  "confidence": 0.95,
  "probabilities": {"angry": 0.95, "disgust": 0.01, "...": 0.0},
  "timeline": {
    "window_sec": 5.0,
    "n_windows": 9,
    "attribution_exact": false,
    "reconstruction_error": 2.18,
    "top_window_idx": 0,
    "windows": [{"window_idx": 0, "start_s": 0.0, "end_s": 5.0, "contribution": 6.7,
                 "contribution_pct": 29.5, "pitch_hz": 200.0, "energy_rms": 0.0775}, "..."],
    "explanation_text": "Predicted 'angry' (95% confidence). ..."
  },
  "warnings": ["This recording was split into multiple windows. ..."],
  "latency_ms": {"validate_ms": 3.1, "predict_ms": 210.4, "explain_ms": 1840.2, "total_ms": 2053.7}
}
```

## 4. Input validation

Bad uploads get a specific `400` with an `error_code`, never a raw `500`:

| error_code | meaning |
|---|---|
| `empty_upload` / `empty_audio` | nothing was uploaded, or the file decoded to zero samples |
| `unreadable_audio` | not a format `soundfile` can decode (convert mp3/m4a to wav first) |
| `too_short` | shorter than `audio.min_duration_sec` in config.yaml (default 1.0s) |
| `too_long` | longer than `audio.max_duration_sec` in config.yaml (default 120s) |
| `silent_audio` | RMS energy below `audio.silence_rms_threshold` |
| `file_too_large` | bigger than `server.max_upload_mb` |

`/predict/youtube` adds:

| error_code | meaning |
|---|---|
| `empty_url` | no `url` field, or it was blank |
| `invalid_youtube_url` | not a `youtube.com`/`youtu.be` link — scoped to YouTube on purpose, not a general downloader |
| `youtube_unavailable` | private, age-restricted, region-locked, or removed video |
| `ffmpeg_missing` | ffmpeg isn't installed/on PATH — see section 1b |
| `youtube_download_failed` | anything else that went wrong extracting audio |
| `too_short` / `too_long` | same limits as uploads, checked from YouTube's own duration metadata **before** downloading |

## 5. Why `attribution_exact` matters (read before wiring up a frontend)

Per `SER_Stage5_Explainability_Finding.md`: the per-window contribution numbers are **exactly**
the classifier's own arithmetic only when a clip is short enough to be a single window. On a real
multi-window recording (anything longer than `window_sec`, i.e. most real 30–60s device
recordings), independently re-embedding each window doesn't perfectly reconstruct the whole-clip
embedding — measured reconstruction error 2.18 on a real 44s clip. The API surfaces this itself:
`timeline.attribution_exact` is `false` whenever `n_windows > 1`, and a plain-English `warnings`
entry is included. **A frontend showing the timeline should say "approximate supporting
evidence," never "exact,"** whenever this flag is false — don't just show the raw percentages
without that caveat.

The headline `emotion`/`confidence`/`probabilities` fields are unaffected by any of this — they're
computed the same way everywhere else in the project (whole-clip embedding straight into the
locked classifier).

## 6. Testing

```bash
pytest tests/ -v
```

The test suite mocks the Emotion2Vec+ encoder (no GPU, no network, no real model download needed)
with a content-deterministic mock, the same discipline used in every notebook's test.py in this
project — so it exercises the real windowing, attribution, and validation logic end to end, just
against a fake encoder. It covers: health/version endpoints, a short single-window clip (must be
exact), a long multi-window clip (must be flagged approximate, windows must tile the clip with no
gaps or overlaps), a correctly-classified clean tone, and all six input-validation error paths.

This test suite passing does **not** confirm the real Emotion2Vec+ encoder is wired correctly —
run one real request against a running server (`scripts/manual_test.py`) once the real artifacts
are in place, and compare against a clip you already know the answer for (e.g. `angry_big_clip.wav`
from Stage 5), before trusting the service.

`/predict/youtube`'s tests mock `ser_api.youtube.fetch_audio` the same way (no real network call,
no real yt-dlp/ffmpeg needed for CI) — they check the URL-scoping, the error-code mapping, and
that the returned `audio_base64` round-trips correctly. They do **not** confirm a real YouTube
download works; that was verified manually against a real, stable public video
(`https://www.youtube.com/watch?v=jNQXAC9IVRw`, YouTube's own "Me at the zoo" test clip) end to
end through both `curl` and the actual frontend page before shipping this.

## 7. YouTube scope and responsibility

`/predict/youtube` is deliberately restricted to `youtube.com`/`youtu.be` URLs — it is a
convenience for testing this SER pipeline against found speech, not a general-purpose media
downloader. Downloading a video's audio this way may conflict with YouTube's Terms of Service
depending on how you use it; this tool is meant for personal, local analysis, and the API surfaces
that as a `warnings` entry on every YouTube-sourced response. You're responsible for how you use
it — don't redistribute downloaded audio.

## 8. Adaptive routing (additive — the locked classifier is untouched)

A second, optional path that decides **per recording** whether a cheap model is good enough or
whether the clip is worth paying ~24× more compute for the heavy one:

- **Light branch** — MFCC + a small MLP. Runs on every clip (~0.052s).
- **Router** — logistic regression over 12 acoustic-complexity features + 3 confidence signals
  from the light branch. Escalates when `P(heavy) >= 0.375`.
- **Heavy branch** — Emotion2Vec+ + logistic regression (~1.235s extra). Runs **only** on escalation.

Point `adaptive.artifacts_dir` in `config.yaml` at the `Router/` folder. Leave it `null` to
disable the feature — the service still starts and `/predict` is unaffected.

| Endpoint | What it does |
|---|---|
| `POST /predict/adaptive` | Routed decision for an uploaded clip; returns an `adaptive_routing` block. `heavy_*` fields are `null` unless it escalated. |
| `GET /adaptive/info` | Certified threshold, published evaluation metrics (read from `notebook28_results.json`), and demo-clip metadata. |
| `GET /adaptive/demo/{case}/audio` | One of the three demo WAVs (`easy`, `borderline`, `hard`). |

### The trap that fails silently

**The two branches consume different audio from the same recording.** Swapping them raises no
error — the models just return confident, wrong numbers.

| | Light branch | Router (ASCA) |
|---|---|---|
| Audio | **full length** | **exactly 3.00s** (zero-padded / first-3s truncated) |
| Extractor | `n_fft=400, hop=160` | librosa defaults `n_fft=2048, hop=512` |
| Dimensions | 72 | 12 |

`tests/test_adaptive.py` asserts the serving path reproduces every number notebook 28 recorded
for the three demo clips (light prediction/confidence, router score, escalate/keep), and that the
encoder is **never** invoked when the router keeps a clip light.

### Things that are not tuning knobs

- **0.375 is Learn-then-Test certified**, not a parameter. It is read from `serving_spec.json`;
  if `adaptive.threshold` in `config.yaml` disagrees, the service refuses to start. Changing it
  needs a new calibration run.
- **Don't quote the routing accuracy as the product's accuracy.** 67.76% routed / 67.84%
  heavy-only / 46.64% light-only are all on the **acted** evaluation corpus. This project's own
  measured real-world figure is ~32.8%, and routing does not change it.
- The latency figures are from a Colab CPU on 3-second clips — compute cost, not device speed.

### Open item

Which 3 seconds of a 30–60s recording feed the router is **not validated**. The system was only
ever evaluated on clips whose first 3 seconds *were* the whole clip. The first-3s rule is a
placeholder; see the comment on `prepare_router_audio()`.

## 9. What's deliberately not here yet

- **Auth / rate limiting** — fine for local/demo use, add before any public deployment.
- **Async queueing for concurrent requests** — FastAPI handles concurrency, but a slow request
  (encoder inference) will still block others on a single worker. Run behind `uvicorn --workers N`
  or add a queue once real traffic shows this matters (see
  `SER_Audit_and_Production_Architecture.md` section 4).
- **GPU/ONNX export** — CPU inference is fine for a demo; the architecture doc flags this as an
  optimization for later, not a blocker.
- **Persistent request logging** — logs go to stdout only, structured with per-stage latency, no
  raw audio kept. Wire to a real log sink before production.
