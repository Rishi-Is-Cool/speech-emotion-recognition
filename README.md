# Speech Emotion Recognition

Record up to 30 seconds of speech and get an estimate of which of six emotions it carries —
**angry, disgust, fear, happy, neutral, sad** — together with the evidence behind the answer.

**Live demo:** https://speech-emotion-theta.vercel.app

> **Read this first.** Measured on real, spontaneous voices this system is right about **one time
> in three (~33%)**, against 17% for guessing. It is a final-year research project, not a
> diagnostic tool, and must not be used for clinical, hiring or safety decisions.

## What it does

- **Emotion detected** — a headline prediction from a locked classifier on top of the
  [Emotion2Vec+](https://modelscope.cn/models/iic/emotion2vec_plus_base) speech encoder.
- **Where the answer came from** — the recording is split into 5-second windows and each window's
  push toward (or against) the prediction is shown on a timeline you can tap to replay.
- **Adaptive routing ("Green SER")** — a cheap MFCC + MLP model always runs first; a calibrated
  router decides whether the clip is worth the ~24× cost of the heavy Emotion2Vec+ branch.
  The page animates that decision live.
- **Hear your recording**, mobile-friendly design, and a page for contributing labelled clips.

Audio is processed in memory and discarded; nothing is stored.

## Repository layout

```
ser_frontend.html      the whole website (single file, no build step)
ser_api/               FastAPI service: inference, explainability, adaptive routing, tests
  ser_api/artifacts/   locked classifier head (tier1b_best_head.joblib)
Router/                light model, router, heavy head, scalers, serving spec, 3 demo clips
deploy/                packaging for Vercel (used for the live demo) and a Docker/HF Space option
```

## Run it locally

```bash
cd ser_api
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn ser_api.main:app --host 0.0.0.0 --port 8001
```

Then open http://localhost:8001 — the API serves the website itself. The Emotion2Vec+ encoder
(~1 GB) downloads from ModelScope on first start. Tests: `pytest tests/` from `ser_api/`.

See [`ser_api/README.md`](ser_api/README.md) for the API, input validation and the adaptive-routing
details, and [`deploy/build_vercel.py`](deploy/build_vercel.py) for how the live demo is built.

## Honest numbers

| | Accuracy |
|---|---|
| Locked classifier, real spontaneous voices | ~33% (project measurement) |
| Adaptive routing, acted evaluation corpus | 67.76% (heavy-only 67.84%, light-only 46.64%) |

The routing figures are on **acted** speech; they do not transfer to real voices. On real voices
the light model is close to chance, which is why the router sends most clips to the heavy branch.

## Datasets and licences

Models were trained on CREMA-D, RAVDESS, IEMOCAP and MELD. The three demo clips in `Router/` come
from CREMA-D (Open Database License) and RAVDESS (CC BY-NC-SA 4.0) and are included for
non-commercial demonstration with attribution to their authors. Emotion2Vec+ is by the
FunASR/ModelScope team.

## Team

- Rishikesh ([@Rishi-Is-Cool](https://github.com/Rishi-Is-Cool))
- Maitry Mahesh Mohite ([@maitry-mohite](https://github.com/maitry-mohite))
- Tanvi Yerram ([@tanviyerram08](https://github.com/tanviyerram08))
