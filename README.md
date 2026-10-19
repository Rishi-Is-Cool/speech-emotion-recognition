# Speech Emotion Recognition

Record up to 30 seconds of speech and get an estimate of which of six emotions it carries —
**angry, disgust, fear, happy, neutral, sad** — together with the evidence behind the answer.

**Live demo:** https://speech-emotion-theta.vercel.app

> **Read this first.** Measured on 516 real voices from people the models never trained on, this
> system is right about **one time in three (~35%)**, against 17% for guessing. It is a final-year
> research project, not a diagnostic tool, and must not be used for clinical, hiring or safety
> decisions.

## What it does

- **One answer per recording**, from one of two models:
  - **light**: MFCC + a small MLP, milliseconds per clip;
  - **heavy**: the [Emotion2Vec+](https://modelscope.cn/models/iic/emotion2vec_plus_base) speech
    encoder plus a linear classifier.
- **Green Inference Engine.** The light model always runs first. A router trained on real voices
  decides whether the clip needs the ~24× cost of the heavy model, and the page animates that
  decision live. **Heavy only** mode skips routing.
- **Why this emotion.** A per-window timeline you can tap to replay, plus a plain-words
  explanation the page can read aloud.
- **Hear your recording**, mobile-friendly design, a page for contributing labelled clips, and a
  team page.

Audio is processed in memory and discarded; nothing is stored.

## Repository layout

```
ser_frontend.html      the whole website (single file, no build step)
ser_api/               FastAPI service: inference, explainability, adaptive routing, tests
  ser_api/artifacts/   heavy model's classifier head (tier1b_best_head.joblib)
Router/                v2 light model + router + serving_spec_v2.json (served), the v1 originals
                       (kept for rollback), scalers, results, 3 demo clips
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

Real voices: 516 clean recordings from 4 people, scored leave-one-person-out (notebook 31).

| | Real-voice accuracy | Heavy model skipped |
|---|---|---|
| Heavy only | 34.9% | 0% |
| Green Inference Engine (v2 light + router) | 34.1% | 31.4% |
| Light model alone | 28.3% | 100% |

The router's threshold (0.63) was chosen with Learn-then-Test, targeting a final answer that
differs from the heavy model's on at most 10% of clips. On new people it differed on 25.4%, so the
target was **not** met out of sample. Four people is a small test.

The original acted-speech system (notebook 28; still in `Router/` as v1) scored 67.76% on the acted
test set, against 67.84% heavy-only. On real voices its router sent every clip to the heavy model.

## Datasets and licences

Models were trained on CREMA-D, RAVDESS, IEMOCAP and MELD. The three demo clips in `Router/` come
from CREMA-D (Open Database License) and RAVDESS (CC BY-NC-SA 4.0) and are included for
non-commercial demonstration with attribution to their authors. Emotion2Vec+ is by the
FunASR/ModelScope team.

## Team

- Rishikesh Patil ([@Rishi-Is-Cool](https://github.com/Rishi-Is-Cool)): website, heavy model,
  adaptive routing research, hardware, deployment, debugging
- Maitry Mahesh Mohite ([@maitry-mohite](https://github.com/maitry-mohite)): data collection,
  preprocessing, HuBERT layer-wise / CKA research, hardware
- Tanvi Yerram ([@tanviyerram08](https://github.com/tanviyerram08)): data collection,
  preprocessing, acoustic explainability research, hardware, testing
- Project guide: Dr. Sheetal Patil (research guidance, real-voice dataset)
