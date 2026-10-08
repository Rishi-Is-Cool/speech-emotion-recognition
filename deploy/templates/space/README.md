---
title: Speech Emotion Recognition
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Hear what a voice is feeling, with the evidence behind it.
---

# Speech Emotion Recognition

Record 30–60 seconds of speech and get an estimate of which of six emotions it carries
(angry, disgust, fear, happy, neutral, sad), plus a per-window timeline showing which moments
drove the answer, and an adaptive-routing view showing whether a cheap model was trusted or
an expensive one was called in.

- **Real-voice accuracy is about 33%** across six classes (chance is 17%). This is a research
  demonstration, not a diagnostic tool. Not for clinical, hiring, or safety decisions.
- Audio is processed in memory and discarded. Nothing is stored.

The API is served alongside the page: `GET /health`, `POST /predict`, `POST /predict/adaptive`,
`GET /adaptive/info`, interactive docs at `/docs`.
