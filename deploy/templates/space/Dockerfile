# Speech Emotion Recognition API + web page, as a Hugging Face Space (Docker SDK).
# Free "CPU basic" hardware (2 vCPU, 16 GB RAM) runs it; the encoder needs ~1.2 GB.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements.txt

# Bake the Emotion2Vec+ encoder into the image. Otherwise the first visitor after every restart
# waits for a ~1 GB download from ModelScope.
RUN python -c "from funasr import AutoModel; AutoModel(model='iic/emotion2vec_plus_base', hub='ms', disable_update=True)"

COPY --chown=user . .

ENV SER_API_CONFIG=/home/user/app/config.yaml \
    SER_API_MAX_CONCURRENT=1
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "ser_api.main:app", "--host", "0.0.0.0", "--port", "7860"]
