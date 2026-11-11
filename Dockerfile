# RetinaAI — CPU inference container (Azure Container Apps / any Docker host)
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-space.txt

# Bake model weights into the image so scale-from-zero starts need no download
ARG WEIGHTS_REPO=AdityaPathanjalii/retinaai-weights
ENV WEIGHTS_REPO=${WEIGHTS_REPO}
RUN python -c "import os; from huggingface_hub import hf_hub_download as d; r=os.environ['WEIGHTS_REPO']; [d(r,f) for f in ('seg_attention_unet.pth','cls_hybrid_effnetb4.pth')]"
ENV HF_HUB_OFFLINE=1

COPY --chown=user . .

ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s \
  CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
