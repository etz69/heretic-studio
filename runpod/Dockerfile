# CUDA runtime + PyTorch base suitable for Heretic (large image).
# Push to GHCR or Docker Hub, then set container_image in the studio UI.
#
# RunPod + this CUDA base are linux/amd64. Pass platform at build time (not a constant
# in FROM — BuildKit warns FromPlatformFlagConstDisallowed), e.g.:
#   docker build --platform linux/amd64 -t yourname/freedom-runpod:latest -f runpod/Dockerfile .

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0:11434

# bitsandbytes (Heretic) imports Triton; Triton may JIT-compile tiny CUDA helpers at import — needs gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Ollama (optional local GGUF workflows); chat in this stack uses Transformers on HF weights.
RUN curl -fsSL https://ollama.com/install.sh | sh || true

WORKDIR /app
COPY runpod/serve.py runpod/heretic_driver.py runpod/model_manifest.py runpod/entrypoint.sh /app/
COPY heretic /opt/heretic
RUN chmod +x /app/entrypoint.sh

RUN pip install --no-cache-dir \
    "uvicorn[standard]>=0.32" \
    "fastapi>=0.115" \
    "accelerate>=1.0" \
    "litellm>=1.55" \
    "sentencepiece>=0.2.0" \
    "tiktoken>=0.7.0" \
    "/opt/heretic[research]"

# Heretic + sidecar share this env. Transformers ~5.3 MXFP4 expects torch.accelerator (PyTorch 2.6+);
# the 2.5.1 base image lacks it — pin torch and upgrade the matching vision/audio wheels.
RUN pip install --no-cache-dir --upgrade \
    "torch>=2.6.0" \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Residual geometry export (--print-residual-geometry) needs research extras; disable with 0.
ENV HERETIC_PRINT_RESIDUAL_GEOMETRY=1

EXPOSE 8888 4000 11434

ENTRYPOINT ["/app/entrypoint.sh"]
