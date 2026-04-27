from typing import Literal

from pydantic import BaseModel, Field


class CredentialsIn(BaseModel):
    runpod_api_key: str | None = None
    huggingface_token: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    s3_bucket: str | None = None


class CredentialsOut(BaseModel):
    runpod_configured: bool
    huggingface_configured: bool
    s3_configured: bool


class JobCreate(BaseModel):
    name: str | None = None
    hf_model: str = Field(..., examples=["Qwen/Qwen3-4B-Instruct-2507"])
    job_mode: Literal["HERETIC", "CHAT", "BOTH"] = Field(
        default="BOTH",
        description='Job behavior: "HERETIC" (run decensoring), "CHAT" (download/snapshot only), or "BOTH" (default).',
    )
    gpu_type_id: str
    cloud_type: Literal["COMMUNITY", "SECURE", "ALL"] = "COMMUNITY"
    container_image: str = Field(
        default="YOUR_REGISTRY/freedom-runpod:latest",
        description="Docker image you pushed (CUDA + entrypoint).",
    )
    skip_original_hf_snapshot: bool = Field(
        default=False,
        description="Sets SKIP_ORIGINAL_HF_SNAPSHOT=1 on the pod: skip pre-run HF snapshot (saves time/disk); original-weights chat disabled.",
    )
    n_trials: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Number of Heretic optimization trials.",
    )
    refusal_markers_append: list[str] = Field(
        default_factory=list,
        description="Additional refusal markers appended to Heretic defaults (case-insensitive substring match).",
    )
    volume_gb: int = Field(
        default=100,
        ge=1,
        le=2000,
        description="RunPod pod volume size (GB) for /workspace (HF cache, Heretic output, logs).",
    )


class StudioModelLoadBody(BaseModel):
    """Preload chat weights on the pod sidecar without sending a chat message."""

    studio_weights: Literal["decensored", "original"] = "decensored"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    studio_weights: Literal["decensored", "original"] | None = Field(
        default=None,
        description='Which weights the pod sidecar should load: "decensored" merged dir or "original" HF snapshot.',
    )
