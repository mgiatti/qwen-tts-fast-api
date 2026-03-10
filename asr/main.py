import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import TranscribeRequest, JobResponse, StatusResponse, JobStatus
from .worker import ASRJobQueue


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup – create singleton and start background worker
    queue = ASRJobQueue()
    queue.start_worker()
    yield
    # Shutdown – signal worker to stop
    queue.stop_worker()


app = FastAPI(
    title="Qwen3 ASR API",
    description=(
        "Async speech-to-text API powered by Qwen3-ASR-1.7B. "
        "Supports local file uploads and remote URLs. "
        "Long audio (e.g. 40-min recordings) is automatically chunked."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/transcribe", response_model=JobResponse, summary="Transcribe audio from a URL")
async def transcribe_url(request: TranscribeRequest):
    """
    Submit a transcription job for audio at a remote URL.

    - **audio_url**: publicly accessible audio URL (wav, mp3, flac, …)
    - **language**: optional ISO language name (e.g. `"English"`, `"Chinese"`).
      Leave `null` for automatic detection.
    - **return_timestamps**: if `true`, per-word timestamps are included (requires
      the ForcedAligner model to be enabled on the server).
    - **chunk_duration_seconds**: seconds per chunk for local files (ignored here).
    """
    if not request.audio_url:
        raise HTTPException(status_code=400, detail="audio_url is required for this endpoint.")

    queue = ASRJobQueue()
    job_id = queue.add_job(request, audio_path=None)
    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.post(
    "/transcribe/upload",
    response_model=JobResponse,
    summary="Transcribe an uploaded audio file",
)
async def transcribe_upload(
    file: UploadFile = File(..., description="Audio file to transcribe (wav, mp3, flac, ogg, …)"),
    language: Optional[str] = Form(None, description="Language name or None for auto-detection"),
    return_timestamps: bool = Form(False, description="Return per-word timestamps"),
    chunk_duration_seconds: int = Form(
        300,
        description=(
            "Chunk size in seconds for long audio. "
            "0 disables chunking (may OOM on very long files). "
            "Default is 300 s (5 min)."
        ),
    ),
):
    """
    Upload an audio file and submit it for transcription.

    The file is saved temporarily; it is deleted automatically once the
    worker has finished processing the job.

    **40-min audio**: will be split into ~5-min chunks automatically.
    Each chunk is transcribed independently and results are concatenated
    with globally-correct timestamps.
    """
    # Save upload to a temp file so the worker can access it
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="asr_upload_") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    request = TranscribeRequest(
        language=language,
        return_timestamps=return_timestamps,
        chunk_duration_seconds=chunk_duration_seconds,
    )

    queue = ASRJobQueue()
    job_id = queue.add_job(request, audio_path=tmp_path)
    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.get("/status/{job_id}", response_model=StatusResponse, summary="Get transcription job status")
async def get_status(job_id: str):
    """
    Poll the status of a transcription job.

    - **QUEUED** – waiting in queue
    - **PROCESSING** – transcription in progress
    - **COMPLETED** – done; `full_text` and `result` are populated
    - **FAILED** – an error occurred; `error` contains the message
    """
    queue = ASRJobQueue()
    status = queue.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/", summary="Health check")
async def root():
    return {"message": "Qwen3 ASR API is running", "model": os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")}
