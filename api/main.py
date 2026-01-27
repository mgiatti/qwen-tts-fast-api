from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import GenerateRequest, JobResponse, StatusResponse
from .worker import JobQueue

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    queue = JobQueue()
    queue.start_worker()
    yield
    # Shutdown
    queue.stop_worker()

app = FastAPI(title="Qwen3 TTS API", lifespan=lifespan)

# CORS (Optional, but good for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=JobResponse)
async def generate_audio(request: GenerateRequest):
    queue = JobQueue()
    job_id = queue.add_job(request)
    return JobResponse(job_id=job_id, status="queued")

@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    queue = JobQueue()
    status = queue.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/")
async def root():
    return {"message": "Qwen3 TTS API is running"}
