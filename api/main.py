from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from .models import GenerateRequest, JobResponse, StatusResponse, JobStatus, GenerationType
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
async def generate_audio(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    type: GenerationType = Form(...),
    ref_audio_path: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    clone_voice_name: Optional[str] = Form(None),
    language: str = Form("English"),
):
    uploaded_file_text = None
    if file:
        content = await file.read()
        uploaded_file_text = content.decode("utf-8")
    
    if not text and not uploaded_file_text:
        raise HTTPException(status_code=400, detail="Either 'text' or 'file' must be provided.")

    # Construct the Pydantic model manually from form fields
    request = GenerateRequest(
        text=text,
        type=type,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        instruction=instruction,
        speaker=speaker,
        clone_voice_name=clone_voice_name,
        uploaded_file_text=uploaded_file_text,
        language=language
    )
    
    queue = JobQueue()
    job_id = queue.add_job(request)
    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)

@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    queue = JobQueue()
    status = queue.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/download/{job_id}")
async def download_audio(job_id: str):
    queue = JobQueue()
    job_status = queue.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job_status.status != JobStatus.COMPLETED:
         raise HTTPException(status_code=400, detail="Job not completed yet")
         
    if not job_status.output_path or not os.path.exists(job_status.output_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    return FileResponse(job_status.output_path, media_type="audio/wav", filename=f"{job_id}.wav")

@app.get("/")
async def root():
    return {"message": "Qwen3 TTS API is running"}
