from pydantic import BaseModel
from typing import Optional
from enum import Enum

class GenerationType(str, Enum):
    OPEN_VISION = "open_vision"
    CLONE_VOICE = "clone_voice"
    CUSTOM_VOICE = "custom_voice"

class GenerateRequest(BaseModel):
    text: str
    type: GenerationType
    
    # Optional fields depending on type
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    instruction: Optional[str] = None
    speaker: Optional[str] = None
    
    # Defaults
    language: str = "English"

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "Job submitted successfully"

class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    output_path: Optional[str] = None
    error: Optional[str] = None
