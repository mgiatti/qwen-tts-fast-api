from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscribeRequest(BaseModel):
    """
    Request model for transcription jobs submitted via JSON body
    (audio_url only). For file uploads use the multipart endpoint.
    """
    audio_url: Optional[str] = None
    language: Optional[str] = None          # None = auto-detect
    return_timestamps: bool = False
    chunk_duration_seconds: int = 300       # 5 min chunks (0 = no chunking)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "Job submitted successfully"


class TranscriptSegment(BaseModel):
    """A single word/phrase timestamp entry from the forced aligner."""
    text: str
    start_time: float
    end_time: float


class TranscriptResult(BaseModel):
    """Full transcription result for one audio (or one chunk)."""
    language: Optional[str] = None
    text: str
    time_stamps: Optional[List[TranscriptSegment]] = None


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[List[TranscriptResult]] = None   # list of chunk results
    full_text: Optional[str] = None                   # concatenated text across chunks
    error: Optional[str] = None
