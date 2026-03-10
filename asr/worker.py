import threading
import queue
import uuid
import time
import traceback
from typing import Dict, Optional, List

from .models import TranscribeRequest, JobStatus, StatusResponse, TranscriptResult
from .asr_service import ASRService


class ASRJobQueue:
    """
    Singleton job queue for ASR – mirrors the TTS JobQueue pattern exactly.

    Why a queue?
    ─────────────
    GPU inference is not thread-safe. A single worker thread serialises all
    requests so the GPU is never accessed concurrently.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.queue = queue.Queue()
            cls._instance.jobs: Dict[str, dict] = {}
            cls._instance.worker_thread = None
            cls._instance.stop_event = threading.Event()
            cls._instance.asr_service = None   # Lazy init in worker
        return cls._instance

    # ── Public API ───────────────────────────────────────────────────────────

    def add_job(self, request: TranscribeRequest, audio_path: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "request": request,
            "audio_path": audio_path,      # set when a file was uploaded
            "status": JobStatus.QUEUED,
            "result": None,
            "full_text": None,
            "error": None,
            "created_at": time.time(),
        }
        self.queue.put(job_id)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[StatusResponse]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return StatusResponse(
            job_id=job["id"],
            status=job["status"],
            result=job["result"],
            full_text=job["full_text"],
            error=job["error"],
        )

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(
                target=self._process_queue, daemon=True
            )
            self.worker_thread.start()

    def stop_worker(self):
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    # ── Worker loop ──────────────────────────────────────────────────────────

    def _process_queue(self):
        print("[ASR Worker] started.")
        # Initialise service here so CUDA context is created inside the thread
        self.asr_service = ASRService()

        while not self.stop_event.is_set():
            try:
                job_id = self.queue.get(timeout=1)
            except queue.Empty:
                continue

            job = self.jobs.get(job_id)
            if not job:
                self.queue.task_done()
                continue

            print(f"[ASR Worker] Processing job {job_id} …")
            job["status"] = JobStatus.PROCESSING

            try:
                request: TranscribeRequest = job["request"]
                results: List[TranscriptResult] = self.asr_service.transcribe(
                    audio_path=job["audio_path"],
                    audio_url=request.audio_url,
                    language=request.language,
                    return_timestamps=request.return_timestamps,
                    chunk_duration_seconds=request.chunk_duration_seconds,
                )

                job["result"] = results
                job["full_text"] = " ".join(r.text for r in results).strip()
                job["status"] = JobStatus.COMPLETED
                print(f"[ASR Worker] Job {job_id} completed.")
            except Exception as e:
                print(f"[ASR Worker] Job {job_id} failed: {e}")
                traceback.print_exc()
                job["error"] = str(e)
                job["status"] = JobStatus.FAILED
            finally:
                # Clean up temp upload file if present
                audio_path = job.get("audio_path")
                if audio_path and audio_path.startswith("/tmp"):
                    try:
                        import os
                        os.remove(audio_path)
                    except OSError:
                        pass
                self.queue.task_done()
