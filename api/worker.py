import threading
import queue
import uuid
import time
import traceback
from typing import Dict, Optional
from .models import GenerateRequest, JobStatus, StatusResponse
from .tts_service import TTSService

class JobQueue:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobQueue, cls).__new__(cls)
            cls._instance.queue = queue.Queue()
            cls._instance.jobs: Dict[str, dict] = {} # stores job data
            cls._instance.worker_thread = None
            cls._instance.stop_event = threading.Event()
            cls._instance.tts_service = None # Lazy init in worker
        return cls._instance

    def add_job(self, request: GenerateRequest) -> str:
        job_id = str(uuid.uuid4())
        job_info = {
            "id": job_id,
            "request": request,
            "status": JobStatus.QUEUED,
            "output_path": None,
            "error": None,
            "created_at": time.time()
        }
        self.jobs[job_id] = job_info
        self.queue.put(job_id)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[StatusResponse]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return StatusResponse(
            job_id=job["id"],
            status=job["status"],
            output_path=job["output_path"],
            error=job["error"]
        )

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _process_queue(self):
        print("Worker thread started.")
        # Initialize service here to ensure it runs in the worker thread context if needed
        # (Though primarily for CUDA, typical separate-thread initialization is fine as long as inference is in same thread)
        self.tts_service = TTSService()
        
        while not self.stop_event.is_set():
            try:
                # Wait for job with timeout to check stop_event
                job_id = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            
            job = self.jobs.get(job_id)
            if not job:
                # Should not happen
                self.queue.task_done()
                continue
                
            print(f"Processing job {job_id}...")
            job["status"] = JobStatus.PROCESSING
            
            try:
                # Execute generation
                output_path = self.tts_service.generate(job_id, job["request"])
                job["output_path"] = output_path
                job["status"] = JobStatus.COMPLETED
                print(f"Job {job_id} completed. Output: {output_path}")
            except Exception as e:
                error_msg = str(e)
                print(f"Job {job_id} failed: {error_msg}")
                traceback.print_exc()
                job["error"] = error_msg
                job["status"] = JobStatus.FAILED
            finally:
                self.queue.task_done()
