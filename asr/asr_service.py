import os
import math
import tempfile
import torch
import soundfile as sf
import numpy as np

from typing import List, Optional, Tuple
from qwen_asr import Qwen3ASRModel

from .models import TranscribeRequest, TranscriptResult, TranscriptSegment


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
DEFAULT_DEVICE = os.getenv("ASR_DEVICE", "cuda:0")
# Set to "" or "none" to disable forced aligner (saves ~2 GB VRAM)
FORCED_ALIGNER_ID = os.getenv("ASR_FORCED_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")

# Safety margin: max seconds of audio per chunk sent to the model.
# The ForcedAligner only supports up to 5 min; the ASR model itself handles
# longer audio but chunking avoids GPU OOM for 40-min files.
DEFAULT_CHUNK_SECONDS = 300  # 5 minutes


class ASRService:
    """
    Singleton-backed ASR service that wraps Qwen3-ASR.

    Key design decisions
    ─────────────────────
    • Lazy model loading (first request triggers load)
    • Chunked inference for long audio to avoid OOM
    • Chunk size is configurable per-request but capped at DEFAULT_CHUNK_SECONDS
    • Uses flash_attention_2 when available (set via env)
    """

    def __init__(self):
        self._model: Optional[Qwen3ASRModel] = None
        self._model_id = DEFAULT_MODEL_ID
        self._device = DEFAULT_DEVICE
        self._use_flash_attn = os.getenv("ASR_USE_FLASH_ATTN", "true").lower() == "true"

        # Resolved aligner id (empty string disables it)
        aligner_raw = FORCED_ALIGNER_ID.strip()
        self._forced_aligner_id = aligner_raw if aligner_raw and aligner_raw.lower() != "none" else None

    # ── Model management ────────────────────────────────────────────────────

    def _get_model(self) -> Qwen3ASRModel:
        """Load model on first call, then reuse."""
        if self._model is None:
            print(f"[ASR] Loading model: {self._model_id}")

            attn_impl = "flash_attention_2" if self._use_flash_attn else None

            aligner_kwargs = None
            if self._forced_aligner_id:
                aligner_kwargs = dict(
                    dtype=torch.bfloat16,
                    device_map=self._device,
                )
                if attn_impl:
                    aligner_kwargs["attn_implementation"] = attn_impl

            self._model = Qwen3ASRModel.from_pretrained(
                self._model_id,
                dtype=torch.bfloat16,
                device_map=self._device,
                attn_implementation=attn_impl,
                max_inference_batch_size=8,
                max_new_tokens=4096,   # ~5 min of audio at ~12.5 tok/sec ≈ 3750 tokens
                **({
                    "forced_aligner": self._forced_aligner_id,
                    "forced_aligner_kwargs": aligner_kwargs,
                } if self._forced_aligner_id else {}),
            )
            print(f"[ASR] Model loaded successfully.")
        return self._model

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            print("[ASR] Model unloaded.")

    # ── Audio utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _load_audio_from_path(path: str) -> Tuple[np.ndarray, int]:
        """Load audio file into a numpy array."""
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)   # mix down to mono
        return data, sr

    @staticmethod
    def _split_into_chunks(
        data: np.ndarray, sr: int, chunk_seconds: int
    ) -> List[np.ndarray]:
        """
        Split audio array into chunks of `chunk_seconds` each.
        Returns a list of numpy arrays.
        """
        chunk_samples = chunk_seconds * sr
        num_chunks = math.ceil(len(data) / chunk_samples)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(data))
            chunks.append(data[start:end])
        return chunks

    # ── Core transcription ───────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: Optional[str],
        audio_url: Optional[str],
        language: Optional[str],
        return_timestamps: bool,
        chunk_duration_seconds: int,
    ) -> List[TranscriptResult]:
        """
        Transcribe audio from a local file path or a URL.

        For local files:  chunked inference (handles 40-min recordings).
        For URLs:         single-call inference (let the model handle it;
                          chunking URLs is more complex and most remote
                          audio is shorter).

        Returns list of TranscriptResult (one per chunk or one for URL).
        """
        model = self._get_model()

        if audio_url:
            return self._transcribe_url(model, audio_url, language, return_timestamps)

        if audio_path:
            return self._transcribe_file(
                model, audio_path, language, return_timestamps, chunk_duration_seconds
            )

        raise ValueError("Either audio_path or audio_url must be provided.")

    def _transcribe_url(
        self,
        model: Qwen3ASRModel,
        url: str,
        language: Optional[str],
        return_timestamps: bool,
    ) -> List[TranscriptResult]:
        """Single-call transcription for a remote URL."""
        print(f"[ASR] Transcribing URL: {url}")
        raw_results = model.transcribe(
            audio=url,
            language=language,
            return_time_stamps=return_timestamps and bool(self._forced_aligner_id),
        )
        return [self._convert_result(r, return_timestamps) for r in raw_results]

    def _transcribe_file(
        self,
        model: Qwen3ASRModel,
        path: str,
        language: Optional[str],
        return_timestamps: bool,
        chunk_duration_seconds: int,
    ) -> List[TranscriptResult]:
        """
        Chunked transcription for a local audio file.

        Strategy
        ─────────
        1. Load the full file into RAM as a numpy array.
        2. Split into chunks (default 5 min each).
        3. Write each chunk to a temp WAV file and transcribe.
        4. Collect all results; timestamps within each chunk are relative
           to chunk start, so we offset them when building the response.
        """
        data, sr = self._load_audio_from_path(path)
        total_duration = len(data) / sr
        print(
            f"[ASR] Audio loaded: {total_duration:.1f}s at {sr}Hz. "
            f"Chunk size: {chunk_duration_seconds}s"
        )

        if chunk_duration_seconds <= 0 or total_duration <= chunk_duration_seconds:
            # Short audio or chunking disabled – process as single call
            raw = model.transcribe(
                audio=(data, sr),
                language=language,
                return_time_stamps=return_timestamps and bool(self._forced_aligner_id),
            )
            return [self._convert_result(r, return_timestamps) for r in raw]

        chunks = self._split_into_chunks(data, sr, chunk_duration_seconds)
        results: List[TranscriptResult] = []
        time_offset = 0.0

        for i, chunk in enumerate(chunks):
            chunk_start = i * chunk_duration_seconds
            chunk_end = chunk_start + len(chunk) / sr
            print(f"[ASR] Chunk {i+1}/{len(chunks)}: {chunk_start:.1f}s – {chunk_end:.1f}s")

            # Write chunk to temp file (qwen-asr accepts (ndarray, sr) tuples,
            # which avoids disk I/O – but using a tuple is the cleanest API)
            raw = model.transcribe(
                audio=(chunk, sr),
                language=language,
                return_time_stamps=return_timestamps and bool(self._forced_aligner_id),
            )

            for r in raw:
                result = self._convert_result(r, return_timestamps, time_offset=time_offset)
                results.append(result)

            time_offset += len(chunk) / sr

        return results

    # ── Result conversion ────────────────────────────────────────────────────

    @staticmethod
    def _convert_result(raw, return_timestamps: bool, time_offset: float = 0.0) -> TranscriptResult:
        """
        Convert qwen-asr result object into our Pydantic model.
        Applies `time_offset` to all timestamps (for chunked audio).
        """
        segments = None
        if return_timestamps and hasattr(raw, "time_stamps") and raw.time_stamps:
            segments = [
                TranscriptSegment(
                    text=ts.text,
                    start_time=round(ts.start_time + time_offset, 3),
                    end_time=round(ts.end_time + time_offset, 3),
                )
                for ts in raw.time_stamps
            ]

        return TranscriptResult(
            language=getattr(raw, "language", None),
            text=getattr(raw, "text", ""),
            time_stamps=segments,
        )
