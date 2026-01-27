import os
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel
from .models import GenerationType, GenerateRequest

class TTSService:
    def __init__(self):
        self.base_model = None
        self.custom_model = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
        self.base_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.custom_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        self.device = "cuda:0"
        
    def _unload_models(self):
        """Unload all models to free VRAM"""
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if self.custom_model is not None:
            del self.custom_model
            self.custom_model = None
        torch.cuda.empty_cache()

    def get_base_model(self):
        """Load base model if not loaded, unloading custom if necessary"""
        if self.base_model is None:
            if self.custom_model is not None:
                print("Unloading Custom Model to load Base Model...")
                del self.custom_model
                self.custom_model = None
                torch.cuda.empty_cache()
            
            print(f"Loading Base Model: {self.base_model_id}")
            self.base_model = Qwen3TTSModel.from_pretrained(
                self.base_model_id,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
        return self.base_model

    def get_custom_model(self):
        """Load custom model if not loaded, unloading base if necessary"""
        if self.custom_model is None:
            if self.base_model is not None:
                print("Unloading Base Model to load Custom Model...")
                del self.base_model
                self.base_model = None
                torch.cuda.empty_cache()
            
            print(f"Loading Custom Model: {self.custom_model_id}")
            self.custom_model = Qwen3TTSModel.from_pretrained(
                self.custom_model_id,
                device_map=self.device,
                dtype=torch.float16, # Custom voice example uses float16
            )
        return self.custom_model

    def generate(self, job_id: str, request: GenerateRequest) -> str:
        """
        Routes the request to the appropriate generation method.
        Returns the absolute path to the generated audio file.
        """
        output_filename = f"{job_id}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            if request.type == GenerationType.OPEN_VISION:
                self._generate_open_vision(request, output_path)
            elif request.type == GenerationType.CLONE_VOICE:
                self._generate_clone_voice(request, output_path)
            elif request.type == GenerationType.CUSTOM_VOICE:
                self._generate_custom_voice(request, output_path)
            else:
                raise ValueError(f"Unknown generation type: {request.type}")
                
            return os.path.abspath(output_path)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise e

    def _generate_open_vision(self, request: GenerateRequest, output_path: str):
        model = self.get_base_model()
        if hasattr(model, "generate_speech"):
            wavs, sr = model.generate_speech(
                text=request.text,
                language=request.language
            )
        elif hasattr(model, "generate_voice_design"):
            # generate_defaults is likely a dict, so we use generate_voice_design
             wavs, sr = model.generate_voice_design(
                 text=request.text,
                 language=request.language,
                 instruct=request.instruction or "Speak naturally."
             )
        else:
             raise AttributeError("No suitable generation method found (checked generate_speech and generate_voice_design)")
             
        sf.write(output_path, wavs[0], sr)

    def _generate_clone_voice(self, request: GenerateRequest, output_path: str):
        if not request.ref_audio_path or not request.ref_text:
            raise ValueError("ref_audio_path and ref_text are required for CLONE_VOICE")
            
        model = self.get_base_model() # Clone also uses the base model typically? 
        # Checking tts.py, generate_voice_clone uses Base model ("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        
        wavs, sr = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            ref_audio=request.ref_audio_path,
            ref_text=request.ref_text,
            instruct=request.instruction
        )
        sf.write(output_path, wavs[0], sr)

    def _generate_custom_voice(self, request: GenerateRequest, output_path: str):
        if not request.speaker:
            raise ValueError("speaker is required for CUSTOM_VOICE")
            
        model = self.get_custom_model()
        # Custom voice in tts.py uses generate_custom_voice
        wavs, sr = model.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruction,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        sf.write(output_path, wavs[0], sr)
