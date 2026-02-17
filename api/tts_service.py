import os
import torch
import soundfile as sf
import numpy as np

from qwen_tts import Qwen3TTSModel
from .models import GenerationType, GenerateRequest

# ... (omitted class start)

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
            
            print(f"Loading Base Model AAA: {self.base_model_id}")
            self.base_model = Qwen3TTSModel.from_pretrained(
                self.base_model_id,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
            print(f"Returning base model: {self.base_model_id}")
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
            # Determine text source
            text_to_process = request.text
            if request.uploaded_file_text:
                text_to_process = request.uploaded_file_text
            
            if not text_to_process:
                raise ValueError("No text provided for generation (neither 'text' nor 'uploaded_file_text')")
            
            # Temporary: Update request.text for downstream methods that rely on it
            # A cleaner approach would be passing text_to_process directly to methods
            request.text = text_to_process

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
        
        ref_audio_path, ref_text = self._get_voice_data(request.clone_voice_name)
        print(f"Ref audio: {ref_audio_path}")
        print(f"Ref text: {ref_text}")

        if ref_audio_path == '' or ref_text == '':
            if not request.ref_audio_path or not request.ref_text:
                raise ValueError("ref_audio_path and ref_text are required for CLONE_VOICE")
            else:
                ref_audio_path = request.ref_audio_path
                ref_text = request.ref_text
            
        # Get model once
        model = self.get_base_model()
        
        print(f"Generating voice clone for {request.clone_voice_name}")
        print(f"Text: {request.text}")
        print(f"Language: {request.language}")
        print(f"Ref audio: {ref_audio_path}")
        print(f"Ref text: {ref_text}")
        print(f"Instruction: {request.instruction}")
        
        chunks = [request.text[i:i+800] for i in range(0, len(request.text), 800)]  # Simple char-based split

        audio_chunks = []
        sample_rates = set()
        sr = 24000

        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}/{len(chunks)}...")
            audio, chunk_sr = self._synthesize_clone_voice_chunk(model, chunk, request.language, ref_audio_path, ref_text, request.instruction)
            audio_chunks.append(audio)
            sample_rates.add(chunk_sr)

        if len(sample_rates) > 1:
            raise ValueError(f"Sample rate inconsistency: {sample_rates}")

        if len(sample_rates) > 0:
             sr = next(iter(sample_rates))

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            sf.write(output_path, full_audio, sr)
        else:
             print("Warning: No audio generated")

    def _synthesize_clone_voice_chunk(self, model, text, language, ref_audio_path, ref_text, instruction):
        # Adjust max_new_tokens based on expected audio length (~12.5 tokens/sec)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            instruct=instruction,
            max_new_tokens=4096,
        )

        return wavs[0], sr

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

    def _get_voice_data(self, clone_voice_name: str):
        ref_audio_path = ""
        ref_text  = ""
        print(f"Getting voice data for {clone_voice_name}")

        file_path = f'clone_voices/{clone_voice_name}.txt'  # Replace with your actual path
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            ref_text = file.read()
            ref_audio_path = f"clone_voices/{clone_voice_name}.wav"
        
        return [ref_audio_path, ref_text]
