import torch
import os
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def debug():
    print("Loading Base Model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    
    print("Attributes:", [d for d in dir(model) if 'generate' in d])
    
    if hasattr(model, 'generate_defaults'):
        print("generate_defaults type:", type(model.generate_defaults))
        print("generate_defaults value:", model.generate_defaults)

    try:
        print("Attempting generate_speech...")
        wavs, sr = model.generate_speech(
            text="Test audio",
            language="English",
        )
        print("generate_speech Success")
    except Exception as e:
        print(f"generate_speech Failed: {e}")

    # Try to find another method if speech fails
    # Maybe it accepts text directly? no.
    
if __name__ == "__main__":
    debug()
