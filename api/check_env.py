import torch
from qwen_tts import Qwen3TTSModel

def check():
    print("Checking Qwen3TTSModel attributes...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("Model loaded.")
        print("Dir(model):", dir(model))
        
        if hasattr(model, 'generate_speech'):
            print("HAS generate_speech")
        else:
            print("MISSING generate_speech")
            # Maybe it is just 'generate'?
            
        print("Type:", type(model))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check()
