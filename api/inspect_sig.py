from qwen_tts import Qwen3TTSModel
import inspect

def check():
    print("Checking signatures...")
    try:
        # We don't need to load the model fully if we can inspect the class, 
        # but from_pretrained returns an instance, lets inspect the class directly if possible or the instance.
        # Actually loading is safer to get the bound method if dynamically added. 
        # But we can try to just import the class.
        
        print("generate_defaults signature:")
        print(inspect.signature(Qwen3TTSModel.generate_defaults))
        print("Doc:", Qwen3TTSModel.generate_defaults.__doc__)
        
        print("generate_voice_clone signature:")
        print(inspect.signature(Qwen3TTSModel.generate_voice_clone))
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check()
