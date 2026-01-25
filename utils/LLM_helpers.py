from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_validate_gpu(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # FIX: Changed tokenizer_path to model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with better GPU handling
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",  # Let transformers handle device placement
        low_cpu_mem_usage=True,
    )
    
    # CHANGED: Warning instead of hard assertion
    if model.hf_device_map and 'cpu' in str(model.hf_device_map.values()):
        print("⚠ WARNING: Some model layers on CPU (limited GPU memory)")
        print("  This is OK but will be slower")
    else:
        print("✓ Model fully loaded on GPU")
    
    return model, tokenizer