from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model_for_data_generation(model_path, tokenizer_path=None):
    """Load model with 4-bit quantization for data generation."""
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Loading model with 4-bit quantization for data generation")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        output_hidden_states=True
    )
    
    return model, tokenizer
