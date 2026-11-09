import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False):
    """Load a model and its tokenizer onto the GPU.

    load_in_4bit=False: bfloat16 weights (the 7B model needs ~15 GB of GPU memory).
    load_in_4bit=True:  4-bit NF4 weights via bitsandbytes (~5.5-6 GB, fits an 8 GB GPU);
                        the output layer stays in higher precision, so token probabilities are still available.
    """
    check_system()

    model_id = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            device_map="cuda:0"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0"
        )

    return tokenizer, model

def check_system():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")

    print("system check:")
    print("Python path:", sys.executable)
    print("Torch version:", torch.__version__)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Capability:", torch.cuda.get_device_capability())
    print("this is include the following functions torch , torch.nn.functional as F, transformers.AutoTokenizer, transformers.AutoModelForCausalLM")