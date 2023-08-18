import huggingface_hub
from llama_cpp import Llama

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-GGML"
MODEL_BASENAME = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = huggingface_hub.hf_hub_download(repo_id=MODEL_NAME_OR_PATH, filename=MODEL_BASENAME)

# GPU
LCPP_LLM = None  
LCPP_LLM = Llama(
    model_path=model_path,
    n_threads=2, 
    n_batch=512,
    n_gpu_layers=32
)

print(LCPP_LLM.params.n_gpu_layers)

PROMPT = "Write a linear regression in python"  
PROMPT_TEMPLATE = f"""
SYSTEM: You are helpful, respectful, and honest.

USER: {PROMPT} 

ASSISTANT:  
"""

response = LCPP_LLM(
    prompt=PROMPT_TEMPLATE,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=150,
    echo=True
)

print(response["choices"][0]["text"])