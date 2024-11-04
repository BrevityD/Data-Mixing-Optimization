# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
# model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")

# save_path = "/mbz/users/liyuan/LLaMA-Factory/checkpoints"

# Load the model and tokenizer directly into the specified directory
# tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", cache_dir=save_path)
# model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", cache_dir=save_path)

from modelscope import snapshot_download
# model_dir = snapshot_download('LLM-Research/OLMo-7B')
model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-70B-Instruct')