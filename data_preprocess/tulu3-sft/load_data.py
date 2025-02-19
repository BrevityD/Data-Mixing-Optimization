import json
import random
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"{base_path}/data/tulu3-sft.json"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds = load_dataset("allenai/tulu-3-sft-mixture")

# Process dataset
structured_data = []
for i in tqdm(range(len(ds['train'])), desc="Processing dataset"):
    try:
        instruction = ds["train"][i]["messages"][0]["content"]
        output = ds["train"][i]["messages"][1]["content"]
        if instruction and output:
            instruction_lang = detect(instruction)
            output_lang = detect(output)
            
            # Check language and token length
            if instruction_lang == 'en' and output_lang == 'en' and len(tokenizer.encode(output)) <= 4096:
                structured_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
    except (LangDetectException, IndexError):
        continue

# Shuffle the data
random.shuffle(structured_data)

# Save to JSON
with open(output_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {output_path}")
