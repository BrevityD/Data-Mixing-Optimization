import json
import random
from datasets import load_dataset
from tqdm import tqdm
# from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"/mbz/users/liyuan/LLaMA-Factory/data/data_mixing/tulu3/tulu3_target.json"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(output_path, "r") as file:
    ds = json.load(file)

# Process dataset
structured_data = []
for i in tqdm(range(len(ds)), desc="Processing dataset"):
    instruction = ds[i]["instruction"]
    output = ds[i]["output"]
    source = ds[i]["source"]
    if instruction and output:
        
        input_len = len(tokenizer.encode(instruction))
        output_len = len(tokenizer.encode(output))
        # Check language and token length
        structured_data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "output_len": output_len,
            "input_len": input_len,
            "source": source
        })

# Shuffle the data
random.shuffle(structured_data)

# Save to JSON
with open(output_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {output_path}")