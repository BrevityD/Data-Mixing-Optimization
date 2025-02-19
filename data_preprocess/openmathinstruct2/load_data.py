import json
import random
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
dataset_name = "openmathinstruct2_1M_len"
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"{base_path}/data/{dataset_name}.json"

structured_data = []

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Process all datasets

ds = load_dataset("nvidia/OpenMathInstruct-2")
subset = "train_1M"
for i in tqdm(range(len(ds[subset])), desc=f"Processing"):
    try:
        instruction = ds[subset][i]["problem"]
        output = ds[subset][i]["generated_solution"]
        output_len=len(tokenizer.encode(output))
        instr_len=len(tokenizer.encode(instruction))
        if output_len <= 4096 and instr_len <= 4096:
                structured_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "len": instr_len + output_len
                })
    except (LangDetectException, IndexError):
        continue

random.shuffle(structured_data)

# Save to JSON
with open(output_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {output_path}")


dataset_info_path = "/mbz/users/liyuan/LLaMA-Factory/data/dataset_info.json"
with open(dataset_info_path, "r") as f:
    try:
        dataset_info = json.load(f)
    except json.JSONDecodeError:
        dataset_info = {}

dataset_info[dataset_name] = {
    "file_name": output_path
}

with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

