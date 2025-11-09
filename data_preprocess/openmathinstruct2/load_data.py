import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

dataset_name = "openmathinstruct2_1M_len"
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "checkpoints" / "Llama-3.1-8B-Instruct"
output_path = project_root / "data" / f"{dataset_name}.json"

structured_data = []

tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    except IndexError:
        continue

random.shuffle(structured_data)

with open(output_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {output_path}")


dataset_info_path = project_root / "data" / "dataset_info.json"
with open(dataset_info_path, "r") as f:
    try:
        dataset_info = json.load(f)
    except json.JSONDecodeError:
        dataset_info = {}

dataset_info[dataset_name] = {
    "file_name": str(output_path)
}

with open(dataset_info_path, "w") as f:
    json.dump(dataset_info, f, indent=2)

