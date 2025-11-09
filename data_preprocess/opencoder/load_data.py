import json
import random
from pathlib import Path

from datasets import load_dataset
from langdetect import LangDetectException, detect
from tqdm import tqdm
from transformers import AutoTokenizer

dataset_name = "opencoder-sft_len"
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "checkpoints" / "Llama-3.1-8B-Instruct"
output_path = project_root / "data" / f"{dataset_name}.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset_configs = [
    ("OpenCoder-LLM/opc-sft-stage1", "filtered_infinity_instruct"),
]

structured_data = []

# Process all datasets
for config in dataset_configs:
    ds = load_dataset(*config)
    for i in tqdm(range(len(ds['train'])), desc=f"Processing {config[1]}"):
        try:
            instruction = ds["train"][i]["instruction"]
            output = ds["train"][i]["output"]
            if instruction and output:
                instruction_lang = detect(instruction)
                output_lang = detect(output)
                output_len = len(tokenizer.encode(output))
                instr_len = len(tokenizer.encode(instruction))
                # Check language and token length
                if (
                    instruction_lang == 'en' and 
                    output_lang == 'en' and 
                    output_len <= 4096 and 
                    instr_len <= 4096
                ):
                    structured_data.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                        "len": output_len + instr_len
                    })
        except (LangDetectException, IndexError):
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

