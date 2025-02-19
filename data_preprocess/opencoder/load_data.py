import json
import random
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
# dataset_name = "opencoder-sft"
dataset_name = "opencoder-sft_len"
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"{base_path}/data/{dataset_name}.json"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)

# dataset_configs = [
#     ("OpenCoder-LLM/opc-sft-stage1", "filtered_infinity_instruct"),
#     ("OpenCoder-LLM/opc-sft-stage1", "largescale_diverse_instruct"),
#     ("OpenCoder-LLM/opc-sft-stage1", "realuser_instruct"),
# ]

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

