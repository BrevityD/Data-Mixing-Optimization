import json
import random
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "checkpoints" / "Llama-3.1-8B-Instruct"
OUTPUT_PATH = REPO_ROOT / "data" / "data_mixing" / "tulu3" / "tulu3_target.json"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
with OUTPUT_PATH.open("r", encoding="utf-8") as file:
    ds = json.load(file)

structured_data = []
for i in tqdm(range(len(ds)), desc="Processing dataset"):
    instruction = ds[i]["instruction"]
    output = ds[i]["output"]
    source = ds[i]["source"]
    if instruction and output:
        input_len = len(tokenizer.encode(instruction))
        output_len = len(tokenizer.encode(output))
        structured_data.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
                "output_len": output_len,
                "input_len": input_len,
                "source": source,
            }
        )

random.shuffle(structured_data)

with OUTPUT_PATH.open("w", encoding="utf-8") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {OUTPUT_PATH}")