import json
import random
from pathlib import Path

from datasets import load_dataset
from langdetect import LangDetectException, detect
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "checkpoints" / "Llama-3.1-8B-Instruct"
OUTPUT_PATH = REPO_ROOT / "data" / "tulu3-sft.json"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
ds = load_dataset("allenai/tulu-3-sft-mixture")

structured_data = []
for i in tqdm(range(len(ds["train"])), desc="Processing dataset"):
    try:
        instruction = ds["train"][i]["messages"][0]["content"]
        output = ds["train"][i]["messages"][1]["content"]
        if instruction and output:
            instruction_lang = detect(instruction)
            output_lang = detect(output)

            if (
                instruction_lang == "en"
                and output_lang == "en"
                and len(tokenizer.encode(output)) <= 4096
            ):
                structured_data.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                    }
                )
    except (LangDetectException, IndexError):
        continue

random.shuffle(structured_data)

with OUTPUT_PATH.open("w", encoding="utf-8") as json_file:
    json.dump(structured_data, json_file, indent=2)

print(f"Structured data saved to {OUTPUT_PATH}")
