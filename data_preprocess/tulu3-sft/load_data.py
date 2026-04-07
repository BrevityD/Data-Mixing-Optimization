import json
import random
from pathlib import Path

from datasets import load_dataset
from langdetect import LangDetectException, detect
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = PROJ_ROOT / "datasets"
MODEL_PATH = PROJ_ROOT.parents[0] / "models" / "Qwen3-1.7B"
OUTPUT_PATH = REPO_ROOT / "data" / "tulu3-sft.json"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
ds = load_dataset(str(DATA_PATH / "general_domain" / "tulu-3-sft-mixture"))

# Shuffle indices beforehand to avoid storing all processed data in memory
indices = list(range(len(ds["train"])))
random.shuffle(indices)

BATCH_SIZE = 50
structured_data = []
first_item = True

with OUTPUT_PATH.open("w", encoding="utf-8") as json_file:
    json_file.write("[\n")
    
    for i in tqdm(indices, desc="Processing dataset"):
        try:
            if len(ds["train"][i]["messages"]) != 2:
                continue
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

        if len(structured_data) >= BATCH_SIZE:
            for item in structured_data:
                if not first_item:
                    json_file.write(",\n")
                else:
                    first_item = False
                
                item_str = json.dumps(item, indent=2)
                item_str = "  " + item_str.replace("\n", "\n  ")
                json_file.write(item_str)
            structured_data.clear()

    # Save remaining data
    if structured_data:
        for item in structured_data:
            if not first_item:
                json_file.write(",\n")
            else:
                first_item = False
            
            item_str = json.dumps(item, indent=2)
            item_str = "  " + item_str.replace("\n", "\n  ")
            json_file.write(item_str)
        structured_data.clear()

    json_file.write("\n]\n")

print(f"Structured data saved to {OUTPUT_PATH}")