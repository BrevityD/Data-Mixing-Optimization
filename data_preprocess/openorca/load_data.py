import json
import random
from pathlib import Path

from datasets import load_dataset
from langdetect import detect, LangDetectException
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "Llama-3.1-8B-Instruct"
OUTPUT_PATH = PROJECT_ROOT / "data" / "orca.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

data = load_dataset("Open-Orca/OpenOrca")
ds = data["train"]

structured_data = []
for record in tqdm(ds, desc="Processing dataset"):
    try:
        system_prompt = record["system_prompt"]
        question = record["question"]
        response = record["response"]
        if system_prompt and question and response:
            system_prompt_len = len(tokenizer.encode(system_prompt))
            question_len = len(tokenizer.encode(question))
            response_len = len(tokenizer.encode(response))

            question_lang = detect(question)
            response_lang = detect(response)

            if (
                question_lang == "en"
                and response_lang == "en"
                and system_prompt_len + question_len <= 4096
                and response_len <= 4096
            ):
                structured_data.append(
                    {
                        "system_prompt": system_prompt,
                        "id": record["id"],
                        "question": question,
                        "response": response,
                        "system_prompt_len": system_prompt_len,
                        "question_len": question_len,
                        "response_len": response_len,
                    }
                )
    except (LangDetectException, IndexError, KeyError):
        continue

random.shuffle(structured_data)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w") as json_file:
    json.dump(structured_data, json_file, indent=3)