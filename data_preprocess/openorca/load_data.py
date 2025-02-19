import json
import random
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"/mbz/users/liyuan/LLaMA-Factory/data/orca.json"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)
from datasets import load_dataset

data = load_dataset("Open-Orca/OpenOrca")

ds = data["train"]
    
    
# Process dataset
structured_data = []
for i in tqdm(range(len(ds)), desc="Processing dataset"):
    try:
        id = ds[i]["id"]
        system_prompt = ds[i]["system_prompt"]
        question = ds[i]["question"]
        response = ds[i]["response"]
        if system_prompt and question and response:
            system_prompt_len = len(tokenizer.encode(system_prompt))
            question_len = len(tokenizer.encode(question))
            response_len = len(tokenizer.encode(response))
            
            question_lang = detect(question)
            response_lang = detect(response)
            
            if question_lang == 'en' and response_lang == 'en' and system_prompt_len + question_len <= 4096 and response_len <= 4096:
            # Check language and token length
                structured_data.append({
                    "system_prompt": system_prompt,
                    "id": id,
                    "question": question,
                    "response": response,
                    "system_prompt_len": system_prompt_len,
                    "question_len": question_len,
                    "response_len": response_len,
                })
    except (LangDetectException, IndexError):
        continue
    
# Shuffle the data
random.shuffle(structured_data)

# Save to JSON
with open(output_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=3)

print(f"Structured data saved to {output_path}")