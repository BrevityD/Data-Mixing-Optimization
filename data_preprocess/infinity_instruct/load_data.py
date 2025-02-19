import json
import random
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths
dataset_name = "cache_Infinity-Instruct_0625_len"
base_path = '/mbz/users/liyuan/LLaMA-Factory'
model_path = f"{base_path}/checkpoints/Llama-3.1-8B-Instruct"
output_path = f"{base_path}/data/{dataset_name}.json"

def filter(input_str, output_str, keywords1, keywords2):
    input_words = set(input_str.split())
    output_words = set(output_str.split())

    keywords = keywords1 | keywords2

    if input_words & keywords or output_words & keywords:
        return False
    
    if len(input_str) < short_threshold or len(output_str) < short_threshold:
        return random.random() < keep_probability

    return True

structured_data = []

short_threshold = 8
keep_probability = 0.02

tokenizer = AutoTokenizer.from_pretrained(model_path)

ds = load_dataset("BAAI/Infinity-Instruct", "0625")

math_keywords = {"calculate", "solve", "equation", "math problem", "algebra", "geometry", "trigonometry", "integral", 
                 "derivative", "sum", "difference", "product", "divide", "multiplier", "average", "percentage", "ratio", "proportion"}

programming_keywords = {"code", "program", "script", "algorithm", "function", "loop", "variable", "array", "object", "class", "method", 
                        "procedure", "execute", "run", "compile", "debug", "syntax", "library", "API", "framework", "software", "hardware", 
                        "command", "terminal", "bash", "shell", "python", "java", "javascript", "C++", "SQL", "query", "json", "xml", "data structure"}

structured_data = []
for i in tqdm(range(len(ds['train'])), desc="Processing dataset"):
    try:
        item = ds["train"][i]["conversations"]
        instruction = item[0]["value"]
        output = item[1]["value"]

        if instruction and output:
            instruction_lang = detect(instruction)
            output_lang = detect(output)
            filtered = filter(instruction, output, math_keywords, programming_keywords)
            output_len = len(tokenizer.encode(output))
            instr_len = len(tokenizer.encode(instruction))
            if instruction_lang == 'en' and output_lang == 'en' and output_len <= 4096 and instr_len <= 4096 and filtered:
                structured_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "len": output_len + instr_len
                })
    except (LangDetectException, IndexError):
        continue

# Shuffle the data
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