"""用于在train_calibrator之后，计算每个领域的checkpoint的PPL
1+1+1+1+1的例子需要手动测，会自动跳过，懒得改了

使用示例：
```bash
CUDA_VISIBLE_DEVICES=0 python ppl_runner.py calculate_ppl \
    --model_name_or_path ../../saves/param_calibrator/Qwen3-1.7B-Base_2000000/insfo1_math1_code1_algebra1_math-grade1/checkpoint-962 \
    --dataset 2000000_insfo_val \
    --template qwen \
    --save_name ppl_insfo.json
```

```bash
python ppl_runner.py run_pipeline \
    --saves_root ../../saves/param_calibrator \
    --gpus "0,1,2,3" \
    --dataset_dir ../../data \
    --template qwen
```

```bash
python ppl_runner.py collect_results \
    --saves_root ../../saves/param_calibrator/Qwen3-1.7B-Base_2000000 \
    --output_json aggregated_ppl.json
```
"""

# Copyright 2024 the LlamaFactory team.
# (Licensed under the Apache License, Version 2.0)

import os
import re
import math
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """
    train_on_prompt: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_features = []
        for feature in features:
            chosen_features.append(
                {
                    "input_ids": feature["chosen_input_ids"],
                    "attention_mask": feature["chosen_attention_mask"],
                    "labels": feature["chosen_input_ids"] if self.train_on_prompt else feature["chosen_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
            )
        return super().__call__(chosen_features)


def calculate_ppl(
    model_name_or_path: str,
    save_name: str = "ppl.json",
    batch_size: int = 4,
    stage: Literal["pt", "sft", "rm"] = "sft",
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 4096,
    max_samples: Optional[int] = None,
    train_on_prompt: bool = False,
):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=stage,
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            train_on_prompt=train_on_prompt,
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=True,
        )
    )
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    trainset = get_dataset(template, model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    
    if stage == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template, tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX
        )
    elif stage == "rm":
        data_collator = PairwiseDataCollatorWithPadding(
            template=template, tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, train_on_prompt=train_on_prompt
        )
    else:
        raise NotImplementedError(f"Stage does not supported: {stage}.")

    dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_log_prob = 0.0
    total_tokens = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset}"):
            batch = batch.to(model.device)
            outputs = model(**batch)
            shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
            shift_labels: "torch.Tensor" = batch["labels"][..., 1:]
            loss_mask = shift_labels != IGNORE_INDEX
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            
            batch_log_prob = (token_logps * loss_mask).sum().item()
            batch_tokens = loss_mask.sum().item()
            total_log_prob += batch_log_prob
            total_tokens += batch_tokens
            
        avg_log_prob = total_log_prob / total_tokens
        token_level_ppl = math.exp(avg_log_prob)
        printout = [{"token-level ppl": token_level_ppl}]
        
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(printout, f, indent=2)
        
    logger.info(f"Perplexity ({token_level_ppl:.4f}) saved at {save_name}.")


def run_pipeline(
    saves_root: str = "saves", 
    gpus: str = "0,1,2,3",
    dataset_dir: str = "data",
    template: str = "default",
    batch_size: int = 4
):
    r"""
    核心调度器：解析文件夹 -> 提取数据集 -> 提取第2个Epoch -> 多卡队列执行
    """
    gpu_list = [g.strip() for g in gpus.split()]
    gpu_queue = Queue()
    for g in gpu_list:
        gpu_queue.put(g)

    tasks = []
    base_dirs = set()
    
    # 1. 扫描所有包含 checkpoint 的顶层实验目录
    for root, dirs, files in os.walk(saves_root):
        for d in dirs:
            if d.startswith("checkpoint-"):
                base_dirs.add(root)
                break 

    # 2. 组装任务信息
    for bdir in base_dirs:
        folder_name = os.path.basename(bdir)
        
        # 解析如：insfo0.5_math1_code1_algebra1_math-grade1
        parts = folder_name.split('_')
        target_domain = None
        for part in parts:
            # 用正则匹配：字母/中划线 作为名字，数字/小数点 作为比例
            match = re.match(r'^([a-zA-Z\-]+)([\d\.]+)$', part)
            if match:
                domain, ratio = match.groups()
                if abs(float(ratio)-1.0) > 1e-4:  # 找到非 1 的那个领域
                    target_domain = domain
                    break
                    
        if not target_domain:
            logger.info(f"[跳过] {bdir}：未检测到比例不为 1 的主验证领域。")
            continue
            
        # 从 bdir 的父目录解析 base_token
        # 例如 param_calibrator/Qwen3-1.7B-Base_2000000 -> 提取 2000000
        parent_dir_name = os.path.basename(os.path.dirname(bdir))
        base_token = parent_dir_name.split('_')[-1]
            
        dataset_name = f"{base_token}_{target_domain}_val"
        
        # 寻找第2个 Epoch (对 checkpoint-数字 后缀进行排序，取索引1)
        ckpts = [d for d in os.listdir(bdir) if d.startswith("checkpoint-")]
        ckpts.sort(key=lambda x: int(x.split('-')[-1]))
        
        if len(ckpts) < 2:
            logger.warning(f"[跳过] {bdir}：仅找到 {len(ckpts)} 个 checkpoint，无法获取第2个 Epoch。")
            continue
            
        target_ckpt = os.path.join(bdir, ckpts[1])
        save_json = os.path.join(bdir, f"ppl_{dataset_name}.json") # 将结果存在实验目录下
        
        tasks.append({
            "ckpt": target_ckpt,
            "dataset": dataset_name,
            "save_json": save_json
        })

    logger.info(f"总计找到 {len(tasks)} 个有效测算任务，开始在 GPU {gpu_list} 上分配执行...")

    # 3. 消费端逻辑 (子进程运行计算避免 OOM)
    def worker(task):
        gpu_id = gpu_queue.get()
        try:
            logger.info(f">>> [GPU {gpu_id}] 分配任务: {task['ckpt']} | 数据集: {task['dataset']}")
            cmd = [
                "python", os.path.abspath(__file__), "calculate_ppl",
                "--model_name_or_path", task["ckpt"],
                "--dataset", task["dataset"],
                "--dataset_dir", dataset_dir,
                "--template", template,
                "--batch_size", str(batch_size),
                "--save_name", task["save_json"]
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            # 使用 subprocess 运行，阻断任何可能的显存泄漏
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"<<< [GPU {gpu_id}] 任务完成: {task['ckpt']}")
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] 任务执行失败 {task['ckpt']}: {str(e)}")
        finally:
            gpu_queue.put(gpu_id) # 释放卡给下一个排队的任务

    # 4. 启动并发池
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
        executor.map(worker, tasks)
        
    logger.info("所有测算任务全部完成！")


def collect_results(saves_root: str = "saves", output_json: str = "aggregated_ppl.json"):
    r"""
    统计之前生成的 ppl_*.json 文件，按 domain 和 ratio 进行聚合汇总。
    """
    from collections import defaultdict
    
    results = defaultdict(dict)
    base_dirs = set()
    
    for root, dirs, files in os.walk(saves_root):
        for d in dirs:
            if d.startswith("checkpoint-"):
                base_dirs.add(root)
                break
                
    for bdir in base_dirs:
        folder_name = os.path.basename(bdir)
        
        # 解析 folder_name
        parts = folder_name.split('_')
        target_domain = None
        target_ratio = None
        for part in parts:
            match = re.match(r'^([a-zA-Z\-]+)([\d\.]+)$', part)
            if match:
                domain, ratio = match.groups()
                if abs(float(ratio)-1.0) > 1e-4:
                    target_domain = domain
                    target_ratio = ratio
                    break
        
        if not target_domain:
            continue
            
        # 寻找 ppl_*.json 文件
        ppl_files = [f for f in os.listdir(bdir) if f.startswith("ppl_") and f.endswith(".json")]
        if not ppl_files:
            continue
            
        # 理论上只有一个符合的 ppl 文件
        ppl_file_path = os.path.join(bdir, ppl_files[0])
        try:
            with open(ppl_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0 and "token-level ppl" in data[0]:
                    ppl_val = data[0]["token-level ppl"]
                    results[target_domain][target_ratio] = ppl_val
        except Exception as e:
            logger.error(f"读取或解析 {ppl_file_path} 失败: {e}")
            
    # 转换为普通字典以便输出
    final_results = {k: v for k, v in results.items()}
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
        
    logger.info(f"统计完成！结果已保存至 {output_json}:")
    print(json.dumps(final_results, indent=2))
    return final_results


if __name__ == "__main__":
    fire.Fire()
