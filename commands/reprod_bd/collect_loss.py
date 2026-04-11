#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saves_root", type=str, default="./saves/exp2_grid")
    parser.add_argument("--output_json", type=str, default="./saves/exp2_grid/aggregated_loss_total.json")
    args = parser.parse_args()

    saves_root = Path(args.saves_root)
    results = {}

    if not saves_root.exists():
        print(f"Error: {saves_root} does not exist.")
        return

    # Walk through the directories looking for trainer_state.json
    for root, dirs, files in os.walk(saves_root):
        if "trainer_state.json" in files and "checkpoint" not in str(root):
            trainer_state_path = Path(root) / "trainer_state.json"
            
            # The folder name is the ratio mix, e.g. insfo0.375_math0.125_code0.125_algebra0.25_math-grade0.125
            # We want to extract the min eval_loss
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                log_history = data.get("log_history", [])
                eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
                
                if eval_losses:
                    min_loss = min(eval_losses)
                    
                    # Assuming the structure saves_root / model_name / exp_name / trainer_state.json
                    # root is something like saves/exp2_grid/Qwen3-1.7B-Base_2000000/insfo0.375_math0.125_code0.125_algebra0.25_math-grade0.125
                    exp_name = Path(root).name
                    model_name = Path(root).parent.name
                    
                    if model_name not in results:
                        results[model_name] = {}
                        
                    results[model_name][exp_name] = min_loss
            except Exception as e:
                print(f"Failed to process {trainer_state_path}: {e}")

    # Output results
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Aggregated loss collected and saved to {args.output_json}")

if __name__ == "__main__":
    main()
