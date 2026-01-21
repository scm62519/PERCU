import os
import json
import re
import sys
import math
from datetime import datetime
from PIL import Image
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from deploy.agent import CUAgent

VLLM_BASE_URL = "http://localhost:8030/v1"

MODEL_NAME = "GUI-R1/GUI-R1-3B"

TASK = "preference_task"

MODEL_PATH = f"/data/zkyao/cmshi/model/{MODEL_NAME}"


DATASET_DIR = f"/data/zkyao/cmshi/code/Agent/final_data/{TASK}"

IMAGE_BASE_DIR = "/data/zkyao/cmshi/code/Agent/data/events" 


client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)


def load_task_data(jsonl_path):
    steps = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    return steps

def is_action_match(pred_action, gt_action):

    if not pred_action or not gt_action:
        return False
        
    pred = pred_action.strip().lower()
    gt = gt_action.strip().lower()
    

    if "finish" in gt:
        return "finish" in pred


    coord_pattern = r"(-?\d+(?:\.\d+)?)"
    pred_coords = re.findall(coord_pattern, pred)
    gt_coords = re.findall(coord_pattern, gt)
    
    if pred_coords and gt_coords and len(pred_coords) == len(gt_coords):
        pred_type = pred.split()[0]
        gt_type = gt.split()[0]
        if pred_type == gt_type:
            try:
                total_distance = 0
                for i in range(0, len(pred_coords), 2):
                    px, py = float(pred_coords[i]), float(pred_coords[i+1])
                    gx, gy = float(gt_coords[i]), float(gt_coords[i+1])
                    total_distance += math.sqrt((px - gx)**2 + (py - gy)**2)
                
                threshold = 50.0 * (len(pred_coords) / 2)
                if total_distance < threshold:
                    return True
            except:
                pass 

    def aggressive_clean(s):
        return re.sub(r'[^a-z0-9]', '', s)
    
    if aggressive_clean(pred) == aggressive_clean(gt):
        return True
    
    return False

def evaluate_single_task(file_path, agent):
    steps = load_task_data(file_path)
    task_name = os.path.basename(file_path)
    
    total_steps = len(steps)
    correct_steps = 0
    results = []


    for i, step in enumerate(steps):
        gt_action = step['action']
        screenshot_rel_path = step['screenshot']
        

        thought = step.get('thought', "")
        second_instruction = step.get('second instruction', "")
        

        task_input = (
            f"These are your memories from the last time you performed this task.\n"
            f"{thought}\n"
            f"{second_instruction}\n"
            f"Given the screenshot as below. What's the next step that you will do to help with the task?"
        )
        

        image_path = os.path.join(IMAGE_BASE_DIR, screenshot_rel_path)
        
        if not os.path.exists(image_path):
            print(f"[Error] Image does not exist at this location: {image_path}")
            results.append({"step": i+1, "status": "error", "reason": "image_missing"})
            continue

        with open(image_path, "rb") as img_f:
            image_bytes = img_f.read()
        
        obs = {'screenshot': image_bytes}

        try:
            plan_text, pred_action = agent.predict(task_input, obs)
        except Exception as e:
            print(f"[Error] Failure of reasoning process: {e}")
            pred_action = "Error"

        match = is_action_match(pred_action, gt_action)
        status = "Pass" if match else "Fail"
        if match:
            correct_steps += 1
        
        print(f"  Step {i+1}: {status}")
        print(f"    GT: {gt_action}")
        print(f"    Pred: {pred_action}")
        
        results.append({
            "step": i + 1,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "match": match
        })

    accuracy = (correct_steps / total_steps * 100) if total_steps > 0 else 0
    
    return {
        "task_name": task_name,
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "accuracy": accuracy,
        "details": results
    }

def main():

    agent = CUAgent(client, MODEL_PATH)
    
    all_task_results = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory {DATASET_DIR} does not exist.")
        return

    jsonl_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.jsonl')]
    jsonl_files.sort()

    
    for jsonl_file in tqdm(jsonl_files, desc="Total Progress"):
        file_path = os.path.join(DATASET_DIR, jsonl_file)
        task_result = evaluate_single_task(file_path, agent)
        all_task_results.append(task_result)


    total_all_steps = sum(r['total_steps'] for r in all_task_results)
    total_all_correct = sum(r['correct_steps'] for r in all_task_results)
    
    if len(all_task_results) > 0:
        avg_task_accuracy = sum(r['accuracy'] for r in all_task_results) / len(all_task_results)
    else:
        avg_task_accuracy = 0.0
        
    overall_step_accuracy = (total_all_correct / total_all_steps * 100) if total_all_steps > 0 else 0


    
    final_report = {
        "summary": {
            "total_tasks": len(all_task_results),
            "total_steps": total_all_steps,
            "total_correct": total_all_correct,
            "macro_acc": avg_task_accuracy,
            "micro_acc": overall_step_accuracy
        },
        "tasks": all_task_results
    }
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/data/zkyao/cmshi/code/Agent/evaluate_report"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    

    output_filename = os.path.join(output_dir, f"{MODEL_NAME}_{TASK}_{timestamp_str}.json")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed evaluation results saved to: {output_filename}")

if __name__ == "__main__":
    main()