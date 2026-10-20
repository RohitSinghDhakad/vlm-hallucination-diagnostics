import os
# Must be set before any huggingface / transformers imports
os.environ["HF_HOME"] = r"D:\huggingface_cache"

import json
import csv
import torch
import requests
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForQuestionAnswering

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
POPE_SPLITS = {
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
    "popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json"
}

IMAGE_DIR = r"D:\datasets\COCO\val2014"
MAX_SAMPLES = 500
MODEL_ID = "Salesforce/blip-vqa-base"

def download_file(url, local_path):
    if not os.path.exists(local_path):
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def evaluate_split(split_name, json_url, processor, model, device, dtype):
    print(f"\n--- Evaluating {split_name.capitalize()} Split ---")
    
    # Download JSON
    json_path = f"coco_pope_{split_name}.json"
    download_file(json_url, json_path)
    
    samples = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                
    samples = samples[:MAX_SAMPLES]
    print(f"Loaded {len(samples)} samples.")
    
    results = []
    yes_count = 0
    tp = tn = fp = fn = 0
    
    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {split_name}")):
        image_name = sample["image"]
        question = sample["text"]
        label = sample["label"].lower()
        
        image_url = f"http://images.cocodataset.org/val2014/{image_name}"
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        # Download image on-demand
        download_file(image_url, image_path)
        
        pred = "no"
        raw_response = ""
        
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"{question} Please answer yes or no."
            
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            if dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].half()
                
            with torch.inference_mode():
                out_ids = model.generate(**inputs, max_new_tokens=10)
            
            raw = processor.decode(out_ids[0], skip_special_tokens=True).strip().lower()
            raw_response = raw
            
            # Parse response
            if raw.startswith("yes"):
                pred = "yes"
            elif raw.startswith("no"):
                pred = "no"
            elif "yes" in raw and "no" not in raw:
                pred = "yes"
            else:
                pred = "no"
                
        finally:
            # Delete local image immediately after processing to save disk space
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    pass
                
        # Append result
        results.append({
            "question_id": sample.get("question_id", idx),
            "image": image_name,
            "question": question,
            "ground_truth": label,
            "prediction": pred,
            "raw_response": raw_response
        })
        
        # Metrics calculation step
        if pred == "yes": yes_count += 1
        
        if label == "yes" and pred == "yes": tp += 1
        elif label == "no" and pred == "no": tn += 1
        elif label == "no" and pred == "yes": fp += 1
        elif label == "yes" and pred == "no": fn += 1

    # Final split metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0
    yes_ratio = yes_count / len(results) if len(results) > 0 else 0.0
    
    # Save CSV for this split
    csv_file = f"pope_{split_name}_results.csv"
    fieldnames = ["question_id", "image", "question", "ground_truth", "prediction", "raw_response"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved predictions to {csv_file}")
        
    metrics = {
        "Split": split_name.capitalize(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Yes Ratio": yes_ratio
    }
    return metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    
    all_metrics = []
    
    # Run evaluation across all 3 protocols
    for split_name, json_url in POPE_SPLITS.items():
        metrics = evaluate_split(split_name, json_url, processor, model, device, dtype)
        all_metrics.append(metrics)
        
    # Export summary CSV
    summary_file = "pope_suite_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Split", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
        
    print(f"\nSaved consolidated summary to {summary_file}")
        
    # Print side-by-side Markdown Table
    print("\n" + "="*90)
    print("  POPE Suite Evaluation Summary (Salesforce/blip-vqa-base)")
    print("="*90)
    print(f"| {'Split':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Yes Ratio':<10} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for m in all_metrics:
        print(f"| {m['Split']:<12} | {m['Accuracy']:.4f}     | {m['Precision']:.4f}     | {m['Recall']:.4f}     | {m['F1 Score']:.4f}     | {m['Yes Ratio']:.4f}     |")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
