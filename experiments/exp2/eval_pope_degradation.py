import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

import json
import csv
import torch
import requests
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, BlipForQuestionAnswering
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
POPE_SPLITS = {
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json"
}

IMAGE_DIR = r"D:\datasets\COCO\val2014"
MAX_SAMPLES = 100
MODEL_ID = "Salesforce/blip-vqa-base"
BLUR_LEVELS = [0, 3, 7, 12]

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

def evaluate_degradation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    all_results = []
    
    for split_name, json_url in POPE_SPLITS.items():
        print(f"\n--- Evaluating {split_name.capitalize()} Split ---")
        json_path = f"coco_pope_{split_name}.json"
        download_file(json_url, json_path)
        
        samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        samples = samples[:MAX_SAMPLES]
        
        metrics_tracker = {r: {'tp':0, 'tn':0, 'fp':0, 'fn':0, 'yes_count':0, 'total':0} for r in BLUR_LEVELS}
        
        for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {split_name}")):
            image_name = sample["image"]
            question = sample["text"]
            label = sample["label"].lower()
            
            image_url = f"http://images.cocodataset.org/val2014/{image_name}"
            image_path = os.path.join(IMAGE_DIR, image_name)
            download_file(image_url, image_path)
            
            try:
                base_image = Image.open(image_path).convert("RGB")
                prompt = f"{question} Please answer yes or no."
                
                for radius in BLUR_LEVELS:
                    if radius > 0:
                        blurred_img = base_image.filter(ImageFilter.GaussianBlur(radius))
                    else:
                        blurred_img = base_image.copy()
                        
                    inputs = processor(images=blurred_img, text=prompt, return_tensors="pt").to(device)
                    if dtype == torch.float16:
                        inputs["pixel_values"] = inputs["pixel_values"].half()
                    
                    with torch.inference_mode():
                        out_ids = model.generate(**inputs, max_new_tokens=10)
                        
                    raw = processor.decode(out_ids[0], skip_special_tokens=True).strip().lower()
                    
                    if raw.startswith("yes"): pred = "yes"
                    elif raw.startswith("no"): pred = "no"
                    elif "yes" in raw and "no" not in raw: pred = "yes"
                    else: pred = "no"
                    
                    m = metrics_tracker[radius]
                    m['total'] += 1
                    if pred == "yes": m['yes_count'] += 1
                    
                    if label == "yes" and pred == "yes": m['tp'] += 1
                    elif label == "no" and pred == "no": m['tn'] += 1
                    elif label == "no" and pred == "yes": m['fp'] += 1
                    elif label == "yes" and pred == "no": m['fn'] += 1

            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
                    
        for r in BLUR_LEVELS:
            m = metrics_tracker[r]
            tp, tn, fp, fn, total = m['tp'], m['tn'], m['fp'], m['fn'], m['total']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / total if total > 0 else 0.0
            yes_ratio = m['yes_count'] / total if total > 0 else 0.0
            
            all_results.append({
                "Split": split_name.capitalize(),
                "Blur Radius": r,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Yes Ratio": yes_ratio
            })
            
    df = pd.DataFrame(all_results)
    
    pdi_data = []
    for r in BLUR_LEVELS:
        adv_yr = df[(df['Split'] == 'Adversarial') & (df['Blur Radius'] == r)]['Yes Ratio'].values[0]
        rnd_yr = df[(df['Split'] == 'Random') & (df['Blur Radius'] == r)]['Yes Ratio'].values[0]
        pdi = adv_yr - rnd_yr
        pdi_data.append({"Blur Radius": r, "PDI": pdi})
    
    df_pdi = pd.DataFrame(pdi_data)
    
    # Export metrics
    df.to_csv("pope_degradation_metrics.csv", index=False)
    print("\nSaved pope_degradation_metrics.csv")
    
    print("Generating Matplotlib plots...")
    
    # 1. Precision Decay Curve
    plt.figure(figsize=(8,6))
    plt.plot(df[df['Split'] == 'Adversarial']['Blur Radius'], df[df['Split'] == 'Adversarial']['Precision'], marker='o', label='Adversarial')
    plt.plot(df[df['Split'] == 'Random']['Blur Radius'], df[df['Split'] == 'Random']['Precision'], marker='s', label='Random')
    plt.title('Precision Decay under Visual Degradation')
    plt.xlabel('Gaussian Blur Radius')
    plt.ylabel('Precision')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig('precision_decay_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Yes Ratio Spike
    plt.figure(figsize=(8,6))
    plt.plot(df[df['Split'] == 'Adversarial']['Blur Radius'], df[df['Split'] == 'Adversarial']['Yes Ratio'], marker='o', label='Adversarial')
    plt.plot(df[df['Split'] == 'Random']['Blur Radius'], df[df['Split'] == 'Random']['Yes Ratio'], marker='s', label='Random')
    plt.title('Yes Ratio / Positive Bias Spike')
    plt.xlabel('Gaussian Blur Radius')
    plt.ylabel('Yes Ratio')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig('yes_ratio_spike.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PDI Trend
    plt.figure(figsize=(8,6))
    plt.bar(df_pdi['Blur Radius'].astype(str), df_pdi['PDI'], color='coral')
    plt.title('Prior Dependency Index (PDI) Trend')
    plt.xlabel('Gaussian Blur Radius')
    plt.ylabel('PDI Score (Adversarial Yes Ratio - Random Yes Ratio)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('pdi_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 3 PNG charts: precision_decay_curve.png, yes_ratio_spike.png, pdi_trend.png")
    
    # Terminal Summary Markdown
    print("\n" + "="*95)
    print("  POPE Degradation Evaluation Summary (Salesforce/blip-vqa-base)")
    print("="*95)
    print(f"| {'Split':<12} | {'Blur':<4} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Yes Ratio':<10} |")
    print(f"|{'-'*14}|{'-'*6}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for _, row in df.iterrows():
        print(f"| {row['Split']:<12} | {int(row['Blur Radius']):<4} | {row['Accuracy']:.4f}     | {row['Precision']:.4f}     | {row['Recall']:.4f}     | {row['F1 Score']:.4f}     | {row['Yes Ratio']:.4f}     |")
    print("="*95)
    
    print("\nPrior Dependency Index (PDI):")
    for _, row in df_pdi.iterrows():
        print(f"Blur {int(row['Blur Radius'])}: PDI = {row['PDI']:.4f}")
        
if __name__ == "__main__":
    evaluate_degradation()
