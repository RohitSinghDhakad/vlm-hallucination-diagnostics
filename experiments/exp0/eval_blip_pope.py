import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"
import json
import csv
import torch
import requests
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForQuestionAnswering

def download_file(url, local_path):
    if not os.path.exists(local_path):
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"  Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def evaluate_pope():
    # ------------------------------------------------------------------
    # Model: Salesforce/blip-vqa-base
    #   ~990 MB, pure PIL/PyTorch, no native DLL deps, runs on CPU.
    #   Designed for yes/no VQA -- maps cleanly to POPE evaluation.
    # ------------------------------------------------------------------
    model_id = "Salesforce/blip-vqa-base"
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype    = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model     = BlipForQuestionAnswering.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    # POPE adversarial JSON
    pope_url  = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json"
    pope_file = "coco_pope_adversarial.json"
    download_file(pope_url, pope_file)

    samples = []
    with open(pope_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[:100]
    print(f"Evaluating on {len(samples)} samples...\n")

    results   = []
    image_dir = r"D:\datasets\COCO\val2014"

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        image_name = sample["image"]
        question   = sample["text"]
        label      = sample["label"].lower()

        # Download individual COCO image on demand
        image_url  = f"http://images.cocodataset.org/val2014/{image_name}"
        image_path = os.path.join(image_dir, image_name)
        download_file(image_url, image_path)

        image = Image.open(image_path).convert("RGB")

        # Prompt: POPE-style yes/no question
        prompt = f"{question} Please answer yes or no."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        if dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=10)
        raw = processor.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Parse binary yes/no
        if   raw.startswith("yes"):                    pred = "yes"
        elif raw.startswith("no"):                     pred = "no"
        elif "yes" in raw and "no" not in raw:         pred = "yes"
        else:                                          pred = "no"

        results.append({
            "question_id" : sample.get("question_id", idx),
            "image"       : image_name,
            "question"    : question,
            "ground_truth": label,
            "prediction"  : pred,
            "raw_response": raw,
        })

    # Metrics
    tp = tn = fp = fn = 0
    for r in results:
        gt, p = r["ground_truth"], r["prediction"]
        if   gt == "yes" and p == "yes": tp += 1
        elif gt == "no"  and p == "no" : tn += 1
        elif gt == "no"  and p == "yes": fp += 1
        elif gt == "yes" and p == "no" : fn += 1

    precision = tp / (tp + fp)                          if (tp + fp) > 0           else 0.0
    recall    = tp / (tp + fn)                          if (tp + fn) > 0           else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0  else 0.0
    accuracy  = (tp + tn) / len(results)                if results                 else 0.0

    print(f"\n{'='*55}")
    print(f"  POPE Adversarial -- {len(results)} samples (blip-vqa-base)")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*55}\n")

    # CSV export
    csv_file   = "llava_pope_results.csv"
    fieldnames = ["question_id", "image", "question", "ground_truth", "prediction", "raw_response"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({})
        for name, val in [("Accuracy", accuracy), ("Precision", precision),
                          ("Recall", recall), ("F1 Score", f1)]:
            writer.writerow({"question_id": name, "image": f"{val:.4f}"})

    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    evaluate_pope()
