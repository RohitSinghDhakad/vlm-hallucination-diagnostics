"""
================================================================================
create_aligned_probing_dataset.py - Aligned POPE Probing Validation Dataset
Target Output: D:\pyth_proj\experiments\exp6\aligned_probing_val_dataset.json
Also synchronized to: D:\pyth_proj\experiments\exp4\aligned_probing_val_dataset.json
================================================================================
"""

import os
import json
import random
import pandas as pd
from collections import OrderedDict

# Set fixed random seed for strict scientific reproducibility
random.seed(42)

# ── 1. Directories Setup ──────────────────────────────────────────────────────
EXP6_DIR = r"D:\pyth_proj\experiments\exp6"
EXP4_DIR = r"D:\pyth_proj\experiments\exp4"
os.makedirs(EXP6_DIR, exist_ok=True)
os.makedirs(EXP4_DIR, exist_ok=True)

# ── 2. Locate Input Evaluation Results (300 Images) ───────────────────────────
INPUT_CANDIDATES = [
    r"D:\pyth_proj\experiments\exp3\advanced_metrics_results.csv",
    r"D:\pyth_proj\experiments\exp3\llava_advanced_metrics_results.csv",
    r"D:\pyth_proj\experiments\exp3\qwen_advanced_metrics_results.csv",
    r"D:\pyth_proj\experiments\exp3\instructblip_advanced_metrics_results.csv",
]

RESULTS_CSV = None
for p in INPUT_CANDIDATES:
    if os.path.exists(p):
        RESULTS_CSV = p
        break

if not RESULTS_CSV:
    raise FileNotFoundError(f"Could not locate evaluation CSV in candidates: {INPUT_CANDIDATES}")

print(f"[1/5] Ingesting evaluation image IDs from: {RESULTS_CSV}")
df_results = pd.read_csv(RESULTS_CSV)
unique_image_ids = sorted(df_results["Image ID"].unique().tolist())
print(f"      Total unique images identified: {len(unique_image_ids)}")

# ── 3. Load COCO Annotations ──────────────────────────────────────────────────
ANNO_CANDIDATES = [
    (r"D:\datasets\COCO\annotations\annotations\instances_val2014.json",
     r"D:\datasets\COCO\annotations\annotations\captions_val2014.json"),
    (r"D:\pyth_proj\datasets\COCO\annotations\instances_val2014.json",
     r"D:\pyth_proj\datasets\COCO\annotations\captions_val2014.json"),
    (r"D:\datasets\COCO\annotations\instances_val2014.json",
     r"D:\datasets\COCO\annotations\captions_val2014.json"),
]

INST_PATH, CAPS_PATH = None, None
for inst_c, caps_c in ANNO_CANDIDATES:
    if os.path.exists(inst_c) and os.path.exists(caps_c):
        INST_PATH, CAPS_PATH = inst_c, caps_c
        break

if not INST_PATH:
    raise FileNotFoundError("COCO annotation files instances_val2014.json / captions_val2014.json not found.")

print(f"[2/5] Loading COCO annotations:\n  - Instances: {INST_PATH}\n  - Captions:  {CAPS_PATH}")
with open(INST_PATH, "r", encoding="utf-8") as f:
    instances_data = json.load(f)
with open(CAPS_PATH, "r", encoding="utf-8") as f:
    captions_data = json.load(f)

# Index all 80 COCO categories
cat_id_to_name = {c["id"]: c["name"].lower() for c in instances_data["categories"]}
all_coco_categories = sorted(list(set(cat_id_to_name.values())))
print(f"      Total COCO ground-truth object classes: {len(all_coco_categories)}")

# Index image -> instances
img_to_instances = {}
for a in instances_data["annotations"]:
    img_to_instances.setdefault(a["image_id"], set()).add(cat_id_to_name[a["category_id"]])

# Index image -> reference captions
img_to_captions = {}
for a in captions_data["annotations"]:
    img_to_captions.setdefault(a["image_id"], []).append(a["caption"])

# ── 4. Construct Balanced Probing Questions ───────────────────────────────────
print("[3/5] Parsing ground-truth objects & constructing balanced POPE binary questions...")

aligned_dataset = []
all_probing_questions = []
question_counter = 1

total_yes = 0
total_no = 0

for img_id in unique_image_ids:
    image_filename = f"COCO_val2014_{img_id:012d}.jpg"
    
    # 1. Ground truth present objects from bounding boxes
    present_objs = set(img_to_instances.get(img_id, set()))
    captions = img_to_captions.get(img_id, [])

    # Fallback to caption object matching if image has 0 bounding box annotations
    if len(present_objs) == 0 and captions:
        cap_text = " ".join(captions).lower()
        for cat in all_coco_categories:
            if cat in cap_text:
                present_objs.add(cat)

    # 2. Absent objects from the remaining 80 COCO categories
    absent_objs = set(all_coco_categories) - present_objs

    # 3. Balanced question sampling:
    # Target k positive questions and exactly k negative questions per image
    present_list = sorted(list(present_objs))
    absent_list = sorted(list(absent_objs))

    k = min(3, len(present_list)) if len(present_list) > 0 else 0
    
    image_questions = []

    if k > 0:
        # Sample positive categories
        sampled_pos = random.sample(present_list, k) if len(present_list) >= k else present_list
        # Sample an equal number of negative categories
        sampled_neg = random.sample(absent_list, len(sampled_pos))

        for pos_cat in sampled_pos:
            q_entry = {
                "question_id": question_counter,
                "image": image_filename,
                "image_id": img_id,
                "text": f"Is there a {pos_cat} in the image?",
                "label": "yes",
                "category": pos_cat
            }
            image_questions.append(q_entry)
            all_probing_questions.append(q_entry)
            question_counter += 1
            total_yes += 1

        for neg_cat in sampled_neg:
            q_entry = {
                "question_id": question_counter,
                "image": image_filename,
                "image_id": img_id,
                "text": f"Is there a {neg_cat} in the image?",
                "label": "no",
                "category": neg_cat
            }
            image_questions.append(q_entry)
            all_probing_questions.append(q_entry)
            question_counter += 1
            total_no += 1

    # Shuffle questions per image deterministically
    random.shuffle(image_questions)

    aligned_dataset.append(OrderedDict([
        ("image_id", img_id),
        ("image", image_filename),
        ("present_objs", sorted(list(present_objs))),
        ("absent_objs_count", len(absent_list)),
        ("captions", captions),
        ("question_count", len(image_questions)),
        ("probing_questions", image_questions)
    ]))

# ── 5. Structure & Export Aligned JSON Artifact ───────────────────────────────
output_payload = OrderedDict([
    ("metadata", OrderedDict([
        ("description", "Aligned POPE Probing Validation Dataset for COCO 300 Evaluation Samples"),
        ("total_images", len(aligned_dataset)),
        ("total_probing_questions", len(all_probing_questions)),
        ("total_yes_labels", total_yes),
        ("total_no_labels", total_no),
        ("balanced_ratio", round(total_yes / total_no, 4) if total_no > 0 else 1.0),
        ("source_annotations", os.path.basename(INST_PATH)),
    ])),
    ("images", aligned_dataset),
    ("questions", all_probing_questions)
])

TARGET_EXP6 = os.path.join(EXP6_DIR, "aligned_probing_val_dataset.json")
TARGET_EXP4 = os.path.join(EXP4_DIR, "aligned_probing_val_dataset.json")

print(f"\n[4/5] Exporting aligned JSON dataset...")
with open(TARGET_EXP6, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)
print(f"      Successfully created: {TARGET_EXP6}")

with open(TARGET_EXP4, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)
print(f"      Synchronized to:     {TARGET_EXP4}")

# Also export standard flat POPE JSONL format for direct drop-in benchmark evaluation
JSONL_EXP6 = os.path.join(EXP6_DIR, "coco_pope_aligned_300.json")
with open(JSONL_EXP6, "w", encoding="utf-8") as f:
    for q in all_probing_questions:
        f.write(json.dumps(q) + "\n")
print(f"      POPE benchmark JSONL created: {JSONL_EXP6}")

# ── 6. Verification & Checklist ───────────────────────────────────────────────
print("\n[5/5] Verification Checklist:")
print(f"  [x] Total Images Evaluated      : {len(aligned_dataset)} / 300")
print(f"  [x] Total Probing Questions     : {len(all_probing_questions)}")
print(f"  [x] Positive Queries ('yes')    : {total_yes}")
print(f"  [x] Negative Queries ('no')     : {total_no}")
print(f"  [x] Yes/No Balance Ratio        : {total_yes / total_no:.2f}:1.00 (Strictly Balanced)")
print(f"  [x] File Exists in exp6         : {os.path.exists(TARGET_EXP6)} ({os.path.getsize(TARGET_EXP6) / 1024:.1f} KB)")
print(f"  [x] File Exists in exp4         : {os.path.exists(TARGET_EXP4)} ({os.path.getsize(TARGET_EXP4) / 1024:.1f} KB)")
print("\nAll pipeline tasks completed successfully.")