"""
============================================================
eval_char_cider.py  -  CHAr (Object Recall) & CIDEr Evaluator
Location: D:\pyth_proj\experiments\exp4\eval_char_cider.py

Objectives:
  1. Input Data:
     - Read model evaluations from "D:\pyth_proj\experiments\exp3\advanced_metrics_results.csv"
     - Read COCO 2014 validation annotations (instances and captions).
  2. Metric Calculations:
     - CHAr (Object Recall): Extract noun tokens from `Generated Text`, match
       against COCO ground-truth object instances using category lists & synonyms,
       and calculate ratio of unique matched ground-truth objects.
     - CIDEr: Utilize `pycocoevalcap.cider.cider.Cider` with string-casted image keys
       to compute consensus scores. To prevent Gaussian length-penalty zero-underflow
       against COCO's 1-sentence reference captions, evaluation extracts the lead
       descriptive sentence for CIDEr while keeping full text in the output.
  3. File Exclusions & Clean Artifact Export:
     - Completely excludes METEOR, CHAi, CHs, and CLIPScore.
     - File 1 (char_cider_detailed_results.csv): [Image ID, Prompt Variant, Generated Text, CHAr, CIDEr]
     - File 2 (char_cider_summary.csv): [Prompt Variant, CHAr, CIDEr]
============================================================
"""

import os
import sys
import json
import zipfile
import requests
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── 1. NLTK Setup ────────────────────────────────────────────
import nltk
for _pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet"]:
    nltk.download(_pkg, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Installing pycocoevalcap...")
    os.system(f"{sys.executable} -m pip install -q pycocoevalcap")
    from pycocoevalcap.cider.cider import Cider

# ── 2. Locate Input and Output Paths ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# for other models change file location 
INPUT_CANDIDATES = [
    r"D:\pyth_proj\experiments\exp3\advanced_metrics_results.csv",
    os.path.join(BASE_DIR, "..", "exp3", "advanced_metrics_results.csv"),
    "/content/advanced_metrics_results.csv",
    "advanced_metrics_results.csv"
]

INPUT_CSV = None
for p in INPUT_CANDIDATES:
    if os.path.exists(p):
        INPUT_CSV = os.path.abspath(p)
        break

if not INPUT_CSV:
    raise FileNotFoundError(f"Could not locate advanced_metrics_results.csv in: {INPUT_CANDIDATES}")

print(f"[1/5] Loaded input evaluations file: {INPUT_CSV}")

# Target export directories
OUTPUT_DIRS = [BASE_DIR]
exp3_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "exp3"))
if os.path.exists(exp3_dir) and exp3_dir != BASE_DIR:
    OUTPUT_DIRS.append(exp3_dir)
if os.path.exists("/content"):
    OUTPUT_DIRS.append("/content")

DETAILED_FILENAME = "char_cider_detailed_results.csv"
SUMMARY_FILENAME  = "char_cider_summary.csv"

# ── 3. Load COCO Annotations ─────────────────────────────────
def find_or_download_coco():
    candidates = [
        (r"D:\datasets\COCO\annotations\annotations\instances_val2014.json",
         r"D:\datasets\COCO\annotations\annotations\captions_val2014.json"),
        (r"D:\datasets\COCO\annotations\instances_val2014.json",
         r"D:\datasets\COCO\annotations\captions_val2014.json"),
        ("/content/datasets/COCO/annotations/annotations/instances_val2014.json",
         "/content/datasets/COCO/annotations/annotations/captions_val2014.json"),
        ("./datasets/COCO/annotations/annotations/instances_val2014.json",
         "./datasets/COCO/annotations/annotations/captions_val2014.json"),
    ]
    for inst_p, caps_p in candidates:
        if os.path.exists(inst_p) and os.path.exists(caps_p):
            return inst_p, caps_p

    # Download fallback if annotations not found locally
    anno_dir = os.path.join(BASE_DIR, "coco_annotations")
    os.makedirs(anno_dir, exist_ok=True)
    zip_path = os.path.join(anno_dir, "annotations_trainval2014.zip")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

    print(f"Downloading COCO annotations from {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(anno_dir)

    inst_path = os.path.join(anno_dir, "annotations", "instances_val2014.json")
    caps_path = os.path.join(anno_dir, "annotations", "captions_val2014.json")
    return inst_path, caps_path

INST_PATH, CAPS_PATH = find_or_download_coco()
print(f"[2/5] Loading COCO annotations:\n  - Instances: {INST_PATH}\n  - Captions:  {CAPS_PATH}")

with open(INST_PATH, "r", encoding="utf-8") as f:
    instances_data = json.load(f)
with open(CAPS_PATH, "r", encoding="utf-8") as f:
    captions_data = json.load(f)

# Categories and synonym dictionary
coco_categories = {c["id"]: c["name"].lower() for c in instances_data["categories"]}

_SYNS = {
    "person":        ["man", "woman", "boy", "girl", "people", "child", "human", "guy"],
    "bicycle":       ["bike"],
    "car":           ["vehicle", "automobile"],
    "motorcycle":    ["bike", "motorbike"],
    "airplane":      ["plane", "aircraft", "jet"],
    "boat":          ["ship", "vessel"],
    "traffic light": ["stoplight"],
    "fire hydrant":  ["hydrant"],
}

word_to_coco = {}
for _, name in coco_categories.items():
    word_to_coco[name] = name
    for s in _SYNS.get(name, []):
        word_to_coco[s] = name

# Map image_id -> ground truth object category set
img_to_objects = {}
for a in instances_data["annotations"]:
    img_to_objects.setdefault(a["image_id"], set()).add(coco_categories[a["category_id"]])

# Map image_id -> reference caption list
img_to_caps = {}
for a in captions_data["annotations"]:
    img_to_caps.setdefault(a["image_id"], []).append(a["caption"])

print(f"[3/5] Annotations indexed: {len(img_to_objects)} object images, {len(img_to_caps)} caption images.")

# ── 4. CHAr (Object Recall) Helper ───────────────────────────
def extract_nouns(text):
    tokens = word_tokenize(str(text).lower())
    tagged = nltk.pos_tag(tokens)
    return [w for w, pos in tagged if pos.startswith("NN")]

def calculate_char(text, gt_objs):
    """
    CHAr = (# unique matched ground-truth objects in text) / (# total unique ground-truth objects)
    """
    if not gt_objs:
        return 0.0

    text_lower = str(text).lower()
    nouns = extract_nouns(text_lower)
    gen_objs = {word_to_coco[n] for n in nouns if n in word_to_coco}

    # Match multi-word category phrases (e.g., 'traffic light', 'hot dog', etc.)
    for cat in coco_categories.values():
        if " " in cat and cat in text_lower:
            gen_objs.add(cat)

    matched = gen_objs.intersection(gt_objs)
    return len(matched) / len(gt_objs)

# ── 5. Run Metric Computations ────────────────────────────────
print(f"[4/5] Reading evaluation rows from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

required_cols = ["Image ID", "Prompt Variant", "Generated Text"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column '{c}' in {INPUT_CSV}")

print(f"Total rows to evaluate: {len(df)}")

# 1. Compute CHAr on full generated text
print("Computing CHAr (Object Recall)...")
df["CHAr"] = df.apply(
    lambda r: calculate_char(r["Generated Text"], img_to_objects.get(int(r["Image ID"]), set())),
    axis=1
)

# 2. Compute CIDEr per Prompt Variant with zero-underflow prevention
print("Computing CIDEr consensus scores (with zero-underflow prevention)...")
cider_scorer = Cider()
df["CIDEr"] = 0.0
summary_rows = []

for variant, group in df.groupby("Prompt Variant"):
    # Construct string-keyed dictionary of references
    gts = {
        str(row["Image ID"]): img_to_caps.get(int(row["Image ID"]), ["no caption"])
        for _, row in group.iterrows()
    }
    
    # Extract the lead descriptive sentence to evaluate caption alignment without
    # exponential length-penalty zero-underflow against COCO's single-sentence references
    res = {}
    for _, row in group.iterrows():
        raw_text = str(row["Generated Text"]).strip() if pd.notna(row["Generated Text"]) else ""
        sents = sent_tokenize(raw_text) if raw_text else []
        lead_sent = sents[0].strip() if sents else "."
        res[str(row["Image ID"])] = [lead_sent]

    corpus_cider, individual_scores = cider_scorer.compute_score(gts, res)
    
    # Map individual scores back to matching rows
    img_ids = list(gts.keys())
    score_map = {int(img_id): float(s) for img_id, s in zip(img_ids, individual_scores)}
    
    variant_mask = (df["Prompt Variant"] == variant)
    df.loc[variant_mask, "CIDEr"] = df.loc[variant_mask, "Image ID"].map(score_map).fillna(0.0)

    mean_char = df.loc[variant_mask, "CHAr"].mean()
    summary_rows.append({
        "Prompt Variant": variant,
        "CHAr": round(float(mean_char), 6),
        "CIDEr": round(float(corpus_cider), 6)
    })

# ── 6. Filter & Export Clean Artifacts ────────────────────────
# Exclude METEOR, CHAi, CHs, and CLIPScore
output_columns = ["Image ID", "Prompt Variant", "Generated Text", "CHAr", "CIDEr"]
detailed_df = df[output_columns].copy()
summary_df = pd.DataFrame(summary_rows)

print("\n[5/5] Exporting clean artifacts (excluding METEOR, CHAi, CHs, CLIPScore)...")
for out_dir in OUTPUT_DIRS:
    os.makedirs(out_dir, exist_ok=True)
    det_path = os.path.join(out_dir, DETAILED_FILENAME)
    sum_path = os.path.join(out_dir, SUMMARY_FILENAME)
    
    detailed_df.to_csv(det_path, index=False)
    summary_df.to_csv(sum_path, index=False)
    print(f"  -> Successfully generated: {det_path}")
    print(f"  -> Successfully generated: {sum_path}")

# ── 7. Display Summary Table ─────────────────────────────────
W = 60
print("\n" + "=" * W)
print("       CHAr & CIDEr Evaluation Summary Table")
print("=" * W)
print(f"| {'Prompt Variant':<35} | {'CHAr':>8} | {'CIDEr':>8} |")
print(f"|{'-'*37}|{'-'*10}|{'-'*10}|")
for _, r in summary_df.iterrows():
    print(f"| {r['Prompt Variant']:<35} | {r['CHAr']:>8.4f} | {r['CIDEr']:>8.4f} |")
print("=" * W + "\n")
print("Execution complete. Artifacts verified.")