# ============================================================
# eval_advanced_metrics_colab.py  –  Flat / Colab-native
# Run as: exec(open("eval_advanced_metrics_colab.py").read())
# or paste cell-by-cell into a Colab notebook.
#
# Prereq cell (run once, then RESTART RUNTIME):
#   !pip install -q transformers accelerate bitsandbytes
# ============================================================

import os, warnings, json, zipfile
warnings.filterwarnings("ignore")

# ── 1. Environment (set BEFORE any HF/torch import) ─────────
os.environ["HF_HOME"]                   = "/content/cache/huggingface"
os.environ["TORCH_HOME"]                = "/content/cache/torch"
os.environ["TMPDIR"]                    = "/content/tmp"
os.environ["TEMP"]                      = "/content/tmp"
os.environ["TMP"]                       = "/content/tmp"
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"   # Colab proxy SSRF opt-in

for _d in ["/content/tmp", "/content/cache/huggingface",
           "/content/cache/torch"]:
    os.makedirs(_d, exist_ok=True)

# ── 2. Imports ───────────────────────────────────────────────
import torch, requests, nltk, pandas as pd
from PIL import Image
from tqdm import tqdm
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          CLIPProcessor, CLIPModel, BitsAndBytesConfig)

# ── 3. NLTK data ─────────────────────────────────────────────
NLTK_DIR = "/content/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)
for _pkg in ["punkt", "punkt_tab", "wordnet",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(_pkg, download_dir=NLTK_DIR, quiet=True)

# ── 4. Paths & constants ─────────────────────────────────────
IMAGE_DIR   = "/content/datasets/COCO/val2014"
ANNO_DIR    = "/content/datasets/COCO/annotations"
RESULTS_CSV = "/content/advanced_metrics_results.csv"
SUMMARY_CSV = "/content/advanced_metrics_summary.csv"
MAX_SAMPLES = 300
MODEL_ID    = "llava-hf/llava-1.5-7b-hf"
CLIP_ID     = "openai/clip-vit-base-patch32"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNO_DIR,  exist_ok=True)

# ── 5. Download helper ───────────────────────────────────────
def download_file(url, path):
    if not os.path.exists(path):
        print(f"  ↓ {url}")
        r = requests.get(url, stream=True); r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    return path

# ── 6. COCO annotations ──────────────────────────────────────
ANNO_ZIP = os.path.join(ANNO_DIR, "annotations_trainval2014.zip")
INST_PATH = os.path.join(ANNO_DIR, "annotations", "instances_val2014.json")
CAPS_PATH = os.path.join(ANNO_DIR, "annotations", "captions_val2014.json")

if not os.path.exists(INST_PATH) or not os.path.exists(CAPS_PATH):
    download_file("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", ANNO_ZIP)
    print("Extracting annotations…")
    with zipfile.ZipFile(ANNO_ZIP) as z: z.extractall(ANNO_DIR)

print("Loading COCO annotations…")
with open(INST_PATH) as f: instances_data = json.load(f)
with open(CAPS_PATH) as f: captions_data  = json.load(f)

coco_categories = {c["id"]: c["name"].lower() for c in instances_data["categories"]}
_SYNS = {
    "person":        ["man","woman","boy","girl","people","child","human","guy"],
    "bicycle":       ["bike"],
    "car":           ["vehicle","automobile"],
    "motorcycle":    ["bike","motorbike"],
    "airplane":      ["plane","aircraft","jet"],
    "boat":          ["ship","vessel"],
    "traffic light": ["stoplight"],
    "fire hydrant":  ["hydrant"],
}
word_to_coco = {}
for _, name in coco_categories.items():
    word_to_coco[name] = name
    for s in _SYNS.get(name, []): word_to_coco[s] = name

img_to_caps    = {}
img_to_objects = {}
for a in captions_data["annotations"]:
    img_to_caps.setdefault(a["image_id"], []).append(a["caption"])
for a in instances_data["annotations"]:
    img_to_objects.setdefault(a["image_id"], set()).add(
        coco_categories[a["category_id"]])

images_info = captions_data["images"][:MAX_SAMPLES]

# ── 7. CHAIR helpers ─────────────────────────────────────────
def extract_nouns(text):
    return [w for w, pos in nltk.pos_tag(word_tokenize(text.lower()))
            if pos.startswith("NN")]

def calculate_chair(text, gt_objs):
    sents = sent_tokenize(text)
    tot = hall = hall_s = 0
    for s in sents:
        hits = [word_to_coco[n] for n in extract_nouns(s) if n in word_to_coco]
        s_hall = False
        for o in hits:
            tot += 1
            if o not in gt_objs: hall += 1; s_hall = True
        if s_hall: hall_s += 1
    return (hall/tot if tot else 0.0), (hall_s/len(sents) if sents else 0.0)

# ── 8. CLIP embedding helpers (robust to any transformers ver) ─
def _unwrap(t):
    """Extract bare tensor from HF model output objects."""
    return t if isinstance(t, torch.Tensor) else (
           t.pooler_output if hasattr(t, "pooler_output") else
           t.last_hidden_state[:, 0])

def clip_img_emb(image):
    inp = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        e = _unwrap(clip_model.get_image_features(**inp))
    return e / e.norm(p=2, dim=-1, keepdim=True)

def clip_txt_emb(text):
    inp = clip_processor(text=[text], return_tensors="pt",
                         padding=True, truncation=True).to(device)
    with torch.no_grad():
        e = _unwrap(clip_model.get_text_features(**inp))
    return e / e.norm(p=2, dim=-1, keepdim=True)

# ── 9. Prompts ───────────────────────────────────────────────
PROMPTS = {
    "Prompt A (Baseline)": (
        "USER: <image>\nDescribe what is happening in this image.\nASSISTANT:"),
    "Prompt B (Forced Hallucination)": (
        "USER: <image>\nDescribe what is happening in this image in extreme detail. "
        "Include all background objects, fine elements, and contextual items.\nASSISTANT:"),
    "Prompt C (Constrained)": (
        "USER: <image>\nDescribe what is happening in this image. "
        "Only mention objects and details you are completely certain are present. "
        "Do not guess.\nASSISTANT:"),
}

# ── 10. Load models (global scope — survives across cells) ───
print(f"\nLoading {MODEL_ID} in 4-bit…")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto")
llava_model.eval()

print(f"Loading {CLIP_ID}…")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model     = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

print("Models ready.\n")

# ── 11. Evaluation loop ──────────────────────────────────────
results = []

for info in tqdm(images_info, desc="Evaluating images"):
    img_id     = info["id"]
    file_name  = info["file_name"]
    local_path = os.path.join(IMAGE_DIR, file_name)

    download_file(info["coco_url"], local_path)

    try:
        image       = Image.open(local_path).convert("RGB")
        gt_captions = img_to_caps.get(img_id, [])
        gt_objects  = img_to_objects.get(img_id, set())
        ie          = clip_img_emb(image)          # CLIP image embedding

        for p_name, p_text in PROMPTS.items():
            llava_in = llava_processor(
                text=p_text, images=image, return_tensors="pt"
            ).to(llava_model.device)

            with torch.inference_mode():
                out_ids = llava_model.generate(
                    **llava_in, max_new_tokens=128,
                    temperature=0.2, do_sample=True)

            gen = llava_processor.decode(out_ids[0], skip_special_tokens=True)
            gen = gen.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in gen else gen

            # METEOR
            tok_gt  = [word_tokenize(c.lower()) for c in gt_captions]
            tok_gen = word_tokenize(gen.lower())
            meteor  = 0.0
            if tok_gt and tok_gen:
                try: meteor = meteor_score.meteor_score(tok_gt, tok_gen)
                except: pass

            chai, chs = calculate_chair(gen, gt_objects)
            te        = clip_txt_emb(gen)
            clipscore = (ie @ te.T).item() * 100.0

            results.append({"Image ID": img_id, "Prompt Variant": p_name,
                             "Generated Text": gen, "METEOR": meteor,
                             "CHAi": chai, "CHs": chs, "CLIPScore": clipscore})
    finally:
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass

# ── 12. Save & print ─────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

summary = (df.groupby("Prompt Variant")[["METEOR","CHAi","CHs","CLIPScore"]]
             .mean().reset_index())
summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nDetailed  → {RESULTS_CSV}")
print(f"Summary   → {SUMMARY_CSV}\n")

W = 87
print("=" * W)
print("  Advanced Metrics Summary — llava-1.5-7b-hf (4-bit, T4)")
print("=" * W)
print(f"| {'Prompt Variant':<35} | {'METEOR':>8} | {'CHAi':>8} | {'CHs':>8} | {'CLIPScore':>9} |")
print(f"|{'-'*37}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|")
for _, r in summary.iterrows():
    print(f"| {r['Prompt Variant']:<35} | {r['METEOR']:>8.4f} | {r['CHAi']:>8.4f} | {r['CHs']:>8.4f} | {r['CLIPScore']:>9.4f} |")
print("=" * W + "\n")
