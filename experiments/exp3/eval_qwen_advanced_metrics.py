# ============================================================
# eval_qwen_advanced_metrics.py  -  Colab T4 GPU
# Benchmarks Qwen2-VL on METEOR / CHAIR / CLIPScore
# Model priority: 7B-AWQ (4.5GB) > 7B-4bit > 2B-fp16 (4GB)
#
# Prereq cell (run ONCE, then RESTART RUNTIME):
#   !pip install -q transformers torch torchvision pillow requests \
#                   nltk pandas tqdm accelerate autoawq
# ============================================================

import os, warnings, json, zipfile
warnings.filterwarnings("ignore")

# ── 1. Env vars (BEFORE any HF/torch import) ─────────────────
os.environ["HF_HOME"]                    = "/content/cache/huggingface"
os.environ["TORCH_HOME"]                 = "/content/cache/torch"
os.environ["TMPDIR"]                     = "/content/tmp"
os.environ["TEMP"]                       = "/content/tmp"
os.environ["TMP"]                        = "/content/tmp"
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True"

for _d in ["/content/tmp","/content/cache/huggingface","/content/cache/torch"]:
    os.makedirs(_d, exist_ok=True)

# ── 2. Imports ───────────────────────────────────────────────
import torch, requests, nltk, pandas as pd
from PIL import Image
from tqdm import tqdm
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (AutoProcessor,
                          Qwen2VLForConditionalGeneration,
                          CLIPProcessor, CLIPModel,
                          BitsAndBytesConfig)

# ── 3. NLTK data ─────────────────────────────────────────────
NLTK_DIR = "/content/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)
for _pkg in ["punkt","punkt_tab","wordnet",
             "averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
    nltk.download(_pkg, download_dir=NLTK_DIR, quiet=True)

# ── 4. Paths ─────────────────────────────────────────────────
IMAGE_DIR   = "/content/datasets/COCO/val2014"
ANNO_DIR    = "/content/datasets/COCO/annotations"
RESULTS_CSV = "/content/qwen_advanced_metrics_results.csv"
SUMMARY_CSV = "/content/qwen_advanced_metrics_summary.csv"
MAX_SAMPLES = 300
CLIP_ID     = "openai/clip-vit-base-patch32"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNO_DIR,  exist_ok=True)

# ── 5. Download helper ───────────────────────────────────────
def download_file(url, path):
    if not os.path.exists(path):
        print(f"  down {url}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    return path

# ── 6. COCO annotations ──────────────────────────────────────
ANNO_ZIP  = os.path.join(ANNO_DIR, "annotations_trainval2014.zip")
INST_PATH = os.path.join(ANNO_DIR, "annotations", "instances_val2014.json")
CAPS_PATH = os.path.join(ANNO_DIR, "annotations", "captions_val2014.json")

if not os.path.exists(INST_PATH) or not os.path.exists(CAPS_PATH):
    download_file(
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        ANNO_ZIP)
    print("Extracting annotations...")
    with zipfile.ZipFile(ANNO_ZIP) as z: z.extractall(ANNO_DIR)

print("Loading COCO annotations...")
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
    sents  = sent_tokenize(text)
    tot = hall = hall_s = 0
    for s in sents:
        hits   = [word_to_coco[n] for n in extract_nouns(s) if n in word_to_coco]
        s_hall = False
        for o in hits:
            tot += 1
            if o not in gt_objs: hall += 1; s_hall = True
        if s_hall: hall_s += 1
    return (hall/tot if tot else 0.0), (hall_s/len(sents) if sents else 0.0)

# ── 8. CLIP helpers — CPU only (preserves all VRAM for Qwen) ─
CLIP_DEVICE = "cpu"

def _unwrap(t):
    if isinstance(t, torch.Tensor): return t
    if hasattr(t, "pooler_output"): return t.pooler_output
    return t.last_hidden_state[:, 0]

def clip_img_emb(image):
    inp = clip_processor(images=image, return_tensors="pt").to(CLIP_DEVICE)
    with torch.no_grad():
        e = _unwrap(clip_model.get_image_features(**inp))
    return e / e.norm(p=2, dim=-1, keepdim=True)

def clip_txt_emb(text):
    inp = clip_processor(text=[text], return_tensors="pt",
                         padding=True, truncation=True).to(CLIP_DEVICE)
    with torch.no_grad():
        e = _unwrap(clip_model.get_text_features(**inp))
    return e / e.norm(p=2, dim=-1, keepdim=True)

# ── 9. Qwen input builder ─────────────────────────────────────
def build_qwen_inputs(pil_image, prompt_text):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text":  prompt_text},
    ]}]
    text_prompt = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return qwen_processor(
        text=[text_prompt], images=[pil_image],
        padding=True, return_tensors="pt",
    ).to(QWEN_DEVICE)

# ── 10. Prompts ───────────────────────────────────────────────
PROMPT_TEXTS = {
    "Prompt A (Baseline)":
        "Describe what is happening in this image.",
    "Prompt B (Forced Hallucination)":
        "Describe what is happening in this image in extreme detail. "
        "Include all background objects, fine elements, and contextual items.",
    "Prompt C (Constrained)":
        "Describe what is happening in this image. "
        "Only mention objects and details you are completely certain are present. "
        "Do not guess.",
}

# ── 11. Smart model loading (3-tier: AWQ > 4-bit > 2B-fp16) ──
qwen_model    = None
qwen_processor = None
MODEL_USED    = ""

# Tier 1: Pre-quantized AWQ (7B quality, ~4.5 GB VRAM, no bitsandbytes)
try:
    _mid = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
    print(f"\nAttempting Tier 1: {_mid} (AWQ, ~4.5 GB)...")
    qwen_processor = AutoProcessor.from_pretrained(_mid, trust_remote_code=True)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        _mid, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    MODEL_USED = _mid
    print(f"Loaded: {_mid}")
except Exception as e:
    print(f"Tier 1 failed: {e}")

# Tier 2: bitsandbytes 4-bit NF4 (7B, ~5 GB VRAM)
if qwen_model is None:
    try:
        _mid = "Qwen/Qwen2-VL-7B-Instruct"
        print(f"Attempting Tier 2: {_mid} (4-bit NF4, ~5 GB)...")
        qwen_processor = AutoProcessor.from_pretrained(_mid, trust_remote_code=True)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            _mid, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True)
        MODEL_USED = _mid + " (4-bit NF4)"
        print(f"Loaded: {_mid} in 4-bit NF4")
    except Exception as e:
        print(f"Tier 2 failed: {e}")

# Tier 3: Qwen2-VL-2B-Instruct float16 (guaranteed ~4 GB, fits easily)
if qwen_model is None:
    _mid = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Attempting Tier 3: {_mid} (float16, ~4 GB — guaranteed fit)...")
    qwen_processor = AutoProcessor.from_pretrained(_mid, trust_remote_code=True)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        _mid, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    MODEL_USED = _mid
    print(f"Loaded: {_mid} in float16")

qwen_model.eval()
QWEN_DEVICE = next(qwen_model.parameters()).device
print(f"\nActive model : {MODEL_USED}")
free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
print(f"Free VRAM   : {free_vram / 1e9:.2f} GB\n")

# ── 12. Load CLIP on CPU ──────────────────────────────────────
print(f"Loading {CLIP_ID} on CPU...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model     = CLIPModel.from_pretrained(CLIP_ID).to(CLIP_DEVICE).eval()
print("All models ready.\n")

# ── 13. Evaluation loop ──────────────────────────────────────
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
        ie          = clip_img_emb(image)

        for p_name, p_text in PROMPT_TEXTS.items():
            inputs    = build_qwen_inputs(image, p_text)
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out_ids = qwen_model.generate(**inputs, max_new_tokens=96)

            gen = qwen_processor.batch_decode(
                out_ids[:, input_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()

            tok_gt  = [word_tokenize(c.lower()) for c in gt_captions]
            tok_gen = word_tokenize(gen.lower())
            meteor  = 0.0
            if tok_gt and tok_gen:
                try: meteor = meteor_score.meteor_score(tok_gt, tok_gen)
                except: pass

            chai, chs = calculate_chair(gen, gt_objects)
            te        = clip_txt_emb(gen)
            clipscore = (ie @ te.T).item() * 100.0

            results.append({
                "Image ID": img_id, "Prompt Variant": p_name,
                "Generated Text": gen, "METEOR": meteor,
                "CHAi": chai, "CHs": chs, "CLIPScore": clipscore,
            })

    finally:
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── 14. Save & print ─────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

summary = (df.groupby("Prompt Variant")[["METEOR","CHAi","CHs","CLIPScore"]]
             .mean().reset_index())
summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nModel used      : {MODEL_USED}")
print(f"Detailed results: {RESULTS_CSV}")
print(f"Summary table   : {SUMMARY_CSV}\n")

W = 87
print("=" * W)
print(f"  Advanced Metrics Summary -- {MODEL_USED}")
print("=" * W)
print(f"| {'Prompt Variant':<35} | {'METEOR':>8} | {'CHAi':>8} | {'CHs':>8} | {'CLIPScore':>9} |")
print(f"|{'-'*37}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|")
for _, r in summary.iterrows():
    print(f"| {r['Prompt Variant']:<35} | {r['METEOR']:>8.4f} | {r['CHAi']:>8.4f} | {r['CHs']:>8.4f} | {r['CLIPScore']:>9.4f} |")
print("=" * W + "\n")

# ── 15. Auto download buttons ────────────────────────────────
try:
    import base64
    from IPython.display import HTML, display
    def _dl_btn(fpath, fname):
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        display(HTML(
            f'<a download="{fname}" href="data:text/csv;base64,{b64}" '
            f'style="background:#4CAF50;color:white;padding:8px 16px;'
            f'border-radius:4px;text-decoration:none;font-weight:bold;margin:4px">'
            f'Download {fname}</a>'))
    _dl_btn(RESULTS_CSV, "qwen_advanced_metrics_results.csv")
    _dl_btn(SUMMARY_CSV, "qwen_advanced_metrics_summary.csv")
except Exception:
    pass
