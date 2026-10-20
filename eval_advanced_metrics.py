# No use of this
import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["TORCH_HOME"] = r"D:\torch_home"
os.environ["TMPDIR"] = r"D:\tmp"
os.environ["TEMP"] = r"D:\tmp"
os.environ["TMP"] = r"D:\tmp"
os.makedirs(r"D:\tmp", exist_ok=True)
os.makedirs(r"D:\torch_home", exist_ok=True)

import json
import csv
import torch
import requests
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import zipfile

# Download NLTK data automatically to D drive
nltk_dir = r"D:\nltk_data"
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.append(nltk_dir)
nltk.download("punkt", download_dir=nltk_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_dir, quiet=True)
nltk.download("wordnet", download_dir=nltk_dir, quiet=True)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_dir, quiet=True)
nltk.download("averaged_perceptron_tagger_eng", download_dir=nltk_dir, quiet=True)

IMAGE_DIR = r"D:\datasets\COCO\val2014"
ANNO_DIR = r"D:\datasets\COCO\annotations"
MAX_SAMPLES = 300
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLIP_ID = "openai/clip-vit-base-patch32"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

anno_zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
anno_zip_path = os.path.join(ANNO_DIR, "annotations_trainval2014.zip")
instances_path = os.path.join(ANNO_DIR, "annotations", "instances_val2014.json")
captions_path = os.path.join(ANNO_DIR, "annotations", "captions_val2014.json")

if not os.path.exists(instances_path) or not os.path.exists(captions_path):
    download_file(anno_zip_url, anno_zip_path)
    print("Extracting annotations...")
    with zipfile.ZipFile(anno_zip_path, "r") as zip_ref:
        zip_ref.extractall(ANNO_DIR)

print("Loading COCO annotations...")
with open(instances_path, "r") as f:
    instances_data = json.load(f)
with open(captions_path, "r") as f:
    captions_data = json.load(f)

# Build COCO category mapping
coco_categories = {cat["id"]: cat["name"].lower() for cat in instances_data["categories"]}
# Very basic synonym mapping for CHAIR
coco_synonyms = {
    "person": ["man", "woman", "boy", "girl", "people", "child", "human"],
    "bicycle": ["bike"],
    "car": ["vehicle", "automobile"],
    "motorcycle": ["bike"],
    "airplane": ["plane", "aircraft"],
    "boat": ["ship"],
    "traffic light": ["stoplight"],
    "fire hydrant": ["hydrant"]
}
word_to_coco = {}
for cat_id, cat_name in coco_categories.items():
    word_to_coco[cat_name] = cat_name
    if cat_name in coco_synonyms:
        for syn in coco_synonyms[cat_name]:
            word_to_coco[syn] = cat_name

img_to_caps = {}
for ann in captions_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in img_to_caps: img_to_caps[img_id] = []
    img_to_caps[img_id].append(ann["caption"])

img_to_objects = {}
for ann in instances_data["annotations"]:
    img_id = ann["image_id"]
    cat_name = coco_categories[ann["category_id"]]
    if img_id not in img_to_objects: img_to_objects[img_id] = set()
    img_to_objects[img_id].add(cat_name)

images_info = captions_data["images"][:MAX_SAMPLES]

def extract_nouns(text):
    tokens = word_tokenize(text.lower())
    tags = nltk.pos_tag(tokens)
    return [word for word, pos in tags if pos.startswith("NN")]

def calculate_chair(text, ground_truth_objects):
    sentences = sent_tokenize(text)
    total_obj_mentions = 0
    hallucinated_obj_mentions = 0
    hallucinated_sentences = 0
    
    for sent in sentences:
        nouns = extract_nouns(sent)
        sent_has_hallucination = False
        
        coco_nouns_in_sent = [word_to_coco[n] for n in nouns if n in word_to_coco]
                
        for obj in coco_nouns_in_sent:
            total_obj_mentions += 1
            if obj not in ground_truth_objects:
                hallucinated_obj_mentions += 1
                sent_has_hallucination = True
                
        if sent_has_hallucination:
            hallucinated_sentences += 1
            
    chai = (hallucinated_obj_mentions / total_obj_mentions) if total_obj_mentions > 0 else 0.0
    chs = (hallucinated_sentences / len(sentences)) if len(sentences) > 0 else 0.0
    return chai, chs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        offload_folder=r"D:\offload"
    )
    
    print(f"Loading CLIP {CLIP_ID}...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device)

    prompts = {
        "Prompt A (Baseline)": "USER: <image>\nDescribe what is happening in this image.\nASSISTANT:",
        "Prompt B (Forced Hallucination)": "USER: <image>\nDescribe what is happening in this image in extreme detail. Include all background objects, fine elements, and contextual items.\nASSISTANT:",
        "Prompt C (Constrained)": "USER: <image>\nDescribe what is happening in this image. Only mention objects and details you are completely certain are present. Do not guess.\nASSISTANT:"
    }

    results = []

    for info in tqdm(images_info, desc="Evaluating Images"):
        img_id = info["id"]
        file_name = info["file_name"]
        image_url = info["coco_url"]
        local_path = os.path.join(IMAGE_DIR, file_name)
        
        download_file(image_url, local_path)
        
        try:
            image = Image.open(local_path).convert("RGB")
            gt_captions = img_to_caps.get(img_id, [])
            gt_objects = img_to_objects.get(img_id, set())
            
            # Precompute CLIP embedding
            clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                img_embeds = clip_model.get_image_features(**clip_inputs)
                if not isinstance(img_embeds, torch.Tensor):
                    img_embeds = img_embeds[0]
                img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

            for p_name, p_text in prompts.items():
                inputs = llava_processor(text=p_text, images=image, return_tensors="pt").to(device)
                
                if dtype == torch.float16:
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].half()
                elif dtype == torch.float32:
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].float()
                
                with torch.inference_mode():
                    out_ids = llava_model.generate(**inputs, max_new_tokens=128)
                
                gen_text = llava_processor.decode(out_ids[0], skip_special_tokens=True)
                if "ASSISTANT:" in gen_text:
                    gen_text = gen_text.split("ASSISTANT:")[-1].strip()
                
                tokenized_gt = [word_tokenize(c.lower()) for c in gt_captions]
                tokenized_gen = word_tokenize(gen_text.lower())
                
                meteor = 0.0
                if tokenized_gt and tokenized_gen:
                    try:
                        meteor = meteor_score.meteor_score(tokenized_gt, tokenized_gen)
                    except Exception:
                        pass
                
                chai, chs = calculate_chair(gen_text, gt_objects)
                
                clip_text = clip_processor(text=[gen_text], return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    txt_embeds = clip_model.get_text_features(**clip_text)
                    if not isinstance(txt_embeds, torch.Tensor):
                        txt_embeds = txt_embeds[0]
                    txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
                
                clipscore = (img_embeds @ txt_embeds.T).item() * 2.5
                
                results.append({
                    "Image ID": img_id,
                    "Prompt Variant": p_name,
                    "Generated Text": gen_text,
                    "METEOR": meteor,
                    "CHAi": chai,
                    "CHs": chs,
                    "CLIPScore": clipscore
                })
        finally:
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except: pass
                
    df = pd.DataFrame(results)
    df.to_csv("advanced_metrics_results.csv", index=False)
    
    summary = df.groupby("Prompt Variant")[["METEOR", "CHAi", "CHs", "CLIPScore"]].mean().reset_index()
    summary.to_csv("advanced_metrics_summary.csv", index=False)
    
    print("\n" + "="*85)
    print("  Advanced Metrics Summary (llava-1.5-7b-hf)")
    print("="*85)
    print(f"| {'Prompt Variant':<35} | {'METEOR':<8} | {'CHAi':<8} | {'CHs':<8} | {'CLIPScore':<9} |")
    print(f"|{'-'*37}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|")
    for _, r in summary.iterrows():
        print(f"| {r['Prompt Variant']:<35} | {r['METEOR']:.4f}   | {r['CHAi']:.4f}   | {r['CHs']:.4f}   | {r['CLIPScore']:.4f}    |")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
