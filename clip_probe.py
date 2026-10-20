import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. Force execution on CPU
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Load the test image
img = Image.open("example.jpg").convert("RGB")

# 3. Define positive and distractor prompts for better context
prompts = ["a photo of a dog", "a photo of a cat", "a photo of a car"]

# 4. Preprocess inputs
inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True)

# 5. Perform inference without gradient tracking
with torch.no_grad():
    outputs = model(**inputs)

# Cosine Similarity (Normalized)
image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
cosine_sim = torch.cosine_similarity(image_embeds, text_embeds[0:1]).item()

# Softmax Probabilities across candidate prompts
probs = outputs.logits_per_image.softmax(dim=-1)

print(f"Target Label: {prompts[0]}")
print(f"Cosine Similarity Score: {cosine_sim:.4f}")
print(f"Confidence Probability: {probs[0][0].item() * 100:.2f}%")