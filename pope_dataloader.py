import os
import json
import zipfile
import urllib.request
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class POPEDataset(Dataset):
    """
    PyTorch Dataset for the POPE (Polling-based Object Probing Evaluation) Benchmark.
    Downloads COCO val2014 images and reads the POPE json lines.
    """
    def __init__(self, pope_json_path, coco_val_dir, transform=None, download=False):
        self.pope_json_path = pope_json_path
        self.coco_val_dir = coco_val_dir
        self.transform = transform
        
        if download:
            self.download_coco()

        if not os.path.exists(self.coco_val_dir):
            raise FileNotFoundError(f"COCO images not found at {self.coco_val_dir}. Set download=True to download them.")

        self.data = []
        with open(self.pope_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def download_coco(self):
        if os.path.exists(self.coco_val_dir) and len(os.listdir(self.coco_val_dir)) > 0:
            print("COCO val2014 directory already exists. Skipping download.")
            return

        os.makedirs(self.coco_val_dir, exist_ok=True)
        url = "http://images.cocodataset.org/zips/val2014.zip"
        zip_path = os.path.join(os.path.dirname(self.coco_val_dir), "val2014.zip")
        
        if not os.path.exists(zip_path):
            print(f"Downloading COCO val2014 images from {url}...")
            # We'll use urllib to download with a progress bar
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="val2014.zip") as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        
        print("Extracting COCO val2014 images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We assume it extracts to a folder 'val2014'
            zip_ref.extractall(os.path.dirname(self.coco_val_dir))
            
        print("Extraction complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['image']
        question_id = item['question_id']
        prompt = item['text']
        label = item['label']

        image_path = os.path.join(self.coco_val_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Could not open image {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        # Map string label "yes"/"no" to binary integer
        binary_label = 1 if label.lower() == "yes" else 0

        return {
            "question_id": question_id,
            "image": image,
            "prompt": prompt,
            "label": binary_label,
            "label_str": label
        }

def get_pope_dataloader(pope_json_path, coco_val_dir, batch_size=32, shuffle=False, num_workers=4, image_size=224, download=False):
    """
    Creates a DataLoader for the POPE dataset.
    """
    # Standard transform for Vision Language Models (like CLIP, Llava, etc.)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    dataset = POPEDataset(
        pope_json_path=pope_json_path,
        coco_val_dir=coco_val_dir,
        transform=transform,
        download=download
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    # Example usage:
    # Set paths according to your environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pope_file = os.path.join(current_dir, "POPE", "output", "coco", "coco_pope_random.json")
    coco_dir = r"D:\datasets\COCO\val2014"

    print(f"Initializing DataLoader with POPE JSON: {pope_file}")
    
    # download=True will download COCO val2014 (~6GB) if not already present
    dataloader = get_pope_dataloader(
        pope_json_path=pope_file,
        coco_val_dir=coco_dir,
        batch_size=4,
        download=True
    )

    print(f"Total samples: {len(dataloader.dataset)}")
    
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Images shape:", batch["image"].shape)
        print("Prompts:", batch["prompt"])
        print("Labels (str):", batch["label_str"])
        print("Labels (int):", batch["label"])
        break
