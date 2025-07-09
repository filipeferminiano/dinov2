import os
import shutil
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import timm
import faiss
import hashlib

# ========================
# Configurable Parameters
# ========================
MIN_SIZE = 128
NSFW_PLACEHOLDER = False  # Replace with real NSFW checker

# ================
# Helper Functions
# ================

def is_nsfw(image):
    # Placeholder: Always returns False (not NSFW)
    return NSFW_PLACEHOLDER

def is_valid_image(path):
    try:
        img = Image.open(path).convert("RGB")
        if img.size[0] < MIN_SIZE or img.size[1] < MIN_SIZE:
            return False
        return not is_nsfw(img)
    except Exception:
        return False

def compute_pca_hash(image):
    """Returns a PCA hash of an image as a string."""
    image = image.resize((32, 32)).convert("L")
    arr = np.array(image).astype(np.float32)
    arr = arr - arr.mean()
    u, s, vh = np.linalg.svd(arr, full_matrices=False)
    return hashlib.md5(u[:, :8].tobytes()).hexdigest()

def deduplicate_images(image_paths):
    hash_set = set()
    unique_images = []
    for path in tqdm(image_paths, desc="Deduplicating"):
        try:
            img = Image.open(path).convert("RGB")
            h = compute_pca_hash(img)
            if h not in hash_set:
                hash_set.add(h)
                unique_images.append(path)
        except Exception:
            continue
    return unique_images

def load_images_from_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def build_image_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def compute_embeddings(image_paths, model, transform, device):
    model.eval()
    embeddings = []
    valid_paths = []

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Embedding images"):
            try:
                image = Image.open(path).convert("RGB")
                x = transform(image).unsqueeze(0).to(device)
                feat = model(x)
                embeddings.append(feat.cpu().squeeze(0).numpy())
                valid_paths.append(path)
            except Exception:
                continue

    return np.vstack(embeddings), valid_paths

def retrieve_top_k(curated_feats, uncurated_feats, uncurated_paths, k=4):
    index = faiss.IndexFlatIP(curated_feats.shape[1])
    faiss.normalize_L2(uncurated_feats)
    faiss.normalize_L2(curated_feats)
    index.add(uncurated_feats)
    _, I = index.search(curated_feats, k)
    retrieved_paths = set()
    for indices in I:
        for idx in indices:
            retrieved_paths.add(uncurated_paths[idx])
    return list(retrieved_paths)

def copy_selected_images(image_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for path in tqdm(image_paths, desc="Copying curated images"):
        fname = os.path.basename(path)
        shutil.copy2(path, os.path.join(output_folder, fname))

# ===========
# Entry Point
# ===========

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated_folder", type=str, required=True, help="Path to curated images (e.g., ImageNet)")
    parser.add_argument("--uncurated_folder", type=str, required=True, help="Path to uncurated web images")
    parser.add_argument("--output_folder", type=str, default="./data/curated_output", help="Where to save curated results")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔍 Loading ViT model...")
    model = timm.create_model("vit_base_patch14_dino", pretrained=True, num_classes=0)
    model = model.to(device)

    transform = build_image_transform(args.image_size)

    print("📂 Loading and filtering curated images...")
    curated_paths = [p for p in load_images_from_folder(args.curated_folder) if is_valid_image(p)]
    curated_paths = deduplicate_images(curated_paths)

    print("📂 Loading and filtering uncurated images...")
    uncurated_paths = [p for p in load_images_from_folder(args.uncurated_folder) if is_valid_image(p)]
    uncurated_paths = deduplicate_images(uncurated_paths)

    print("💡 Computing image embeddings...")
    curated_feats, curated_paths = compute_embeddings(curated_paths, model, transform, device)
    uncurated_feats, uncurated_paths = compute_embeddings(uncurated_paths, model, transform, device)

    print("📎 Retrieving top-k similar uncurated images...")
    selected_paths = retrieve_top_k(curated_feats, uncurated_feats, uncurated_paths, k=args.top_k)

    print(f"✅ Saving curated dataset to {args.output_folder}")
    copy_selected_images(selected_paths, args.output_folder)

    print("🎉 Done.")
