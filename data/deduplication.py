import os
import argparse
import hashlib
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil

# ================
# Helper Functions
# ================

def compute_pca_hash(image):
    """
    Returns a PCA hash string of a grayscale image.
    Uses the U matrix from SVD to identify structure.
    """
    image = image.resize((32, 32)).convert("L")  # grayscale
    arr = np.array(image).astype(np.float32)
    arr -= arr.mean()
    u, s, vh = np.linalg.svd(arr, full_matrices=False)
    hash_str = hashlib.md5(u[:, :8].tobytes()).hexdigest()
    return hash_str

def load_images(folder):
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def deduplicate(image_paths):
    hash_set = set()
    unique_paths = []
    for path in tqdm(image_paths, desc="🔍 Deduplicating"):
        try:
            img = Image.open(path).convert("RGB")
            h = compute_pca_hash(img)
            if h not in hash_set:
                hash_set.add(h)
                unique_paths.append(path)
        except Exception:
            continue
    return unique_paths

def save_images(image_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for path in tqdm(image_paths, desc="💾 Saving unique images"):
        filename = os.path.basename(path)
        shutil.copy2(path, os.path.join(output_folder, filename))

# ============
# Entry Point
# ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder with images to deduplicate")
    parser.add_argument("--output_folder", type=str, default="./deduplicated", help="Where to save unique images")
    args = parser.parse_args()

    print("📂 Loading images...")
    image_paths = load_images(args.input_folder)

    print("🔎 Deduplicating using PCA hash...")
    unique_paths = deduplicate(image_paths)

    print(f"✅ {len(unique_paths)} unique images found (out of {len(image_paths)})")

    print("📁 Saving deduplicated images...")
    save_images(unique_paths, args.output_folder)

    print("🎉 Done.")
