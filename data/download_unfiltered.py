import os
import csv
import argparse
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import hashlib

# ================
# Helper Functions
# ================

def sanitize_filename(url):
    """Create a hash-based filename from a URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"

def download_image(url, timeout=5):
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
    except Exception:
        pass
    return None

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="JPEG", quality=95)

# ============
# Main Script
# ============

def main(url_list_path, output_folder, metadata_path, max_images=None):
    with open(url_list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    if max_images:
        urls = urls[:max_images]

    os.makedirs(output_folder, exist_ok=True)

    with open(metadata_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["url", "filename", "status"])
        writer.writeheader()

        for url in tqdm(urls, desc="Downloading images"):
            filename = sanitize_filename(url)
            out_path = os.path.join(output_folder, filename)

            if os.path.exists(out_path):
                writer.writerow({"url": url, "filename": filename, "status": "already_downloaded"})
                continue

            img = download_image(url)
            if img is not None:
                try:
                    save_image(img, out_path)
                    writer.writerow({"url": url, "filename": filename, "status": "success"})
                except Exception:
                    writer.writerow({"url": url, "filename": filename, "status": "save_failed"})
            else:
                writer.writerow({"url": url, "filename": filename, "status": "download_failed"})

    print(f"✅ Finished. Metadata saved to {metadata_path}")

# ===============
# Entry Point CLI
# ===============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_list", type=str, required=True, help="Path to a text file with image URLs")
    parser.add_argument("--output_folder", type=str, default="./data/unfiltered", help="Where to save downloaded images")
    parser.add_argument("--metadata_csv", type=str, default="./data/unfiltered_metadata.csv", help="Where to log metadata")
    parser.add_argument("--max_images", type=int, default=None, help="Optional max number of images to download")
    args = parser.parse_args()

    main(args.url_list, args.output_folder, args.metadata_csv, args.max_images)
