import os
import json
import csv
import argparse
import urllib.request
import zipfile
from math import ceil

def download_and_extract(url, dest_zip, extract_to):
    if not os.path.exists(dest_zip):
        print(f"Downloading from {url} to {dest_zip}...")
        urllib.request.urlretrieve(url, dest_zip)
        print("Download complete.")
    else:
        print(f"{dest_zip} already exists, skipping download.")

    print(f"Extracting {dest_zip} to {extract_to}...")
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_flickr8k_dataset(base_dir="flickr8k"):
    """
    Downloads and extracts the Flickr8k images and captions.
    Returns:
        annotations_json: path to the generated COCO-style annotations JSON file.
        img_dir: path to the folder containing the images.
    """
    os.makedirs(base_dir, exist_ok=True)

    # URLs for the Flickr8k dataset (images and text)
    images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url   = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

    images_zip = os.path.join(base_dir, "Flickr8k_Dataset.zip")
    # text_zip   = os.path.join(base_dir, "Flickr8k_text", "Flickr8k_text.zip")
    text_zip   = os.path.join(base_dir, "Flickr8k_text.zip")

    download_and_extract(images_url, images_zip, base_dir)
    download_and_extract(text_url, text_zip, base_dir)

    # After extraction, the images are in a folder named "Flickr8k_Dataset"
    img_dir = os.path.join(base_dir, "Flicker8k_Dataset")
    # The captions file is inside "Flickr8k_text"
    tokens_path = os.path.join(base_dir, "Flickr8k.token.txt")

    # Create a COCO-style JSON from the tokens file.
    # Format: each line in tokens file is "image_name#idx caption"
    images = {}
    annotations = []
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            # Some versions may separate with a space; if not found try splitting by space.
            if len(parts) < 2:
                parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            token, caption = parts[0], parts[1]
            if "#" not in token:
                continue
            image_file, cap_idx = token.split("#")
            # We take only the first caption (#0) per image.
            if cap_idx != "0":
                continue
            if image_file not in images:
                images[image_file] = len(images) + 1  # assign a unique id
            annotations.append({
                "image_id": images[image_file],
                "caption": caption
            })

    # Build the images list for JSON.
    images_list = [{"id": id, "file_name": file_name} for file_name, id in images.items()]
    coco_json = {"images": images_list, "annotations": annotations}

    annotations_json = os.path.join(base_dir, "annotations.json")
    with open(annotations_json, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, indent=4)

    print(f"COCO-style annotations saved to {annotations_json}")
    return annotations_json, img_dir

def prepare_dataset(input_json, img_dir, output_dir, batch_size=200):
    """
    Reads a COCO-style JSON file containing image info and captions.
    Splits the data into CSV files with at most `batch_size` rows each.
    Each CSV row has columns: image_path and prompt.
    Only includes images that actually exist on disk.
    """
    with open(input_json, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Map image id to file name.
    image_id_to_filename = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # Map image id to caption (we assume one caption per image here)
    image_id_to_caption = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_id_to_caption:
            image_id_to_caption[img_id] = ann["caption"]

    rows = []
    missing_count = 0
    valid_count = 0

    for img_id, file_name in image_id_to_filename.items():
        if img_id in image_id_to_caption:
            full_path = os.path.join(img_dir, file_name)

            # Check if the image file actually exists
            if os.path.isfile(full_path):
                caption = image_id_to_caption[img_id]
                rows.append({"image_path": full_path, "prompt": caption})
                valid_count += 1
            else:
                missing_count += 1

    print(f"Found {valid_count} valid images with captions, skipped {missing_count} missing images.")

    os.makedirs(output_dir, exist_ok=True)
    total_entries = len(rows)
    num_batches = ceil(total_entries / batch_size)
    print(f"Total entries: {total_entries}, creating {num_batches} CSV batch file(s) with up to {batch_size} entries each.")

    for batch_idx in range(num_batches):
        batch_rows = rows[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        output_file = os.path.join(output_dir, f"batch_{batch_idx + 1}.csv")
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["image_path", "prompt"])
            writer.writeheader()
            writer.writerows(batch_rows)
        print(f"Wrote {len(batch_rows)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare an image-to-text dataset into CSV batches for training. "
                    "If no input is provided, the script downloads the Flickr8k dataset."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the input COCO-style JSON file.")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, default="csv_batches", help="Directory to save CSV batch files.")
    parser.add_argument("--batch_size", type=int, default=200, help="Max number of entries per CSV file (default: 200)")

    args = parser.parse_args()

    # If no input JSON or image directory is provided, download the Flickr8k dataset.
    if args.input is None or args.img_dir is None:
        print("No dataset provided. Downloading the Flickr8k dataset...")
        annotations_json, img_dir = download_flickr8k_dataset()
        input_json = annotations_json
    else:
        input_json = args.input
        img_dir = args.img_dir

    prepare_dataset(input_json=input_json, img_dir=img_dir, output_dir=args.output_dir, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
