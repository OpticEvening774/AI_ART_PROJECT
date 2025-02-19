import os
import csv
import torch
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    descriptions_file = os.path.join(base_dir, "descriptions.csv")

    # Gather all images
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, dataset_dir)
                image_paths.append((img_path, rel_path))

    total_images = len(image_paths)
    print(f"Found {total_images} images in '{dataset_dir}'. Starting caption generation...")

    processed_count = 0
    skipped_count = 0

    # Create or overwrite the descriptions.csv
    with open(descriptions_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "description"])

        for (img_path, rel_path) in tqdm(image_paths, desc="Generating Captions", ncols=80):
            try:
                # 1. Load & resize image
                image = Image.open(img_path).convert("RGB")
                image = image.resize((224, 224))

                # 2. Processor call with image + empty text
                inputs = processor(images=image, text=[""], return_tensors="pt").to(device)

                # 3. Generate caption
                caption_ids = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(caption_ids[0], skip_special_tokens=True)

                writer.writerow([rel_path, caption])
                processed_count += 1

            except Exception as e:
                skipped_count += 1
                print(f"❌ Skipping {rel_path} due to error: {e}")

    print("\n==== Caption Generation Completed ====")
    print(f"Total Images Found: {total_images}")
    print(f"Successfully Processed: {processed_count}")
    print(f"Skipped Due to Errors: {skipped_count}")
    print(f"Captions saved to: {descriptions_file}")

if __name__ == "__main__":
    main()
