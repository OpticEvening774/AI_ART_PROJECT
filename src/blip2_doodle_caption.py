# src/blip2_doodle_caption.py

import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def generate_doodle_caption(doodle_path):
    """
    Generate a BLIP-2 caption for a user doodle.
    Returns the caption as a string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    
    image = Image.open(doodle_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

def main():
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doodle_path = os.path.join(base_dir, "user_sketches", "example_sketch.jpg")
    
    try:
        doodle_caption = generate_doodle_caption(doodle_path)
        print(f"Doodle caption: {doodle_caption}")
    except Exception as e:
        print(f"Error processing user doodle: {e}")

if __name__ == "__main__":
    main()
