# src/blip2_doodle_caption.py

import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def generate_doodle_caption(doodle_path):
    """
    Generate a text caption for a user doodle using the 6.7B BLIP-2 model.
    Returns the caption as a string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the 6.7B model from Hugging Face (or later switch to a local copy)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b").to(device)
    
    # Load and preprocess the doodle image
    image = Image.open(doodle_path).convert("RGB")
    image = image.resize((224, 224))
    
    # Provide an empty text prompt so the model knows to perform image-to-text generation
    inputs = processor(images=image, text=[""], return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doodle_path = os.path.join(base_dir, "user_sketches", "example_sketch.jpg")
    try:
        cap = generate_doodle_caption(doodle_path)
        print("Doodle caption:", cap)
    except Exception as e:
        print(f"Error generating doodle caption: {e}")

if __name__ == "__main__":
    main()
