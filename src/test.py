import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# Pick an image that previously failed
image_path = "dataset/Jamini roy Rama/jamini-roy-untitled-(drummer-with-santhals).jpg"

try:
    image = Image.open(image_path).convert("RGB")
    
    # 🔴 Resize the image to a safe shape (BLIP-2 expects 224x224)
    image = image.resize((224, 224))

    # 🔍 Check before passing to BLIP-2
    inputs = processor(images=image, return_tensors="pt").to(device)
    print(f"Processed Image Tensor Shape: {inputs['pixel_values'].shape}")

    caption_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    print(f"Generated Caption: {caption}")

except Exception as e:
    print(f"Error processing {image_path}: {e}")
