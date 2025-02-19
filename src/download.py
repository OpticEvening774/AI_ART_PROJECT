from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1. Download and cache the processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
processor.save_pretrained("blip2_opt_6.7b_processor")

# 2. Download and cache the model weights
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b")
model.save_pretrained("blip2_opt_6.7b_model")
