# src/text_matching.py

import os
import csv
from sentence_transformers import SentenceTransformer, util

def match_doodle_to_paintings(descriptions_file, doodle_caption, top_k=3):
    """
    Uses a local Sentence-BERT model to compare the doodle caption with painting captions.
    Returns the top_k matches as a list of tuples (filename, caption, score).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.join(base_dir, "model", "all-minilm-model")
    
    # Load the local Sentence-BERT model
    sbert_model = SentenceTransformer(local_model_path)

    # Read painting captions from descriptions.csv
    painting_descriptions = []
    with open(descriptions_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        # Skip header
        for line in lines[1:]:
            if not line.strip():
                continue
            # Assumes CSV is simple: filename,description
            filename, desc = line.split(",", 1)
            painting_descriptions.append((filename.strip(), desc.strip()))

    # Embed the doodle caption
    doodle_emb = sbert_model.encode(doodle_caption, convert_to_tensor=True)

    # Compare with each painting caption
    results = []
    for filename, painting_caption in painting_descriptions:
        painting_emb = sbert_model.encode(painting_caption, convert_to_tensor=True)
        score = util.cos_sim(doodle_emb, painting_emb).item()
        results.append((filename, painting_caption, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    descriptions_file = os.path.join(base_dir, "descriptions.csv")
    doodle_caption = "An abstract doodle depicting a peace sign and a heart"
    top_matches = match_doodle_to_paintings(descriptions_file, doodle_caption, top_k=3)
    print("Top matches:", top_matches)

if __name__ == "__main__":
    main()
