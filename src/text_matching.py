# src/text_matching.py

import os
import csv
from sentence_transformers import SentenceTransformer, util
from blip2_doodle_caption import generate_doodle_caption
from post_processing import refine_matches

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    descriptions_file = os.path.join(base_dir, "descriptions.csv")
    doodle_path = os.path.join(base_dir, "user_sketches", "example_sketch.jpg")

    # 1. Load painting descriptions from CSV
    painting_descriptions = []
    with open(descriptions_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        # first line is header
        for line in lines[1:]:
            if not line.strip():
                continue
            filename, desc = line.split(",", 1)
            painting_descriptions.append((filename.strip(), desc.strip()))

    # 2. Generate user doodle caption using BLIP-2
    try:
        doodle_caption = generate_doodle_caption(doodle_path)
        print("Doodle caption:", doodle_caption)
    except Exception as e:
        print(f"Error generating doodle caption: {e}")
        return

    # 3. Load Sentence-BERT for text similarity
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_model = SentenceTransformer(model_name)

    # 4. Compute embedding for doodle caption
    doodle_emb = sbert_model.encode(doodle_caption, convert_to_tensor=True)

    # 5. Compute embeddings for painting captions + similarity
    results = []
    for filename, desc in painting_descriptions:
        painting_emb = sbert_model.encode(desc, convert_to_tensor=True)
        # cos_sim returns a 2D matrix if we pass both as tensor
        score = util.cos_sim(doodle_emb, painting_emb).item()
        results.append((filename, desc, score))

    # 6. Refine + sort matches
    # Sort descending by score, optionally discard if < threshold
    results = sorted(results, key=lambda x: x[2], reverse=True)
    refined = refine_matches(results, threshold=0.2)

    # 7. Print top 3 matches
    print("\nTop 3 Matches:")
    for i, (filename, desc, score) in enumerate(refined[:3]):
        print(f"{i+1}. {filename} | score={score:.4f}\n   => {desc}")

if __name__ == "__main__":
    main()
