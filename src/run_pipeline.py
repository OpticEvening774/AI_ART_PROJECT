# src/run_pipeline.py

import os
from .blip2_doodle_caption import generate_doodle_caption
from .text_matching import match_doodle_to_paintings

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use the existing painting captions file
    descriptions_file = os.path.join(base_dir, "descriptions.csv")
    print(f"Using painting captions from: {descriptions_file}")

    # Generate doodle caption using BLIP-2 6.7B
    doodle_path = os.path.join(base_dir, "user_sketches", "example_sketch.jpg")
    doodle_caption = generate_doodle_caption(doodle_path)
    print("Doodle caption:", doodle_caption)

    # Perform text-based matching
    print("Matching doodle to paintings...")
    top_matches = match_doodle_to_paintings(descriptions_file, doodle_caption, top_k=3)

    print("\n=== TOP 3 MATCHES ===")
    for i, (filename, painting_caption, score) in enumerate(top_matches, start=1):
        print(f"{i}. {filename} (score={score:.4f}) => {painting_caption}")

if __name__ == "__main__":
    main()
