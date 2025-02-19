# src/post_processing.py

def refine_matches(results, threshold=0.2):
    """
    Optional function to refine matches by discarding those with a score < threshold.
    results: list of tuples (filename, description, score)
    Returns a sorted (descending by score) refined list.
    """
    # Filter out matches below threshold
    refined = [(f, d, s) for (f, d, s) in results if s >= threshold]
    if not refined:
        # fallback if everything is below threshold
        refined = results
    return sorted(refined, key=lambda x: x[2], reverse=True)
