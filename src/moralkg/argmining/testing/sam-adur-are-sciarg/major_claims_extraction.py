import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import sys
import os

def load_own_claims(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    own_claims = []
    for doc in data:
        for a in doc.get("adus", []):
            if a.get("label") == "claim" and a.get("original_label") == "own_claim":
                own_claims.append(a)
    # Remove duplicates by text
    seen = set()
    unique_claims = []
    for a in own_claims:
        text = a["text"]
        if text not in seen:
            seen.add(text)
            unique_claims.append(a)
    return unique_claims

def get_top_percent_indices(scores, percent):
    k = max(1, int(percent * len(scores)))
    return np.argsort(scores)[:k]

def write_markdown_table(headers, rows, md_lines):
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

def main():
    if len(sys.argv) != 2:
        print("Usage: python major_claim_extraction.py <PAPER_CODE>")
        sys.exit(1)
    code = sys.argv[1]
    json_path = f"results/{code}/{code}_cleaned.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        sys.exit(1)
    own_claims = load_own_claims(json_path)
    if not own_claims:
        print("No own_claims found.")
        sys.exit(0)
    scores = np.array([a["score"] for a in own_claims])
    texts = [a["text"] for a in own_claims]
    # Top 3% by lowest score
    idxs = get_top_percent_indices(scores, 0.03)
    selected_texts = [texts[i] for i in idxs]
    selected_scores = [scores[i] for i in idxs]
    # Embed
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embeddings = model.encode(selected_texts, convert_to_tensor=True, show_progress_bar=True)
    centroid = embeddings.mean(dim=0, keepdim=True)
    sims_to_centroid = util.cos_sim(embeddings, centroid).squeeze().cpu().numpy()
    # Sort by centroid similarity descending
    order = np.argsort(-sims_to_centroid)
    # Write markdown table
    md_lines = [f"# Top 3% own_claims for {code} (sorted by centroid similarity)\n"]
    headers = ["Rank", "Centroid", "Score", "Text"]
    rows = [
        [str(i+1), f"{sims_to_centroid[j]:.4f}", f"{selected_scores[j]:.4g}", selected_texts[j]]
        for i, j in enumerate(order)
    ]
    write_markdown_table(headers, rows, md_lines)
    out_path = f"results/{code}/major_claims.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main() 