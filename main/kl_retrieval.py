import os
import json
import numpy as np
from tqdm import tqdm

# --- 1. Pixel-wise KL divergence function ---
def kl_map(P, Q, eps=1e-12):
    """
    Compute pixel-wise KL divergence map between distributions P and Q.
    """
    return P * np.log((P + eps) / (Q + eps))

# --- 2. Configuration ---
P_DIR        = 'data/iu_xray/heatmaps/agcam_kl/P'     # Test heatmaps directory
Q_DIR        = 'data/iu_xray/heatmaps/agcam_kl/Q'     # Train heatmaps directory
MATCHES_FILE = 'data/iu_xray/iu_matches_top30.json'   # Matches JSON file
OUTPUT_FILE  = 'data/iu_xray/iu_kl_top10.json'         # Output JSON file
TOPK         = 10                                      # Number of top train IDs to keep

# --- 3. Load matches ---
with open(MATCHES_FILE, 'r', encoding='utf-8') as f:
    matches = json.load(f)

output_list = []

# --- 4. Process exact_matches: compute KL where available, else fallback ---
for test_id, train_ids in tqdm(matches.get('exact_matches', {}).items(), desc='Exact matches'):
    # Gather test heatmaps
    p_files = [f for f in os.listdir(P_DIR) if f.startswith(test_id + '_') and f.endswith('.npy')]
    if not p_files:
        topk_ids = train_ids[:TOPK]
    else:
        # Load and average all P maps for this test
        P_list = [np.load(os.path.join(P_DIR, fname)) for fname in p_files]
        P = np.mean(np.stack(P_list, axis=0), axis=0)
        scores = []
        for tr_id in train_ids:
            # Gather train heatmaps
            q_files = [f for f in os.listdir(Q_DIR) if f.startswith(tr_id + '_') and f.endswith('.npy')]
            if not q_files:
                continue
            Q_list = [np.load(os.path.join(Q_DIR, fname)) for fname in q_files]
            Q = np.mean(np.stack(Q_list, axis=0), axis=0)
            # Compute mean KL
            mean_kl = float(np.mean(kl_map(P, Q)))
            scores.append((tr_id, mean_kl))
        if not scores:
            topk_ids = train_ids[:TOPK]
        else:
            # Sort by KL ascending
            scores.sort(key=lambda x: x[1])
            sorted_ids = [tid for tid, _ in scores]
            if len(sorted_ids) >= TOPK:
                topk_ids = sorted_ids[:TOPK]
            else:
                needed = TOPK - len(sorted_ids)
                extras = [tid for tid in train_ids if tid not in sorted_ids][:needed]
                topk_ids = sorted_ids + extras
    # Prepare entry
    test_prefix = test_id.split('_')[0]
    train_prefixes = [tid.split('_')[0] for tid in topk_ids]
    output_list.append({
        "testid": test_prefix,
        "trainids": train_prefixes
    })

# --- 5. Process partial_matches: take first TOPK ---
for test_id, train_ids in tqdm(matches.get('partial_matches', {}).items(), desc='Partial matches'):
    topk_ids = train_ids[:TOPK]
    test_prefix = test_id.split('_')[0]
    train_prefixes = [tid.split('_')[0] for tid in topk_ids]
    output_list.append({
        "testid": test_prefix,
        "trainids": train_prefixes
    })

# --- 6. Write output ---
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=2, ensure_ascii=False)

print(f"Saved Top-{TOPK} KL results to {OUTPUT_FILE}")