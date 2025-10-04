import os
import json
import torch
import numpy as np
from tqdm import tqdm

# ------------------ Load matching pairs ------------------
def load_matches(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ------------------ Load image features ------------------
def load_image_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    ids_arr = data['ids']
    feats_arr = data['features']
    return {img_id: feats_arr[i] for i, img_id in enumerate(ids_arr)}

# ------------------ Collect all suffix features ------------------
def gather_feats(base_id, feats_dict):
    matched = []
    prefix = base_id + '-'
    for full_id, feat in feats_dict.items():
        if full_id == base_id or full_id.startswith(prefix):
            matched.append(feat)
    if not matched:
        return None
    return np.stack(matched, axis=0)

# ------------------ GPU cosine similarity ------------------
def compute_cosine_similarity_gpu(f1, f2):
    if f1 is None or f2 is None:
        return None
    t1 = torch.from_numpy(f1).float().cuda()
    t2 = torch.from_numpy(f2).float().cuda()
    if t1.ndimension() == 1: t1 = t1.unsqueeze(0)
    if t2.ndimension() == 1: t2 = t2.unsqueeze(0)
    t1 = t1 / t1.norm(dim=1, keepdim=True)
    t2 = t2 / t2.norm(dim=1, keepdim=True)
    return torch.mm(t1, t2.t()).cpu().numpy()

# ------------------ Sorting function ------------------
def sort_train_ids(test_id, train_ids, feats_dict, topk=30):
    test_feats = gather_feats(test_id, feats_dict)
    scores = []
    for tr_id in train_ids:
        tr_feats = gather_feats(tr_id, feats_dict)
        sim_mat = compute_cosine_similarity_gpu(test_feats, tr_feats)
        if sim_mat is None:
            continue

        max_sim = float(np.max(sim_mat))
        scores.append((tr_id, max_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in scores[:topk]]

# ------------------ Main program ------------------
def main():
    matches_file  = 'data/iu_xray/iu_matches.json'
    features_file = 'data/iu_xray/iu_all_image_features.npz'
    output_file   = 'data/iu_xray/iu_matches_top30.json'

    matches = load_matches(matches_file)
    feats   = load_image_features(features_file)

    output_list = []

    for category in ('exact_matches', 'partial_matches'):
        for test_id, train_ids in tqdm(
            matches.get(category, {}).items(),
            desc=f"Processing {category}"
        ):
            if category == 'exact_matches':
                top5 = sort_train_ids(test_id, train_ids, feats, topk=30)
            else:
                top5 = train_ids[:30]

            study_prefix   = test_id.split('_')[0]
            train_prefixes = [tid.split('_')[0] for tid in top5]
            output_list.append({
                "testid":  study_prefix,
                "trainid": train_prefixes
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(f"Saved sorted matches list to {output_file}")

if __name__ == "__main__":
    main()
