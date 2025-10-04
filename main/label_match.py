import json
import pandas as pd
import numpy as np

def calculate_iou(test_labels, train_labels):
    """
    English: Map -1 to 1 and compute IoU over all positions
    """
    # 1 Map -1 to 1
    test  = np.where(test_labels  == -1, 1, test_labels)
    train = np.where(train_labels == -1, 1, train_labels)
    
    # 2. Calculate the intersection (both == 1)
    intersection = np.sum((test  == 1) & (train == 1))
    # 3. Calculate the union (either == 1)
    union = np.sum((test  == 1) | (train == 1))
    
    # 4. If there are no positive examples in either case, it is considered a perfect match
    if intersection == 0 and union == 0:
        return 1.0
    
    return intersection / union if union != 0 else 0.0

def main():
    with open('data/iu_xray/annotation.json', 'r', encoding='utf-8') as f:
        ann = json.load(f)

    test_ids  = [entry['id'] for entry in ann['test']]
    train_ids = [entry['id'] for entry in ann['train']]

    test_df      = pd.read_csv('data/iu_xray/iu_14_prediction_label.csv', dtype={'id': str})
    test_id_col  = test_df.columns[0]
    label_cols   = test_df.columns[1:].tolist()
    test_df[label_cols] = test_df[label_cols].fillna(0).astype(int)
    test_sub     = test_df[test_df[test_id_col].isin(test_ids)].copy()

    train_df     = pd.read_csv('data/iu_xray/iu_id_labeled_front.csv', dtype={'Report ID': str})
    train_id_col = train_df.columns[0]
    train_df[label_cols] = train_df[label_cols].fillna(0).astype(int)
    train_sub    = train_df[train_df[train_id_col].isin(train_ids)].copy()

    test_labels_matrix  = test_sub[label_cols].values
    train_labels_matrix = train_sub[label_cols].values

    exact_matches = {}
    for i, test_row in test_sub.iterrows():
        tid      = test_row[test_id_col]
        t_labels = test_labels_matrix[i]
        mapped_test = np.where(t_labels == -1, 0, t_labels)
        for j, train_row in train_sub.iterrows():
            tr_id       = train_row[train_id_col]
            tr_labels   = train_labels_matrix[j]
            mapped_train = np.where(tr_labels == -1, 0, tr_labels)
            if np.array_equal(mapped_test, mapped_train):
                exact_matches.setdefault(tid, []).append(tr_id)

    partial_matches = {}
    for i, test_row in test_sub.iterrows():
        tid = test_row[test_id_col]
        if tid in exact_matches:
            continue
        t_labels = test_labels_matrix[i]
        
        scores = []
        for j, train_row in train_sub.iterrows():
            tr_id     = train_row[train_id_col]
            tr_labels = train_labels_matrix[j]
            iou = calculate_iou(t_labels, tr_labels)
            if iou > 0:
                scores.append((tr_id, iou))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            partial_matches[tid] = [tr for tr, _ in scores]

    matches = {
        "exact_matches":  exact_matches,
        "partial_matches": partial_matches
    }
    with open('data/iu_xray/iu_matches.json', 'w', encoding='utf-8') as f:
        json.dump(matches, f, ensure_ascii=False, indent=4)

    print("—— Match statistics ——")
    print(f"Exactly match the number of test ids: {len(exact_matches)}")
    print(f"Partially match the number of test ids: {len(partial_matches)}")

if __name__ == '__main__':
    main()
