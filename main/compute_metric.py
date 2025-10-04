import os
import sys
import json
import argparse

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from transformers import AutoTokenizer

def compute_scores(gts, res):
    """
    Evaluate generated reports using MS COCO metrics.
    Tokenization is done with AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased").

    :param gts: dict mapping report id to a list of ground-truth captions, each as {"caption": text}
    :param res: dict mapping report id to a list of generated captions, each as {"caption": text}
    :return: dict containing scores for each metric
    """
    print("Tokenizing with AutoTokenizer from allenai/scibert_scivocab_uncased...")
    hf_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    
    def custom_tokenize(captions):
        # For each report id, convert each caption dict to a tokenized string
        for report_id, caption_list in captions.items():
            tokenized_list = []
            for caption_dict in caption_list:
                text = caption_dict["caption"]
                tokens = hf_tokenizer.tokenize(text)
                tokenized_text = " ".join(tokens)
                tokenized_list.append(tokenized_text)
            captions[report_id] = tokenized_list  # replace with list of tokenized strings
        return captions
    
    gts = custom_tokenize(gts)
    res = custom_tokenize(res)
    
    # Define list of evaluators
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDER"),
    ]
    eval_res = {}
    # Compute scores for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated radiology reports using MS COCO metrics."
    )
    parser.add_argument(
        "--generated_file",
        type=str,
        required=True,
        help="Path to the generated reports JSON file, e.g., data/iu_xray/iu_ds_mrg.json"
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ground-truth reports JSON file, e.g., data/iu_xray/mimic_ds_mrg.json"
    )
    args = parser.parse_args()

    generated_file = args.generated_file
    ground_truth_file = args.ground_truth_file

    # Load generated reports; each element contains "id" and "Final Report"
    with open(generated_file, "r", encoding="utf-8") as f:
        generated_reports = json.load(f)

    # Load ground-truth reports; each element contains "id" and "report"
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_reports = json.load(f)

    # Build ground-truth dict: key is report id, value is a list of {"caption": text}
    gts = {}
    for item in ground_truth_reports:
        report_id = item.get("id")
        report_text = item.get("report", "")
        if report_id and report_text:
            gts[report_id] = [{"caption": report_text}]

    # Build generated reports dict: key is report id, value is a list of {"caption": text}
    res = {}
    for item in generated_reports:
        report_id = item.get("id")
        final_report = item.get("Final Report", "")
        if report_id and final_report:
            res[report_id] = [{"caption": final_report}]

    # Only evaluate report ids present in both files
    common_ids = set(gts.keys()) & set(res.keys())
    if not common_ids:
        print("No matching report ids found; cannot evaluate.")
        return

    gts_eval = {rid: gts[rid] for rid in common_ids}
    res_eval = {rid: res[rid] for rid in common_ids}

    # Compute evaluation scores
    eval_res = compute_scores(gts_eval, res_eval)
    print("Evaluation results:")
    for metric, score in eval_res.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
