# Explainable Radiology Report Generation with Multi-Stage Retrieval Augmented Chain-of-Thought Prompting

 The llm cot mrg selects the reference report associated with the image most similar to the queried image and generates the diagnostic report using the thought chain pipeline in the Large Language Model (LLM).


## System Overview

The llm cot mrg system consists of four main components:

1. Datasets Preparation
2. Entity and Relation Extraction
3. CheXpert Classification
4. Report Generation

### Installation and Dependencies
`conda env create -f environment.yml`

### LLM Cot Mrg 

1. **Datasets Preparation**:

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic`. You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

CheXpert NLP tool to extract observations from radiology reports. Read more about our project [here](https://github.com/stanfordmlgroup/chexpert-labeler)


Identifier splits for the five‐fold cross‐validation of the RAD-DINO classifier are included in the dataset’s metadata. Entity‐and‐relation extraction prompts are implemented in `dsr1_get_entity.py` and `dsr1_get_relation.py`, respectively, while the chain‐of‐thought prompting script for DeepSeek-R1‐driven report generation is provided in `dsr1_mrg_cot_prompt.py`.


2. **Entity and Relation Extraction**:

   Run python dsr1_get_entity.py \
   --input_json_file    data/iu_xray/iu_radiology_reports.json \
   --save_json_file     data/iu_xray/iu_entities_chexpert_plus.json \
   --error_json_file    data/iu_xray/iu_error_reports.json ' to extract entity from the report.

   Run python dsr1_get_relation.py \
   --input_json_file                data/iu_xray/iu_entities_chexpert_plus.json \
   --extracted_save_json_file       data/iu_xray/iu_entities_relations_chexpert_plus.json \
   --postprocess_json_file          data/iu_xray/iu_entities_relations_chexpert_plus_post.json \
   --error_report_file              data/iu_xray/iu_normalize_error_reports.json ' to extract relation from the report.
   
   %If you want to replace the MIMIC data set, you only need to modify the path to the MIMIC data set


3. **CheXpert Classification**:

   Run python rad_train.py to train rad-dino model.\
   --input_json_file   data/iu_xray/path_configs_chexpert.json \
   --output_dir        data/iu_xray/snapshot_path \
   --model_name        rad-dino

   Run python rad_class.py to get CheXpert 14 label.\
   --annotation_json   data/iu_xray/annotation.json \
   --image_dir         data/iu_xray/iu_front_images \
   --weights           data/iu_xray/snapshot_path\
   --csv_labels        data/iu_xray/iu_id_labeled_front.csv \
   --output_csv        data/iu_xray/iu_14_prediction_label.csv \

   Run python label_match.py to get Exact match/Partial match/No match.\
   --ann_json          data/iu_xray/annotation.json \
   --pred_csv          data/iu_xray/iu_14_prediction_label.csv \
   --train_csv         data/iu_xray/iu_id_labeled_front.csv \
   --out_json          data/iu_xray/iu_matches.json

   Run python extract_features.py to extract image features.\
   --config_json  data/iu_xray/path_configs_chexpert.json \
   --weights      data/iu_xray/snapshot_path \
   --out_npz      data/iu_xray/iu_image_features.npz \

   Run python feature_match.py to Coarse retrieval.\
   --matches_json  data/iu_xray/iu_matches.json \
   --features_npz  data/iu_xray/iu_all_image_features.npz \
   --output_json   data/iu_xray/iu_matches_top30.json \

   Run python atcam_kl.py to generate AGCAM.\
   --csv_labels      data/iu_id_labeled_front.csv \
   --input_dir       data/iu_xray/iu_front_images \
   --matches_json    data/iu_xray/iu_matches_top30.json \
   --weights         data/iu_xray/snapshot_path \
   --out_dir         data/iu_xray/heatmaps/agcam_kl \

   Run python kl_retrieval.py to Fine-grained retrieval.\
   --p_dir       data/iu_xray/heatmaps/agcam_kl/P \
   --q_dir       data/iu_xray/heatmaps/agcam_kl/Q \
   --matches     data/iu_xray/iu_matches_top30.json \
   --output      data/iu_xray/iu_kl_top10.json \

   Run python kg_alignment.py to knowledge alignment.\
   --top10_json  data/iu_xray/iu_kl_top10.json \
   --report_json data/iu_xray/iu_entities_relations_chexpert_plus_post.json \
   --output_json data/iu_xray/iu_kg_top5.json \ 

   %If you want to replace the MIMIC data set, you only need to modify the path to the MIMIC data set
r
4. **Report Generation**:
   Run python dsr1_main_mrg.py to generate reports.\
  --matches_file       data/iu_xray/iu_kg_top5.json \
  --report_train_file  data/iu_xray/iu_radiology_reports.json \
  --output_file_path   data/iu_xray/iu_ds_mrg.json \
  --log_file_path      data/iu_xray/iu_ds_log.json \





