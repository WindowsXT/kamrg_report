import openai
import json
import re
import requests
from tqdm import tqdm
import argparse

########################################
# Part 1: Relation Extraction using OpenAIClient
########################################

class OpenAIClient:
    def __init__(self, api_key, base_url):
        # Using fixed API key and base URL; modify as needed
        self.api_key = "YOU API_KEY"
        self.api_base = "https://api.deepseek.com/v1"
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        # Adjust the model name as needed
        self.model = 'deepseek-reasoner'

    def get_messages(self, query):
        fewshot_samples = [
            {
                'context': (
                    "{'Bones are stable with mild degenerative changes of the spine.':"
                    "{'Bones': 'anatomy', 'stable': 'concept', 'mild': 'concept', "
                    "'degenerative changes': 'disorder_present', 'spine': 'anatomy'}}"
                ),
                'response': (
                    "{'Bones are stable with mild degenerative changes of the spine.': "
                    "[{'stable': 'Bones', 'relation':'modify'}, "
                    "{'mild':'degenerative changes', 'relation':'modify'}, "
                    "{'degenerative changes':'spine','relation':'located_at'}]}"
                )
            },
            {
                'context': (
                    "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': "
                    "{'dense': 'concept','retrocardiac': 'anatomy','opacity': 'disorder_present',"
                    "'slight': 'concept','blunting': 'disorder_present','left': 'concept',"
                    "'costophrenic': 'anatomy','angle': 'anatomy','small': 'concept','effusion': 'disorder_present'}}"
                ),
                'response': (
                    "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': "
                    "[{'dense': 'opacity', 'relation': 'modify'}, "
                    "{'opacity': 'retrocardiac', 'relation': 'located_at'}, "
                    "{'slight': 'blunting', 'relation': 'modify'}, "
                    "{'blunting': 'angle', 'relation': 'modify'}, "
                    "{'left': 'costophrenic', 'relation': 'modify'}, "
                    "{'small': 'effusion', 'relation': 'modify'}, "
                    "{'effusion': 'costophrenic', 'relation': 'located_at'}, "
                    "{'opacity':'effusion','relation':'suggestive_of'}, "
                    "{'blunting':'effusion','relation':'suggestive_of'}]}"
                )
            }
        ]

        messages = [
            {"role": "system", "content": (
                "You are a radiologist who is performing an entity's relationship extraction from the report portion of a radiology report. "
                "A clinical term can be one of ['anatomy', 'disorder_present', 'disorder_notpresent', 'procedures', 'devices', 'concept', "
                "'devices_present', 'devices_notpresent', 'size']. The relation can be one of ['modify', 'located_at', 'suggestive_of']. "
                "'suggestive_of' means the source entity (findings) may suggest the target entity (disease). "
                "'located_at' means the source entity is located at the target entity. "
                "'modify' denotes that the source entity modifies the target entity. "
                "Whenever there is a 'modify' relationship between a concept and an anatomy term, the direction should be concept -> anatomy. "
                "For example, in 'right pleural effusion', 'right' (concept) modifies 'pleural' (anatomy), while 'effusion' (disorder) is located_at 'pleural' (anatomy). "
                "Ensure that the source/target entity direction is correct. "
                "Given a piece of radiology text input in the JSON format: "
                "{'sentence':{'entity':'entity_type'}, 'sentence':{'entity':'entity_type'}}, "
                "please reply with the following JSON format: "
                "{'sentence':[{source_entity: 'target_entity', relation: 'relation'}, {source_entity: 'target_entity', relation: 'relation'}]}"
            )}
        ]

        for sample in fewshot_samples:
            messages.append({"role": "user", "content": sample['context']})
            messages.append({"role": "assistant", "content": sample['response']})

        messages.append({"role": "user", "content": query})
        return messages

    def chat_completion(self, messages):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                # Removed response_format parameter since deepseek-reasoner does not support JSON output
                request_timeout=600
            )
            # Obtain the model's reply content
            res = response["choices"][0]["message"]["content"]
            # Remove possible Markdown code block markers such as ```json ... ```
            res = re.sub(r'^```(?:json)?\s*\n', '', res)
            res = re.sub(r'\n```$', '', res)
            # Attempt to parse as JSON
            return json.loads(res)
        except (requests.Timeout, openai.error.Timeout) as e:
            print("Request timed out:", e)
            return None
        except openai.error.OpenAIError as e:
            print("OpenAI API error:", e)
            return None
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return None
        except Exception as e:
            print("Unexpected error:", e)
            return None

    def test_prompt(self, input_json):
        messages = self.get_messages(input_json)
        # Print messages sent to the model for debugging
        print("Messages sent to model:")
        for msg in messages:
            print(f"Role: {msg['role']}, Content: {msg['content']}")
        res = self.chat_completion(messages)
        return res

# Instantiate the client object
client = OpenAIClient(
    api_key="YOU API_KEY",
    base_url="https://api.deepseek.com/v1"
)

def evaluate_notes(json_file, save_json_file):
    """
    Reads the input file (e.g. iu_entities_chexpert_plus.json) with the following format:
    {
        "train": {
            "CXR2384_IM-0942": {
                "section_findings": "...",
                "res": { ... }
            },
            ...
        },
        "val": { ... },
        "test": { ... }
    }
    Performs relation extraction for every report in each group (train/val/test),
    saves the result in the 'res_relation' field, and records error reports.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    save_data_dict = {}
    error_reports = []

    for split in json_data:
        print(f"Processing split: {split}")
        save_data_dict[split] = {}
        for report_id, report_data in tqdm(json_data[split].items(), desc=f"Processing {split} Reports"):
            # Check if the report contains the key 'res'
            if 'res' not in report_data:
                print(f"Report {report_id} in {split} does not contain key 'res'. Skipping this report.")
                error_reports.append({"split": split, "report_id": report_id, "data": report_data})
                continue

            input_json = report_data['res']
            # Use the client object to interact with the model
            res = client.test_prompt(json.dumps(input_json))
            if res is None:
                print(f"Failed to process report: {report_id} in {split}")
                error_reports.append({"split": split, "report_id": report_id, "data": report_data})
                continue

            report_data['res_relation'] = res
            save_data_dict[split][report_id] = report_data

            print(f"Processed report: {report_id} in {split}")

    with open(save_json_file, 'w') as outfile:
        json.dump(save_data_dict, outfile, indent=4)

    if error_reports:
        with open('./error_reports.json', 'w') as error_file:
            json.dump(error_reports, error_file, indent=4)
        print(f"Error reports saved to './error_reports.json' with {len(error_reports)} entries.")

########################################
# Part 2: Postprocessing – Format Conversion and Flattening
########################################

def convert_json_format(input_dict):
    if not isinstance(input_dict, dict) or not input_dict:
        print("Error: convert_json_format received an invalid object:", input_dict)
        return None

    keys = list(input_dict.keys())
    if len(keys) >= 2:
        if "relation" in input_dict:
            # Choose the first non-"relation" key as the source entity
            source_entity = keys[0] if keys[0] != "relation" else keys[1]
            target_entity = keys[1] if keys[1] != "relation" else keys[0]
            return {
                "source entity": source_entity,
                "target entity": input_dict[source_entity],
                "relation": input_dict["relation"]
            }
        elif "source" in input_dict and "target" in input_dict:
            input_dict["source entity"] = input_dict.pop("source")
            input_dict["target entity"] = input_dict.pop("target")
            return input_dict
    else:
        print('Error in convert_json_format with unexpected dictionary structure:', input_dict)
        return None

def flatten_dict(d):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value))
        else:
            flat_dict[key] = value
    return flat_dict

def postprocess_json(input_json_file, save_json_file, error_report_file, start_idx=0, end_idx=None):
    """
    Postprocess the relation extraction results:
      1. Flatten the 'res' portion for each report;
      2. Use the 'res_relation' portion to convert the format for every sentence;
      3. If the number of sentences does not match, take the intersection;
      4. Record error reports when exceptions or format issues occur.
    """
    with open(input_json_file, 'r') as file:
        json_data = json.load(file)

    save_data_dict = {}
    error_reports = []

    for split in json_data:
        print(f"Postprocessing split: {split}")
        save_data_dict[split] = {}
        for report_id, report_data in tqdm(json_data[split].items(), desc=f"Processing {split} Reports"):
            save_data_dict_idx = report_data.copy()

            res_dict_idx = report_data.get('res', {})
            res_relation_dict_idx = report_data.get('res_relation', {})

            save_res_relation_dict_idx = {}

            sentence_list = list(res_dict_idx.keys())
            relation_sentence_list = list(res_relation_dict_idx.keys())

            # If the sentence counts do not match, take the intersection.
            if len(sentence_list) != len(relation_sentence_list):
                matching_sentences = set(sentence_list).intersection(set(relation_sentence_list))
                matching_count = len(matching_sentences)
                print(f"Adjusted Sentence count to {matching_count} for report {report_id} in {split}.")
                sentence_list = list(matching_sentences)
                relation_sentence_list = list(matching_sentences)

            # Flatten the entity results for each sentence.
            for sentence in res_dict_idx:
                res_dict_idx[sentence] = flatten_dict(res_dict_idx[sentence])

            for sentence in relation_sentence_list:
                if sentence not in res_dict_idx:
                    continue

                res_annotation = res_dict_idx.get(sentence, {})
                sentence_relation_list = res_relation_dict_idx.get(sentence, []) or []
                save_sentence_relation_list = []

                if not res_annotation:
                    print(f"Skipping empty sentence annotation for report {report_id} in {split}, sentence '{sentence}'.")
                    continue

                for sentence_relation_dict in sentence_relation_list:
                    try:
                        if isinstance(sentence_relation_dict, dict) and sentence_relation_dict:
                            save_sentence_relation_dict = convert_json_format(sentence_relation_dict)
                            if save_sentence_relation_dict:
                                save_sentence_relation_list.append(save_sentence_relation_dict)
                        else:
                            print(f"Skipping empty relation for report {report_id} in {split}, sentence '{sentence}'.")
                            error_reports.append({
                                'split': split,
                                'report_id': report_id,
                                'report': report_data,
                                'sentence': sentence,
                                'relation': sentence_relation_dict,
                                'reason': "Empty dictionary in res_relation"
                            })
                    except AttributeError as e:
                        print(f"Error: {e} in report {report_id} in {split}, sentence '{sentence}', relation: {sentence_relation_dict}")
                        error_reports.append({
                            'split': split,
                            'report_id': report_id,
                            'report': report_data,
                            'sentence': sentence,
                            'relation': sentence_relation_dict,
                            'reason': "Unexpected format or AttributeError in res_relation"
                        })

                save_res_relation_dict_idx[sentence] = save_sentence_relation_list

            save_data_dict_idx['res_relation'] = save_res_relation_dict_idx
            save_data_dict[split][report_id] = save_data_dict_idx

    with open(save_json_file, 'w') as outfile:
        json.dump(save_data_dict, outfile, indent=4)

    if error_reports:
        with open(error_report_file, 'w') as error_file:
            json.dump(error_reports, error_file, indent=4)
        print(f"Error reports saved to '{error_report_file}' with {len(error_reports)} entries.")

########################################
# Main: First perform relation extraction, then postprocessing
########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract relations from entities and postprocess the results."
    )
    # IU dataset parameters
    parser.add_argument(
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input JSON file for relation extraction (e.g., data/iu_xray/iu_entities_chexpert_plus.json)"
    )
    parser.add_argument(
        "--extracted_save_json_file",
        type=str,
        required=True,
        help="Path to save the relation extraction results (e.g., data/iu_xray/iu_entities_relations_chexpert_plus.json)"
    )
    parser.add_argument(
        "--postprocess_json_file",
        type=str,
        required=True,
        help="Path to save the postprocessed JSON file (e.g., data/iu_xray/iu_entities_relations_chexpert_plus_post.json)"
    )
    parser.add_argument(
        "--error_report_file",
        type=str,
        required=True,
        help="Path to save the error reports (e.g., data/iu_xray/iu_normalize_error_reports.json)"
    )
    
    # Mimic dataset parameters
    parser.add_argument(
        "--mimic_input_json_file",
        type=str,
        required=True,
        help="Path to the input JSON file for relation extraction for Mimic (e.g., data/mimic/mimic_entities_chexpert_plus.json)"
    )
    parser.add_argument(
        "--mimic_extracted_save_json_file",
        type=str,
        required=True,
        help="Path to save the relation extraction results for Mimic (e.g., data/mimic/mimic_entities_relations_chexpert_plus.json)"
    )
    parser.add_argument(
        "--mimic_postprocess_json_file",
        type=str,
        required=True,
        help="Path to save the postprocessed JSON file for Mimic (e.g., data/mimic/mimic_entities_relations_chexpert_plus_post.json)"
    )
    parser.add_argument(
        "--mimic_error_report_file",
        type=str,
        required=True,
        help="Path to save the error reports for Mimic (e.g., data/mimic/mimic_normalize_error_reports.json)"
    )
    
    args = parser.parse_args()

    # Process IU dataset
    print("Processing IU dataset for relation extraction...")
    evaluate_notes(args.input_json_file, args.extracted_save_json_file)
    print("Postprocessing IU dataset...")
    postprocess_json(args.extracted_save_json_file, args.postprocess_json_file, args.error_report_file)

    # Process Mimic dataset
    print("Processing Mimic dataset for relation extraction...")
    evaluate_notes(args.mimic_input_json_file, args.mimic_extracted_save_json_file)
    print("Postprocessing Mimic dataset...")
    postprocess_json(args.mimic_extracted_save_json_file, args.mimic_postprocess_json_file, args.mimic_error_report_file)
