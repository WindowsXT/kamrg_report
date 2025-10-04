import openai
import json
from tqdm import tqdm
import re
import argparse

# Initialize DeepSeek client
class OpenAIClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

client = OpenAIClient(
    api_key="YOUR-API-KEY",
    base_url="https://api.deepseek.com/v1"
)
MODEL = 'deepseek-reasoner'

def get_messages(query):
    fewshot_samples = [
        {
            'context': "<Input> Unchanged position of the left upper extremity PICC line. Again seen are surgical clips projecting over the right hemithorax. Increased stranding opacities are noted in the left retrocardiac region.<\\Input>",
            'response': '{"Unchanged position of the left upper extremity PICC line.": {"Unchanged": "concept", "position": "concept", "left": "concept", "upper": "concept", "extremity": "anatomy", "PICC line": "device_present"}, "Again seen are surgical clips projecting over the right hemithorax.": {"surgical clips": "device_present", "right": "concept", "hemithorax": "anatomy"}, "Increased stranding opacities are noted in the left retrocardiac region.": {"Increased": "concept", "stranding": "concept", "opacities": "disorder_present", "left": "concept", "retrocardiac": "anatomy", "region": "anatomy"}}'
        }
    ]

    messages = [
        {"role": "system", "content": (
            "You are a radiologist performing clinical term extraction from the FINDINGS and IMPRESSION sections in the radiology report. "
            "A clinical term can belong to one of the following: ['anatomy', 'disorder_present', 'disorder_notpresent', 'procedures', 'devices', 'concept', 'devices_present', 'devices_notpresent', 'size']. "
            "'anatomy' refers to the anatomical body; "
            "'disorder_present' refers to findings or diseases that are present according to the sentence; "
            "'disorder_notpresent' refers to findings or diseases that are not present; "
            "'procedures' refers to procedures used to diagnose, measure, monitor, or treat problems; "
            "'devices' refers to any instrument or apparatus for medical purposes; "
            "'size' refers to the measurement of disorders or anatomy (e.g., '3mm' or '4x5 cm'); "
            "'concept' refers to descriptors such as 'acute' or 'chronic', 'large', measurements of size or severity, or other modifiers, or descriptors indicating normal anatomy. "
            "For example, in 'right pleural effusion', 'right' should be a 'concept', 'pleural' should be 'anatomy', and 'effusion' should be 'disorder_present' or 'disorder_notpresent'. "
            "Similarly, in 'normal cardiomediastinal silhouette', both 'normal' and 'silhouette' are 'concept', while 'cardiomediastinal' is 'anatomy'. "
            "Please extract terms one word at a time whenever possible, avoiding multi-word phrases. Note that words like 'no' and phrases such as 'no evidence of' are not considered entities. "
            "Given a list of radiology sentences in the format: <Input><sentence><sentence><\\Input>, please reply with JSON following this template: "
            "{'<sentence>':{'entity':'entity type','entity':'entity type'}, '<sentence>':{'entity':'entity type','entity':'entity type'}}"
        )}
    ]
    for sample in fewshot_samples:
        messages.append({"role": "user", "content": sample['context']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": query})
    return messages

def deepseek_input(messages):
    try:
        # Print messages sent to the model
        print("Messages sent to model:")
        for msg in messages:
            print(f"Role: {msg['role']}, Content: {msg['content']}")
        # Call the openai.ChatCompletion.create interface
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            request_timeout=600
        )
        print("Original model response:", response)
        # Extract and clean the returned content
        res = response["choices"][0]["message"]["content"].strip()
        # Remove Markdown code block markers such as ```json and ``` and extra whitespace
        res = re.sub(r"^```(?:json)?\s*\n", "", res)
        res = re.sub(r"\n```$", "", res)
        res = re.sub(r'{No entities as the sentence indicates absence of disorders}', '{"message": "No entities present"}', res)
        res = res.replace("'", '"')
        res = res.replace("XXXX", "[Invalid term]")
        res = res.replace('".No"', '"message": "No entities present"')
        res = re.sub(r'"[\d]+(\.\s*)?"', '"', res)
        parsed_res = json.loads(res)
        return parsed_res
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": f"JSON Decode Error: {e}", "raw_response": res if 'res' in locals() else ""}
    except Exception as e:
        print(f"Error in deepseek_input: {e}")
        return {"error": f"Error in deepseek_input: {e}"}

def test_prompt(findings_text):
    content = '<Input>' + findings_text + '<\\Input>'
    messages = get_messages(content)
    return deepseek_input(messages)

def process_report(report_data):
    findings_text = report_data['report']
    report_id = report_data['id']
    print(f"Processing report ID: {report_id}")
    res = test_prompt(findings_text)
    # If the returned result is None or contains an error, consider processing as failed
    if res is None or (isinstance(res, dict) and "error" in res):
        error_message = res["error"] if (isinstance(res, dict) and "error" in res) else "Unknown error"
        return report_id, {'section_findings': findings_text, 'res': None, 'error': error_message}
    else:
        for key in res.keys():
            if not res[key]:
                res[key] = {"message": f"No entities to extract from '{key}'."}
        return report_id, {'section_findings': findings_text, 'res': res}

def evaluate_notes(input_json_file, save_json_file, error_json_file):
    with open(input_json_file, 'r') as infile:
        data = json.load(infile)

    save_data = {}
    error_reports = []

    for split in ['train', 'val', 'test']:
        if split in data:
            save_data[split] = {}
            reports_list = data[split]
            for report_data in tqdm(reports_list, desc=f"Processing {split} Reports"):
                report_id, result = process_report(report_data)
                save_data[split][report_id] = result
                # If the result is empty or contains an error, save the error report
                if result.get('res') is None or result.get('error'):
                    error_reports.append({
                        'id': report_id,
                        'split': split,
                        'report': result['section_findings'],
                        'error': result.get('error', "Unknown error")
                    })

    with open(save_json_file, 'w') as outfile:
        json.dump(save_data, outfile, ensure_ascii=False, indent=4)

    with open(error_json_file, 'w') as error_file:
        json.dump(error_reports, error_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform entity extraction from the specified radiology report JSON file and save the results and error reports."
    )
    # IU dataset parameters
    parser.add_argument(
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input IU radiology reports JSON file, e.g., data/iu_xray/iu_radiology_reports.json"
    )
    parser.add_argument(
        "--save_json_file",
        type=str,
        required=True,
        help="Path to save IU entity extraction results JSON file, e.g., data/iu_xray/iu_entities_chexpert_plus.json"
    )
    parser.add_argument(
        "--error_json_file",
        type=str,
        required=True,
        help="Path to save IU error reports JSON file, e.g., data/iu_xray/iu_error_reports.json"
    )
    
    # Mimic dataset parameters
    parser.add_argument(
        "--mimic_input_json_file",
        type=str,
        required=True,
        help="Path to the input Mimic report JSON file, e.g., data/mimic/mimic_reports.json"
    )
    parser.add_argument(
        "--mimic_save_json_file",
        type=str,
        required=True,
        help="Path to save Mimic entity extraction results JSON file, e.g., data/mimic/mimic_entities_chexpert_plus.json"
    )
    parser.add_argument(
        "--mimic_error_json_file",
        type=str,
        required=True,
        help="Path to save Mimic error reports JSON file, e.g., data/mimic/mimic_error_reports.json"
    )
    
    args = parser.parse_args()

    openai.api_key = client.api_key
    openai.api_base = client.base_url

    print("Processing IU dataset...")
    evaluate_notes(args.input_json_file, args.save_json_file, args.error_json_file)

    print("Processing Mimic dataset...")
    evaluate_notes(args.mimic_input_json_file, args.mimic_save_json_file, args.mimic_error_json_file)
