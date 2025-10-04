import json
import os
from tqdm import tqdm
from openai import OpenAI
import pandas as pd  # 
from dsr1_mrg_cot_prompt import ReportGenerationProcess

# ----------------------------------------------------------------------------
# DeepSeek client initialization
# ----------------------------------------------------------------------------
client = OpenAI(
    api_key="",  # Please replace it with your DeepSeek API Key
    base_url="https://api.deepseek.com/v1"
)

def load_report_content(filepath, report_id_key, report_id_value, content_key):
    """Load the specified report content from the JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            content = next((item[content_key] for item in data if item.get(report_id_key) == report_id_value), None)
            if not content:
                raise ValueError(f"No content found for {report_id_key}={report_id_value}")
            return content
    except Exception as e:
        print(f"Error loading content from {filepath}: {e}")
        return None


def save_report_as_json(report, test_id, output_file_path):
    """Save the report as a JSON file and update or append the entries"""
    try:
        if not report or not isinstance(report, dict):
            raise ValueError(f"Invalid report format for test ID {test_id}: must be a non-empty dictionary, got {type(report)}, value: {report}")
        if "Final Report" not in report:
            raise ValueError(f"Missing 'Final Report' key in report for test ID {test_id}: {report}")

        result = report.copy()
        result["id"] = test_id

        existing_data = []
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    print(f"Error: Existing data in {output_file_path} is not a list. Overwriting.")
                    existing_data = []

        existing_ids = {item.get("id") for item in existing_data}
        if test_id in existing_ids:
            for item in existing_data:
                if item.get("id") == test_id:
                    item.update(result)
                    print(f"Updated report with ID {test_id}.")
                    break
        else:
            existing_data.append(result)
            print(f"Added new report with ID {test_id}.")

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(existing_data, output_file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file_path}")
        return True
    except Exception as e:
        print(f"Error saving results for test ID {test_id}: {e}")
        return False


def process_report(matches_file, report_train_file, output_file_path, log_file_path, label_df):
    interaction_log = {}
    try:
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches_data = json.load(f)

        for entry in tqdm(matches_data, desc="Processing Reports", unit="report"):
            test_id = entry.get("testid")
            train_ids = entry.get("trainids")
            interaction_log[test_id] = {"conversation": [], "final_generated_report": None, "error": None}

            if not test_id or not train_ids or not isinstance(train_ids, list) or len(train_ids) < 1:
                err = f"Missing test ID or invalid train IDs: {entry}"
                interaction_log[test_id]["error"] = err
                print(f"Skipping {test_id}: {err}")
                continue

            reference_reports = []
            load_success = True
            for i, train_id in enumerate(train_ids, 1):
                content = load_report_content(report_train_file, "id", train_id, "report")
                if not content or not isinstance(content, str) or not content.strip():
                    err = f"Invalid or missing content for train ID {train_id}"
                    interaction_log[test_id]["error"] = err
                    print(f"Error: {err}")
                    load_success = False
                    break
                if len(train_ids) > 1:
                    content = f"Reference Report {i}: {content}"
                reference_reports.append(content)

            if not load_success:
                continue


            positive_labels = set()
            for train_id in train_ids:
                prefix = str(train_id)
                prefix_pattern = f"{prefix}_"
 
                matches = label_df.index[
                    label_df.index.str.startswith(prefix_pattern) |
                    (label_df.index == prefix)
                ]

                if matches.empty:
                    print(f"[Warning] No entries starting with '{prefix}_' or equal to '{prefix}' were found in the tag table.")
                    continue

                # Summarize the positive labels in these matching rows
                df_sub = label_df.loc[matches]
                pos = df_sub.columns[(df_sub == 1).any(axis=0)]
                positive_labels.update(pos.tolist())

            # Determine if it is No Finding
            if not positive_labels:
                selected_labels = ["No Finding"]
            else:
                selected_labels = sorted(positive_labels)

            # Perform multi-step generation
            try:
                report_count = len(reference_reports)
                process = ReportGenerationProcess(
                    reference_reports=reference_reports,
                    client=client,
                    report_count=report_count,
                    labels=selected_labels
                )
                final_report = process.run()

                interaction_log[test_id]["conversation"] = process.conversation
                interaction_log[test_id]["final_generated_report"] = final_report

                if final_report:
                    ok = save_report_as_json(final_report, test_id, output_file_path)
                    if ok:
                        print(f"Saved generated report for ID {test_id}")
                    else:
                        interaction_log[test_id]["error"] = f"Failed to save report {test_id}"
                else:
                    err = f"Model did not return a report for {test_id}"
                    interaction_log[test_id]["error"] = err
                    print(f"Error: {err}")
            except Exception as e:
                import traceback
                err = f"Error processing {test_id}: {e}\n{traceback.format_exc()}"
                interaction_log[test_id]["error"] = err
                print(err)
                continue

    except Exception as e:
        import traceback
        err = f"Error in process_report: {e}\n{traceback.format_exc()}"
        interaction_log["general_error"] = err
        print(err)

    # Save interaction logs
    try:
        with open(log_file_path, 'w', encoding='utf-8') as lf:
            json.dump(interaction_log, lf, indent=4, ensure_ascii=False)
        print(f"Interaction log saved to {log_file_path}")
    except Exception as e:
        print(f"Failed to save interaction log: {e}")


def main():
    matches_file     = 'data/iu_xray/iu_kg_alignment_top5.json'
    train_file       = 'data/iu_xray/iu_radiology_reports.json'
    output_file      = 'data/iu_xray/iu_ds_mrg.json'
    interaction_log  = 'data/iu_xray/iu_ds_log.json'

    # Load and preprocess the label CSV in main
    label_df = pd.read_csv(
        'data/iu_xray/iu_id_labeled_front.csv',
        dtype={'Report ID': str}
    )
    label_df = label_df.fillna(0).replace(-1, 1)
    label_df.set_index('Report ID', inplace=True)

    process_report(
        matches_file,
        train_file,
        output_file,
        interaction_log,
        label_df
    )

if __name__ == '__main__':
    main()
