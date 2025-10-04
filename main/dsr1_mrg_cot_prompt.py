import sys
import time
import json
import re
import traceback
import logging
import os
from typing import List, Optional

class BaseStep:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def get_message(self):
        return {"role": self.role, "content": self.content}

    def process(self, conversation, client, step=None, report_id=None):
        conversation.append(self.get_message())
        response = llama_input(conversation, client, step=step, report_id=report_id)  # 传递 report_id
        if response:
            conversation.append({"role": "assistant", "content": response})
            return self.process_output(response)
        if self.__class__.__name__ == "Step3SummarizeReports":
            return "No findings extracted due to model failure."
        return None

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            if self.__class__.__name__ == "Step3SummarizeReports":
                return "No findings extracted due to empty model output."
            raise ValueError(f"{self.__class__.__name__}: Model output is empty")
        cleaned = re.sub(r'```json\n|\n```|```', '', model_output).strip()
        if not cleaned:
            if self.__class__.__name__ == "Step3SummarizeReports":
                return "No findings extracted due to empty cleaned output."
            raise ValueError(f"{self.__class__.__name__}: Cleaned output is empty")
        return cleaned

def llama_input(conversation, client, step=None, report_id=None):
    if not conversation or not isinstance(conversation, list):
        return "[ERROR]"

    msgs = conversation[-3:]
    delay = 20
    max_wait_time = 600
    total_wait_time = 0
    empty_count = 0
    max_empty_retries = 5

    while total_wait_time < max_wait_time:
        try:
            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=msgs,
                timeout=600,
                max_tokens=10000,
                temperature=0.0
            )
            raw = response.choices[0].message.content
            content = raw.strip()
            if content:
                return content

            empty_count += 1
            if empty_count >= max_empty_retries:
                sys.exit(1)

        except Exception as e:
            pass

        time.sleep(delay)
        total_wait_time += delay
        delay = min(delay * 2, 600)

    return "[ERROR]"

def extract_json(text):
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        match = re.search(r'\{[^{}]*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                pass
        return None


# --------------------------------------------------------------------------------
# Multi-step interaction classes
# --------------------------------------------------------------------------------
class Step1DefineRole(BaseStep):
    def __init__(self):
        content = """As an experienced radiologist, generate a chest X-ray report based on one or multiple reference reports, associated with CheXpert labels (e.g., 'No Finding' for negative, 'Cardiomegaly' for positive). Follow these guidelines:
1) Exclude irrelevant details, such as imaging technical details, image quality descriptions, or clinical recommendations.
2) Ensure the output uses a professional tone, defined as:
   - Precise, and using standard radiological terminology.
   - Avoidance of redundant verbs or colloquial expressions.
   - Sentence structure that is clear, direct.
   Example of professional tone: 'The heart size is normal. Lungs are clear. No pleural effusion, pneumothorax, or bony abnormality. No acute cardiopulmonary disease.'

**Note**: The above are the instructions for Step 1. Please reply after receiving the instructions for Step 1, and then wait for Step 2 to provide the reference report and label."""
        super().__init__("system", content)

class Step2ProvideReferenceReports(BaseStep):
    def __init__(self, report_count: int = 1, labels: List[str] = None):
        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if not labels or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("labels must be a non-empty list of strings.")
        if any(lbl not in ReportGenerationProcess.VALID_LABELS for lbl in labels):
            raise ValueError(f"Invalid CheXpert labels: {labels}")

        is_negative = (labels == ["No Finding"])
        label_text = "No Finding" if is_negative else ", ".join(labels)
        label_desc = f"negative label: {label_text}" if is_negative else f"positive labels: {label_text}"
        verb_s = "is a single" if report_count == 1 else f"are {report_count}"
        plural_s = "" if report_count == 1 else "s"

        intro = (
            f"Below {verb_s} chest X-ray reference report{plural_s} "
            f"provided by a radiologist using standardized medical terminology. "
            f"The report{plural_s} {'is' if report_count == 1 else 'are'} associated with {label_desc}."
        )
        task_and_wait = (
            "Your task in this step is **only** to confirm receipt of these reference reports.\n "
            "**Note**: The above are the instructions for Step 2. Please reply 'All reference reports have been received.' after receiving the instructions in Step 2, Do **not** generate any summary or report at this stage, and then wait for further explanations in Step 3 to summarize."
        )
        content = (
            intro + "\n\n"
            f"The content of the report{plural_s} is clearly presented as follows:\n\n"
            "```json\n"
            "{reference_reports}\n"
            "```\n\n"
            + task_and_wait
        )

        super().__init__("user", content)
        self.labels = labels

class Step3SummarizeReports(BaseStep):
    def __init__(self, previous_report_content: str, report_count: int = 1, labels: list[str] = None):
        if not isinstance(previous_report_content, str):
            previous_report_content = ""
        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if labels is None or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("labels must be a list of strings (may be empty for all-negative scenario).")
        if any(lbl not in ReportGenerationProcess.VALID_LABELS for lbl in labels):
            raise ValueError(f"Invalid CheXpert labels: {labels}")

        is_negative = (len(labels) == 0 or labels == ["No Finding"])
        label_text = "No Finding" if is_negative else ", ".join(labels)
        label_desc = f"negative label: {label_text}" if is_negative else f"positive labels: {label_text}"

        intro = (
            f"You have a single chest X-ray reference report, associated with {label_desc}." if report_count == 1
            else f"You have {report_count} chest X-ray reference reports, associated with {label_desc}."
        )

        anatomical_sections = (
            ReportGenerationProcess.NEGATIVE_SECTIONS if is_negative else
            ReportGenerationProcess.POSITIVE_SECTIONS
        )
        anatomical_order = ", ".join(anatomical_sections)

        steps = []

        # Step 1: Extraction
        if report_count == 1:
            if is_negative:
                extraction = (
                    "1. **Information Extraction**:\n"
                    f"   - Extract all healthy findings using exact original wording.\n"
                    f"   - Arrange findings in anatomical order: {anatomical_order}. "
                    f"Merge all heart and mediastinum descriptions into one section: 'heart_mediastinum'.\n"
                    "   - Healthy findings include phrases like 'clear lungs', 'normal cardiac silhouette', 'no pleural effusion'."
                )
            else:
                extraction = (
                    "1. **Information Extraction**:\n"
                    f"   - Extract all findings related to {label_text}, using exact original wording.\n"
                    f"   - Arrange them in anatomical order: {anatomical_order}.\n"
                    f"   - Relevant findings include direct or related mentions (e.g., 'cardiac enlargement' for Cardiomegaly)."
                )
        else:
            if is_negative:
                extraction = (
                    "1. **Information Extraction**:\n"
                    "   - Extract healthy findings from each report, using exact wording (verbatim).\n"
                    "   - Treat each complete sentence (ending with '.') as one finding.\n"
                    f"   - Arrange findings in anatomical order: {anatomical_order}, with report index.\n"
                    "   - Example:\n"
                    "     - heart_mediastinum: 'Heart size and pulmonary vascularity within normal limits.' (Report 1)"
                )
            else:
                extraction = (
                    "1. **Information Extraction**:\n"
                    f"   - Extract disease findings related to {label_text} from each report using exact original wording.\n"
                    f"   - Arrange in anatomical order: {anatomical_order}, with report index.\n"
                )

        # Step 2: Integration
        if is_negative:
            integration = (
                "2. **Information Integration**:\n"
                f"   - Merge all heart and mediastinum descriptions into 'heart_mediastinum'.\n"
                f"   - List each sentence under its anatomical section with report index.\n"
                f"   - Example:\n"
                f"     - heart_mediastinum: 'Heart size within normal limits.' (Report 1), 'Cardiac and mediastinal contours unremarkable.' (Report 2)"
            )
        else:
            integration = (
                "2. **Information Integration**:\n"
                f"   - For each anatomical section, list findings like:\n"
                f"     - lungs: 'Bibasilar opacities seen.' (Report 2), 'Patchy consolidation in left lower lobe.' (Report 3)"
            )

        # Step 3: Screening
        screening = (
            "3. **Information Screening**:\n"
            "   - Include only relevant findings.\n"
            "   - Discard irrelevant details such as imaging technique notes, positioning, quality remarks, or follow-up recommendations."
        )

        steps.extend([extraction, integration, screening])

        # Final prompt
        content = (
            intro + "\n\n" +
            "Please summarize the reports according to the following rules:\n\n" +
            "\n\n".join(steps) + "\n\n" +
            "Now generate the summarized findings organized by anatomical section.\n"
            "Use exact original wording with report index. This output will be validated in Step 4.\n"
            "**Do not wait. Perform the summarization task now.**"
        )

        super().__init__("user", content)
        self.labels = labels

class Step4ValidateSummary(BaseStep):
    def __init__(self, previous_summary: str, report_count: int = 1, labels: List[str] = None):
        if not isinstance(previous_summary, str):
            previous_summary = "No findings extracted due to invalid input."
        if not previous_summary.strip():
            previous_summary = "No findings extracted due to empty input."
        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if labels is None or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("labels must be a list of strings (may be empty for all-negative scenario).")
        if labels and any(lbl not in ReportGenerationProcess.VALID_LABELS for lbl in labels):
            raise ValueError(f"Invalid CheXpert labels: {labels}")

        self.is_negative = not labels or labels == ["No Finding"]
        self.labels = labels

        sections = (
            ReportGenerationProcess.NEGATIVE_SECTIONS
            if self.is_negative else
            ReportGenerationProcess.POSITIVE_SECTIONS
        )

        label_text = "No Finding" if self.is_negative else ", ".join(labels)
        polarity = "negative" if self.is_negative else "positive"


        rules = []
        if report_count == 1:
            if self.is_negative:
                rules.extend([ 
                    f"- Verify that all healthy findings from the Step 3 summary are included and arranged in anatomical order: {', '.join(sections)}.",
                    "- When placing findings in the JSON, use the exact original wording from the summary with no changes or abbreviations."
                ])
            else:
                rules.extend([ 
                    f"- Verify that all findings related to {label_text} from the Step 3 summary are included and arranged in anatomical order: {', '.join(sections)}.",
                    f"- Ensure each finding is associated with its corresponding CheXpert label ({label_text}).",
                    "- When placing findings in the JSON, use the exact original wording from the summary with no changes or abbreviations."
                ])
        else:

            rules.extend([ 
                f"- Verify that all findings from the Step 3 summary are included and arranged in anatomical order: {', '.join(sections)}.",
                f"- Ensure each finding is associated with its corresponding CheXpert label ({label_text}).",
                "- Use exact wording; do not introduce abbreviations or paraphrases."
            ])

        rules.append(
            f"- Discard irrelevant findings, such as imaging technique details, image quality descriptions, or clinical recommendations."
        )

        checks = [
            f"Confirm all required findings related to {label_text} are included using exact original wording from the reference report(s)."
        ]

        finding_field = "unhealthy_findings" if not self.is_negative else "healthy_findings"
        finding_structure = (
            "{\"text\": \"finding description\", \"report\": [report_id, ...]}"
            if self.is_negative else
            "{\"text\": \"finding description\", \"report\": [report_id, ...]}"
        )

        content = (
            f"CheXpert labels: {label_text} ({polarity}).\n\n"
            f"Please validate the summary from Step 3:\n\n"
            "Summary to validate:\n"
            "---\n"
            f"{previous_summary}\n"
            "---\n\n"
            "Validation Rules:\n"
            + "\n".join(rules) + "\n\n"
            "Perform the following checks:\n"
            + "\n\n".join(f"- {c}\n" for c in checks) + "\n\n"
            "Return **only** the JSON object in the following format:\n"
            "```json\n"
            "{\n"
            f"  \"{finding_field}\": {{\n"
            + ",\n".join(
                f"    \"{section}\": [{finding_structure}]"
                for section in sections
            ) +
            "\n  }\n"
            "}\n"
            "```\n"
            "**Important**: Do NOT include any explanations, commentary, or confirmation messages. Only return the JSON object."
        )

        super().__init__("user", content)

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        key = "unhealthy_findings" if not self.is_negative else "healthy_findings"
        sections = (
            ReportGenerationProcess.NEGATIVE_SECTIONS
            if self.is_negative else
            ReportGenerationProcess.POSITIVE_SECTIONS
        )

        def fallback_json():
            return json.dumps({
                key: {section: [] for section in sections}
            })

        if not model_output:
            return fallback_json()

        cleaned = model_output.strip()
        json_match = re.search(r'```json\n([\s\S]*?)\n```', cleaned)
        cleaned = json_match.group(1) if json_match else cleaned

        try:
            obj = json.loads(cleaned)
            if not isinstance(obj, dict) or key not in obj:
                raise ValueError(f"Output must include '{key}' field")
            if not all(section in obj[key] for section in sections):
                raise ValueError(f"{key} must include all sections: {', '.join(sections)}")
            for section in obj[key]:
                for finding in obj[key][section]:
                    if not isinstance(finding, dict):
                        raise ValueError(f"Each entry in {section} must be a dictionary")
                    if not "text" in finding or not "report" in finding:
                        raise ValueError(f"Missing 'text' or 'report' in finding of section {section}")
        except Exception as e:
            return fallback_json()

        return cleaned

class Step5SetSystemNegativePromptTemplates(BaseStep):
    def __init__(self, previous_report_content: str, report_count: int = 1, labels: List[str] = None):
        if not isinstance(previous_report_content, str):
            previous_report_content = json.dumps({
                "healthy_findings": {section: [] for section in ReportGenerationProcess.NEGATIVE_SECTIONS}
            })

        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if labels is None or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("labels must be a list of strings.")

        try:
            report_data = json.loads(previous_report_content)
            if not isinstance(report_data, dict) or "healthy_findings" not in report_data:
                raise ValueError("previous_report_content must include 'healthy_findings' field")
            findings = report_data["healthy_findings"]
            required_sections = ReportGenerationProcess.NEGATIVE_SECTIONS
            if not all(section in findings for section in required_sections):
                raise ValueError(f"healthy_findings must include sections: {required_sections}")
        except (json.JSONDecodeError, ValueError) as e:
            findings = {section: [] for section in ReportGenerationProcess.NEGATIVE_SECTIONS}

        self.healthy_findings = findings
        self.labels = labels

        label_text = ", ".join(labels) if labels else "No Finding"

        High_Frequency_Sentence_Chains = (
            f"   Ultra-High Frequency:\n"
            f"   - 'The heart is normal in size' → 'The mediastinum is unremarkable' → 'The lungs are clear'\n"
            f"   - 'Cardiac and mediastinal contours are within normal limits' → 'The lungs are clear' → 'Bony structures are intact'\n"
            f"   Extremely High Frequency:\n"
            f"   - 'The lungs are clear' → 'There is no pneumothorax or pleural effusion' → 'The heart and mediastinum are normal' → 'The skeletal structures are normal'\n"
            f"   - 'The cardiomediastinal silhouette is normal in size and contour' → 'No focal consolidation, pneumothorax or large pleural effusion' → 'Negative for acute bone abnormality'\n"
            f"   - 'The lungs are clear' → 'There is no pneumothorax or pleural effusion' → 'The heart and mediastinum are within normal limits' → 'Bony structures are intact'\n"
            f"   High Frequency:\n"
            f"   - 'The lungs are clear bilaterally' → 'Specifically, no evidence of focal consolidation, pneumothorax, or pleural effusion' → 'Cardio mediastinal silhouette is unremarkable' → 'Visualized osseous structures of the thorax are without acute abnormality'\n"
            f"   - 'Heart size and mediastinal contours appear within normal limits' → 'Pulmonary vascularity is within normal limits' → 'No focal consolidation, suspicious pulmonary opacity, pneumothorax or definite pleural effusion' → 'Visualized osseous structures appear intact'\n"
            f"   - 'Heart size within normal limits' → 'No pleural effusions' → 'There is no evidence of pneumothorax' → 'Osseous structures are intact'\n"
            f"   - 'The heart size and pulmonary vascularity appear within normal limits' → 'The lungs are free of focal airspace disease' → 'No pleural effusion or pneumothorax is seen'\n"
            f"   Medium-High Frequency:\n"
            f"   - 'Heart size and mediastinal contours are normal in appearance' → 'No consolidative airspace opacities' → 'No radiographic evidence of pleural effusion or pneumothorax' → 'Visualized osseous structures are intact'\n"
            f"   - 'The lungs are clear without evidence of focal airspace disease' → 'There is no evidence of pneumothorax or large pleural effusion' → 'The cardiac and mediastinal contours are within normal limits' → 'The bones are unremarkable'\n"
            f"   - 'The cardiac contours are normal' → 'The lungs are clear' → 'Thoracic spondylosis'\n"
            f"   - 'Both lungs are clear and expanded' → 'Heart and mediastinum normal'\n"
            f"   - 'The heart is normal in size and contour' → 'There is no mediastinal widening' → 'The lungs are clear bilaterally' → 'No large pleural effusion or pneumothorax' → 'The bones are intact'\n"
            f"   Medium Frequency:\n"
            f"   - 'Heart size normal' → 'Lungs are clear' → 'mediastinal contours are normal' → 'No pneumonia, effusions, edema, pneumothorax, adenopathy, nodules or masses'\n"
            f"   - 'The heart size is normal' → 'The mediastinal contour is within normal limits' → 'The lungs are free of any focal infiltrates' → 'There are no nodules or masses' → 'No visible pneumothorax' → 'No visible pleural fluid' → 'There is no visible free intraperitoneal air under the diaphragm'\n"
            f"   - 'The cardiomediastinal silhouette is within normal limits for size and contour' → 'The lungs are clear of focal airspace disease, pneumothorax, or pleural effusion' → 'There are no acute bony findings'\n"
            f"   - 'Mediastinal contours are normal' → 'Lungs are clear' → 'There is no pneumothorax or large pleural effusion'\n"
            f"   - 'Lungs are clear' → 'No pleural effusions or pneumothoraces' → 'Heart and mediastinum of normal size and contour'\n"
            f"   - 'The lungs and pleural spaces show no acute abnormality' → 'Heart size and pulmonary vascularity within normal limits' → 'No pneumothorax or pleural effusion. Osseous structures are grossly intact'\n"
            f"   - 'Normal cardiomediastinal silhouette' → 'There is no focal consolidation' → 'There are no evidience of a large pleural effusion' → 'There is no pneumothorax' → 'There is no acute bony abnormality seen'\n"
                )

        content = (
            "1. CheXpert Labels & Healthy Findings\n"
            f"   - Labels: '{label_text}'.\n"
            "   - Healthy Findings (JSON) from Step 4 **must be passed exactly as-is**, with **no changes, abbreviations, or reformatting**:\n"
            "```json\n"
            f"{json.dumps({'healthy_findings': self.healthy_findings}, indent=2)}\n"
            "```\n\n"
            "2. High-Frequency Sentence Chains (provided for Step 6 report generation):\n"
            f"{High_Frequency_Sentence_Chains}\n\n"
            "3. **Note**: The above are the instructions for Step 5. Do **not** alter the JSON above; return it verbatim in next prompt to Step 6, then await instructions to generate the report."
        )

        super().__init__("user", content)

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            return json.dumps({"healthy_findings": self.healthy_findings})

        cleaned_output = re.sub(r'```json\n|\n```|```', '', model_output).strip()

        try:
            parsed_output = json.loads(cleaned_output)
            if not isinstance(parsed_output, dict) or "healthy_findings" not in parsed_output:
                raise ValueError("Missing 'healthy_findings' field in model output")

            value = parsed_output["healthy_findings"]
            if not isinstance(value, dict):
                raise ValueError("'healthy_findings' must be a dict")

            for section in ReportGenerationProcess.NEGATIVE_SECTIONS:
                if section not in value or not isinstance(value[section], list):
                    raise ValueError(f"Missing or invalid section '{section}' in healthy_findings")

            return json.dumps({"healthy_findings": value}, indent=2)

        except Exception as e:
            return json.dumps({"healthy_findings": self.healthy_findings})

class Step5SetSystemPositivePromptTemplates(BaseStep):
    HIGH_FREQ_PHRASES = {
        "Enlarged Cardiomediastinum": [
            "enlarged cardiac silhouette", "widened mediastinum", "cardiomediastinal silhouette is enlarged",
            "enlargement of the cardiac silhouette", "mediastinal widening"
        ],
        "Cardiomegaly": [
            "cardiomegaly", "enlarged heart", "cardiac enlargement", "moderate cardiomegaly", "severe cardiomegaly"
        ],
        "Lung Lesion": [
            "lung nodule", "pulmonary nodule", "mass", "cavitary lesion", "nodular opacity"
        ],
        "Lung Opacity": [
            "opacity", "hazy opacity", "patchy opacity", "increased opacity", "focal opacity"
        ],
        "Edema": [
            "pulmonary edema", "interstitial edema", "edema", "pulmonary vascular congestion", "vascular congestion"
        ],
        "Consolidation": [
            "consolidation", "airspace consolidation", "focal consolidation", "confluent consolidation",
            "consolidation concerning for pneumonia"
        ],
        "Pneumonia": [
            "pneumonia", "infectious process", "consolidation concerning for pneumonia", "pneumonic infiltrate",
            "developing pneumonia"
        ],
        "Atelectasis": [
            "atelectasis", "collapse", "volume loss", "compressive atelectasis", "subsegmental atelectasis"
        ],
        "Pneumothorax": [
            "pneumothorax", "small pneumothorax", "pneumothoraces", "tension pneumothorax"
        ],
        "Pleural Effusion": [
            "pleural effusion", "small pleural effusion", "moderate pleural effusion", "bilateral pleural effusions",
            "pleural effusion is stable"
        ],
        "Pleural Other": [
            "pleural thickening", "pleural scarring", "pleural abnormality", "pleural calcification"
        ],
        "Fracture": [
            "rib fracture", "fracture", "rib fractures", "non-united fracture"
        ],
        "Support Devices": [
            "endotracheal tube", "nasogastric tube", "central venous catheter", "chest tube",
            "support devices are unchanged"
        ]
    }

    def __init__(self, previous_report_content: str, report_count: int = 1, labels: List[str] = None):
        if not isinstance(previous_report_content, str):
            previous_report_content = ""

        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if labels is None or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            raise ValueError("labels must be a list of strings.")
        if not labels or labels == ["No Finding"]:
            raise ValueError("Step5SetSystemPositivePromptTemplates expects positive labels.")
        if any(lbl not in ReportGenerationProcess.VALID_LABELS for lbl in labels):
            raise ValueError(f"Invalid CheXpert labels: {labels}")

        try:
            report_data = json.loads(previous_report_content)
            if not isinstance(report_data, dict) or "unhealthy_findings" not in report_data:
                raise ValueError("previous_report_content must include 'unhealthy_findings' field")

            findings = report_data["unhealthy_findings"]
            expected_sections = ReportGenerationProcess.POSITIVE_SECTIONS
            if not all(section in findings for section in expected_sections):
                raise ValueError(f"Missing required sections in 'unhealthy_findings': {expected_sections}")

            for section in findings:
                for item in findings[section]:
                    if not isinstance(item, dict) or "text" not in item:
                        raise ValueError(f"Invalid finding structure in section '{section}'")

        except (json.JSONDecodeError, ValueError):
            findings = {section: [] for section in ReportGenerationProcess.POSITIVE_SECTIONS}

        self.labels = labels
        self.findings = findings

        label_text = ", ".join(labels)
        anatomical_order = ", ".join(ReportGenerationProcess.POSITIVE_SECTIONS)

        header = (
            f"CheXpert labels: {label_text} (positive).\n"
            f"Purpose: Prepare structured findings for generating a radiology report for positive findings in Step 6.\n"
        )

        anatomical_keywords = (
            "Anatomical Keywords:\n"
            "- Heart: 'cardiac', 'heart', 'cardiomediastinal silhouette', 'heart size', 'cardiac contours'.\n"
            "- Mediastinum: 'mediastinum', 'mediastinal', 'mediastinal contours', 'mediastinal widening', 'cardiomediastinal silhouette'.\n"
            "- Lungs: 'lungs', 'pulmonary', 'consolidation', 'airspace disease', 'infiltrates', 'nodules', 'masses', 'pulmonary vascularity'.\n"
            "- Pleura: 'pleural', 'pleural effusion', 'pneumothorax'.\n"
            "- Bones: 'bony', 'osseous', 'bones', 'thoracic spondylosis', 'acute bony findings', 'skeletal', 'rib fracture'.\n"
        )

        instructions = (
            f"Instructions:\n"
            f"- Extract findings from the validated summary related to '{label_text}' for generating a radiology report.\n"
            f"- Organize findings by anatomical region ({anatomical_order}) and associate each finding with its corresponding CheXpert label.\n"
            f"- For findings appearing in ≥2 reports (if report_count > 1) or clinically significant findings (e.g., pneumothorax, cardiomegaly), use exact original wording from the validated summary.\n"
            f"- For non-significant findings, if wording is complex (>30 words) or redundant, simplify using standard radiological terminology from HIGH_FREQ_PHRASES, preserving clinical meaning.\n"
            f"- Include only findings relevant to the provided labels for each anatomical region; omit regions with no relevant findings.\n"
        )

        content = "\n\n".join([header, anatomical_keywords, instructions])
        super().__init__("user", content)

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            return json.dumps({"unhealthy_findings": self.findings})

        cleaned = re.sub(r'```json\n|\n```|```', '', model_output).strip()

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict) or "unhealthy_findings" not in parsed:
                raise ValueError("Missing 'unhealthy_findings' field")

            value = parsed["unhealthy_findings"]
            if not isinstance(value, dict):
                raise ValueError("'unhealthy_findings' must be a dictionary")

            for section in ReportGenerationProcess.POSITIVE_SECTIONS:
                if section not in value or not isinstance(value[section], list):
                    raise ValueError(f"Missing or invalid section '{section}' in unhealthy_findings")

            return json.dumps({"unhealthy_findings": value}, indent=2)

        except Exception:
            return json.dumps({"unhealthy_findings": self.findings})


class Step6GenerateNewReport(BaseStep):
    def __init__(self, step5_json: str, report_count: int = 1, label_type: str = "positive"):
        if not isinstance(step5_json, str):
            step5_json = json.dumps(self._get_empty_template(label_type))

        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if label_type not in ["positive", "negative"]:
            raise ValueError("label_type must be 'positive' or 'negative'.")

        try:
            templates_data = json.loads(step5_json)
            key = "unhealthy_findings" if label_type == "positive" else "healthy_findings"
            expected_sections = (
                ReportGenerationProcess.POSITIVE_SECTIONS if label_type == "positive"
                else ReportGenerationProcess.NEGATIVE_SECTIONS
            )
            if not isinstance(templates_data, dict) or key not in templates_data:
                raise ValueError(f"step5_json must include '{key}' field")
            if not all(section in templates_data[key] for section in expected_sections):
                raise ValueError(f"{key} must include sections: {', '.join(expected_sections)}")

            for section in expected_sections:
                for finding in templates_data[key][section]:
                    if not isinstance(finding, dict):
                        raise ValueError(f"Each finding in '{section}' must be a dictionary")
                    if "text" not in finding or "report" not in finding:
                        raise ValueError(f"Each finding in '{section}' must include 'text' and 'report'")

        except (json.JSONDecodeError, ValueError):
            templates_data = self._get_empty_template(label_type)

        content = self._build_guidelines_section(templates_data, label_type)
        super().__init__("user", content)
        self.label_type = label_type
        self.templates_data = templates_data

    def _get_empty_template(self, label_type: str) -> dict:
        sections = (
            ReportGenerationProcess.POSITIVE_SECTIONS if label_type == "positive"
            else ReportGenerationProcess.NEGATIVE_SECTIONS
        )
        return {
            "unhealthy_findings" if label_type == "positive" else "healthy_findings": {
                section: [] for section in sections
            }
        }

    def _build_guidelines_section(self, templates_data: dict, label_type: str) -> str:
        key = "unhealthy_findings" if label_type == "positive" else "healthy_findings"
        anatomical_order = (
            ", ".join(ReportGenerationProcess.POSITIVE_SECTIONS) if label_type == "positive"
            else ", ".join(ReportGenerationProcess.NEGATIVE_SECTIONS)
        )

        if label_type == "negative":
            guidelines = [
                f"- Parse the Step 5 JSON output (`{key}`) to generate a chest X-ray report, organized by anatomical order: {anatomical_order}.",
                f"- JSON to use:\n```json\n{json.dumps(templates_data, indent=2)}\n```",
                f"- Sentence Chain Matching Instructions:\n"
                f"  1. For each sentence in the Step 5 JSON (under `{key}`), check if it **exactly matches** the first sentence of any high-frequency sentence chain provided.\n"
                f"     → If yes, **output the entire matched sentence chain verbatim as the final report**, and **STOP immediately**. Do NOT process other JSON findings.\n"
                f"  2. If no exact match is found, check for sentence with **≥70% word overlap** and similar structure with any chain first sentence.\n"
                f"     → If such a match is found and terms are in similar order, **output the full matched sentence chain**, and **STOP**.\n"
                f"  3. If still no match:\n"
                f"     → Generate the report by parsing each anatomical region in order:\n"
                f"        - Use each `text` field in `{key}` as a full sentence.\n"
                f"        - If a section is empty, fallback to the **first sentence** in the most frequent sentence chain for that section.\n",
                f"- Output Rules:\n"
                f"  - Output as a **single paragraph**, NOT sectioned.\n"
                f"  - Do NOT include headers like 'Lungs:', 'Pleura:'.\n"
                f"  - Do NOT repeat findings from the same sentence in multiple places.\n"
                f"  - Do NOT introduce invented findings, speculative conclusions, or imaging technique notes.\n"
            ]
        else:
            guidelines = [
                f"- Parse the Step 5 JSON output (`{key}`) to generate a chest X-ray report, organized by anatomical order: {anatomical_order}.",
                f"- JSON to use:\n```json\n{json.dumps(templates_data, indent=2)}\n```",
                f"- Use the `text` field for each finding and associate it with its `report`.",
                f"- Omit anatomical regions with no findings.",
                f"- Merge redundant or overlapping descriptions within a region. Follow these rules:\n"
                f"   • **Simplify and combine** descriptions when they appear in multiple anatomical regions. For example:\n"
                f"     - If the same description appears in both 'heart' and 'mediastinum', combine them into one sentence like 'Mild cardiomegaly and widened mediastinum.'\n"
                f"   • Prefer the most clinically significant or severe phrase.\n"
                f"   • Combine modifiers when needed (e.g., 'stable moderate...' or 'increased and asymmetric...').\n"
                f"   • Remove contradictions (e.g., 'no effusion' + 'small effusion').\n"
                f"   • Collapse lobar repetition into unified descriptions (e.g., 'consolidation in both lower lobes' instead of listing each lobe separately).\n"
                f"   • Target ≤2 sentences per region; use semicolons for joining when appropriate.",
                f"- Use standard radiological terminology from HIGH_FREQ_PHRASES wherever possible.",
                f"- Output should be a **single paragraph**, no headers, no numbered sections."
            ]

        return "\n\n".join(guidelines)

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            raise ValueError("Step6GenerateNewReport: Model output is empty")
        cleaned = model_output.strip()

        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
        if match:
            cleaned = match.group(1).strip()

        if not cleaned:
            raise ValueError("Step6GenerateNewReport: Cleaned output is empty")
        return cleaned


class Step7AddFinishingTouches(BaseStep):
    def __init__(self, report_content: str, report_count: int = 1, label_type: str = "positive"):
        if not isinstance(report_content, str):
            report_content = ""
        if not report_content.strip():
            report_content = "<EMPTY REPORT>"

        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if label_type not in ["positive", "negative"]:
            raise ValueError("label_type must be 'positive' or 'negative'.")

        reference_section = self._build_reference_section(report_content, label_type)
        guidelines_section = self._build_guidelines_section(label_type, report_count)
        content = "\n\n".join([reference_section, guidelines_section])

        super().__init__("user", content)
        self.label_type = label_type

    def _build_reference_section(self, report_content: str, label_type: str) -> str:
        desc = "positive findings" if label_type == "positive" else "negative findings (no abnormalities)"
        return (
            f"Reference report from Step 6 ({desc}):\n"
            f"---\n"
            f"{report_content.strip()}\n"
            f"---\n"
        )

    def _build_guidelines_section(self, label_type: str, report_count: int) -> str:
        sections = (
            ReportGenerationProcess.POSITIVE_SECTIONS if label_type == "positive"
            else ReportGenerationProcess.NEGATIVE_SECTIONS
        )
        anatomical_order = ", ".join(sections)

        if label_type == "negative":
            guidelines = [
                f"- Refine the Step 6 report into a clinically accurate, **non-redundant**, and professional chest X-ray report covering anatomical order: {anatomical_order}.",
                f"- If the Step 6 output is based on a matched **sentence chain**, preserve that entire chain **without modification or expansion**.",
                f"- If not based on a chain, process section-wise as follows:\n"
                f"   • **Remove duplicate or semantically similar sentences** (e.g., 'The lungs are clear.' vs. 'Lungs are clear.').\n"
                f"   • **Merge overlapping findings** (e.g., 'heart size is normal' + 'cardiac silhouette unremarkable' → 'cardiac silhouette and heart size within normal limits').\n"
                f"   • **Omit generic, vague, or excessively long expressions** if more precise statements exist.\n"
                f"   • Avoid repeating phrases like 'Normal heart, lungs, hila...' across multiple sections — assign to **only one section** (e.g., heart_mediastinum).",
                f"- Do not introduce speculative language, invented findings, or invented anatomical regions.",
            ]
        else:
            guidelines = [
                f"- Refine the Step 6 report to retain **clinically significant findings** using standard radiological language.",
                f"- **Eliminate redundant statements**, e.g., avoid listing 'severe cardiomegaly' and 'heart size is borderline enlarged'.",
                f"- Merge modifiers into one clear sentence (e.g., 'stable moderate effusion').",
                f"- Remove contradictions (e.g., 'no effusion' vs. 'small effusion').",
                f"- **Merge overlapping findings**: If the same finding appears in multiple anatomical regions, combine it into a single, simplified sentence (e.g., 'Mild cardiomegaly and widened mediastinum').",
                f"- Use ≤2 sentences per region; use semicolons if necessary.",
                f"- Use terminology from Step 5 HIGH_FREQ_PHRASES where appropriate.",
            ]

        constraints = [
            "- Output must be **plain text** (not JSON or markdown).",
            "- Do NOT use code blocks or placeholders.",
            "- Do NOT invent new sections or findings.",
        ]

        return "\n\n".join([
            "Please refine the Step 6 chest X-ray report based on the following guidelines:",
            "\n".join(f"  {g}" for g in guidelines),
            "**Constraints**:\n" + "\n".join(f"  {c}" for c in constraints),
            "**Note**: The above are the instructions for Step 7. Reply with the complete refined report in plain text only. This will be used for Step 8 final output formatting."
        ])

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            raise ValueError("Step7AddFinishingTouches: Model output is empty")

        cleaned = re.sub(r'```(?:text)?\s*([\s\S]*?)\s*```', r'\1', model_output).strip()
        cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)
        cleaned = re.sub(r'\{[^{}]*?\}', '', cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            raise ValueError("Step7AddFinishingTouches: Cleaned output is empty")

        return cleaned


class Step8ReviewAndOutputJSON(BaseStep):
    def __init__(self, refined_report: str, report_count: int = 1, label_type: str = "positive"):
        if not isinstance(refined_report, str):
            refined_report = "<EMPTY REPORT>"
        if not refined_report.strip():
            refined_report = "<EMPTY REPORT>"

        if not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer.")
        if label_type not in ["positive", "negative"]:
            raise ValueError("label_type must be 'positive' or 'negative'.")

        ref_section = self._build_reference_section(refined_report, label_type)
        guide_section = self._build_guidelines_section(report_count, label_type)
        content = "\n\n".join([ref_section, guide_section])

        super().__init__("user", content)
        self.label_type = label_type

    def _build_reference_section(self, text: str, label_type: str) -> str:
        desc = "positive findings" if label_type == "positive" else "negative findings (no abnormalities)"
        return f"Refined report from Step 7 ({desc}):\n\n---\n{text.strip()}\n---"

    def _build_guidelines_section(self, report_count: int, label_type: str) -> str:
        review_bullets = [
            "- Check the Step 7 report for irrelevant information, including Impression sections, parentheses, imaging technique (e.g., 'PA and lateral views'), image quality notes (e.g., 'well-inspired'), or clinical recommendations (e.g., 'recommend follow-up CT').",
            "- Remove redundant verbs (e.g., 'is identified', 'are present') for conciseness (e.g., 'No acute cardiopulmonary abnormality').",
            "- Eliminate semantically or syntactically **repetitive sentences** across anatomical regions.",
            "- Avoid using **multiple paraphrases** of the same concept (e.g., 'The lungs remain clear.' and 'Lungs are clear.').",
            "- If a sentence appears in multiple regions (e.g., 'Normal heart, lungs, hila, mediastinum, and pleural surfaces.'), retain it **only once** in the most appropriate section.",
            "- If Step 6 was based on a high-frequency sentence chain, do not add additional redundant findings unless clearly distinct.",
            "- **For positive findings**, merge overlapping findings across anatomical regions. For example, if a description of 'cardiomegaly' appears in both the 'heart' and 'mediastinum' sections, combine them into one statement (e.g., 'Mild cardiomegaly and widened mediastinum')."
        ]

        constraints_bullets = [
            "- Include **only** information from the Step 7 report.",
            "- Return the final version of the report strictly in this JSON format:\n" +
            "{\n" +
            '  "Final Report": "Your complete, standard chest X-ray report here."\n' +
            "}",
            "- The report must be written as a **single paragraph**, no line breaks or Markdown formatting, and the JSON must be a **single-line string**."
        ]

        sections = [
            "As a seasoned radiologist, conduct a final review of the Step 7 chest X-ray report to ensure it contains no irrelevant or redundant information and meets radiological standards:",
            "1. **Review Rules**:\n" + "\n".join(f"   {b}" for b in review_bullets),
            "2. **Constraints**:\n" + "\n".join(f"   {c}" for c in constraints_bullets)
        ]
        return "\n\n".join(sections)

    def process_output(self, model_output: Optional[str]) -> Optional[str]:
        if not model_output:
            raise ValueError("Step8ReviewAndOutputJSON: Model output is empty")

        json_match = re.search(r'```json\n([\s\S]*?)\n```', model_output)
        cleaned = json_match.group(1) if json_match else re.sub(r'```[\s\S]*?```', '', model_output).strip()

        if not cleaned:
            raise ValueError("Step8ReviewAndOutputJSON: Cleaned output is empty")

        try:
            obj = json.loads(cleaned)
            if not isinstance(obj, dict) or "Final Report" not in obj:
                raise ValueError("JSON must be an object with 'Final Report' field")
            if not obj["Final Report"] or not isinstance(obj["Final Report"], str):
                raise ValueError("'Final Report' must be a non-empty string")

            final_report = obj["Final Report"].strip()
            final_report = re.sub(r'\s+', ' ', final_report)  # 合并多余空格

            return json.dumps({"Final Report": final_report}, ensure_ascii=False)

        except json.JSONDecodeError as e:
            raise ValueError(f"Step8ReviewAndOutputJSON: Invalid JSON format: {e}")



class ReportGenerationProcess:
    VALID_LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
        "Lung Opacity", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices"
    ]

    NEGATIVE_SECTIONS = ["heart_mediastinum", "lungs", "pleura", "bones"]
    POSITIVE_SECTIONS = ["heart", "mediastinum", "lungs", "pleura", "bones"]

    def __init__(self, reference_reports, client, report_count: int = None, labels: list = None, report_id: str = None):
        self.reference_reports = reference_reports
        self.client = client
        self.report_id = report_id or "unknown_id"

        if report_count is None:
            self.report_count = len(reference_reports) if isinstance(reference_reports, list) else 1
        elif not isinstance(report_count, int) or report_count <= 0:
            raise ValueError("report_count must be a positive integer")
        else:
            self.report_count = report_count

        if not labels or not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
            labels = ["No Finding"]
        if any(lbl not in self.VALID_LABELS for lbl in labels):
            raise ValueError(f"Invalid CheXpert labels: {labels}")
        self.labels = labels

        self.conversation = []
        self.step_outputs = {}

    def get_reference_reports(self):
        if isinstance(self.reference_reports, str):
            return self.reference_reports
        elif isinstance(self.reference_reports, list):
            return "\n\n".join(self.reference_reports)
        raise ValueError("Invalid reference_reports format")

    def get_default_step5_output(self, is_negative: bool) -> str:
        sections = self.NEGATIVE_SECTIONS if is_negative else self.POSITIVE_SECTIONS
        key = "healthy_findings" if is_negative else "findings"
        return json.dumps({key: {section: [] for section in sections}})

    def execute_step(self, step: 'BaseStep') -> Optional[str]:
        step_name = step.__class__.__name__

        if "{reference_reports}" in step.content:
            try:
                step.content = step.content.replace("{reference_reports}", self.get_reference_reports())
            except Exception:
                if step_name == "Step3SummarizeReports":
                    return "No findings extracted due to reference report formatting error."
                return None

        try:
            report_id = getattr(self, 'report_id', 'unknown_id')
            output = step.process(
                self.conversation,
                self.client,
                step=step_name,
                report_id=report_id if step_name == "Step8ReviewAndOutputJSON" else None
            )
            if not output:
                raise ValueError("Step output is empty")
            self.step_outputs[step_name] = output
            return output
        except Exception:
            fallback = self.get_default_step5_output(self.labels == ["No Finding"]) if step_name in [
                "Step4ValidateSummary", "Step5SetSystemPositivePromptTemplates", "Step5SetSystemNegativePromptTemplates"
            ] else ""
            self.step_outputs[step_name] = fallback
            return fallback

    def run(self) -> str:
        if not self.reference_reports:
            return None

        self.conversation.clear()
        self.step_outputs.clear()

        steps = [
            (Step1DefineRole, {}),
            (Step2ProvideReferenceReports, {"report_count": self.report_count, "labels": self.labels}),
            (Step3SummarizeReports, lambda: {
                "previous_report_content": self.step_outputs.get("Step2ProvideReferenceReports", ""),
                "report_count": self.report_count,
                "labels": self.labels
            }),
            (Step4ValidateSummary, lambda: {
                "previous_summary": self.step_outputs.get("Step3SummarizeReports", ""),
                "report_count": self.report_count,
                "labels": self.labels
            }),
        ]

        for step_class, kwargs in steps:
            kwargs_val = kwargs() if callable(kwargs) else kwargs
            output = self.execute_step(step_class(**kwargs_val))
            if output is None:
                return None

        is_negative = self.labels == ["No Finding"]
        step5_class = Step5SetSystemNegativePromptTemplates if is_negative else Step5SetSystemPositivePromptTemplates
        step5_kwargs = {
            "previous_report_content": self.step_outputs.get("Step4ValidateSummary", ""),
            "report_count": self.report_count,
            "labels": self.labels
        }

        step5_output = self.execute_step(step5_class(**step5_kwargs))
        step5_output = re.sub(r'```json\n|\n```|```', '', step5_output).strip()

        try:
            parsed_output = json.loads(step5_output)
            key = "healthy_findings" if is_negative else "unhealthy_findings"
            expected_sections = self.NEGATIVE_SECTIONS if is_negative else self.POSITIVE_SECTIONS

            if not isinstance(parsed_output, dict) or key not in parsed_output or not all(sec in parsed_output[key] for sec in expected_sections):
                step5_output = self.get_default_step5_output(is_negative)
        except json.JSONDecodeError:
            step5_output = self.get_default_step5_output(is_negative)

        self.step_outputs["Step5"] = step5_output

        final_steps = [
            (Step6GenerateNewReport, lambda: {
                "step5_json": self.step_outputs.get("Step5", ""),
                "report_count": self.report_count,
                "label_type": "negative" if is_negative else "positive"
            }),
            (Step7AddFinishingTouches, lambda: {
                "report_content": self.step_outputs.get("Step6GenerateNewReport", ""),
                "report_count": self.report_count,
                "label_type": "negative" if is_negative else "positive"
            }),
            (Step8ReviewAndOutputJSON, lambda: {
                "refined_report": self.step_outputs.get("Step7AddFinishingTouches", ""),
                "report_count": self.report_count,
                "label_type": "negative" if is_negative else "positive"
            }),
        ]

        for step_class, kwargs in final_steps:
            kwargs_val = kwargs() if callable(kwargs) else kwargs
            output = self.execute_step(step_class(**kwargs_val))
            if output is None:
                return None

        final_output = self.step_outputs.get("Step8ReviewAndOutputJSON", "")
        return extract_json(final_output) if final_output else None