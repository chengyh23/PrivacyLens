""" Johan C
Collect corrections from predictions and judgements.

Output JSON format (example):
{
  "main1": {
    "chosen": "<the chosen correction for this case as a string>",
    "rejected": "<the original leaked/unsafe prediction for this case as a string>",
    "leak_info_chosen": false,
    "secret_judgment_chosen": [
      ["Sensitive info item 1", false],
      ["Sensitive info item 2", false]
      //...
    ],
    "leak_info_rejected": true,
    "secret_judgment_rejected": [
      ["Sensitive info item 1", true],
      ["Sensitive info item 2", false]
      //...
    ]
  },
  //...
}

"""
import os
import json
import csv
import argparse

def parse_csv_final_actions(filepath):
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            final_action = row["final_action"].strip()
            results[name] = final_action
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--leaks-json-path',
        type=str,
        default="data_pipeline/leaks/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.json",
        help='Path to leaks json file (old predictions with judgments etc)'
    )
    parser.add_argument(
        '--pred-json-path',
        type=str,
        default="data_pipeline/predictions2/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.json",
        help='Path to new predictions json file (with leak_info/judgment)'
    )
    parser.add_argument(
        '--pred-csv-path',
        type=str,
        default="data_pipeline/predictions2/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.csv",
        help='Path to new predictions CSV file (with final_action column)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default="data_pipeline/corrections/collect_corrections_output.json",
        help='Directory to write the output json to'
    )
    args = parser.parse_args()

    leaks_json_path = args.leaks_json_path
    pred_json_path = args.pred_json_path
    pred_csv_path = args.pred_csv_path

    # Load leaks, predictions, and CSV
    with open(leaks_json_path, "r", encoding="utf-8") as f:
        leaks_cases = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        judgments_2 = json.load(f)
    predictions_2 = parse_csv_final_actions(pred_csv_path)

    output_cases = {}

    for case_name in leaks_cases:
        pred_case = judgments_2.get(case_name, {})
        # Only select where prediction has leak_info: False
        if pred_case.get("leak_info") is not False:
            continue

        # "chosen" comes from prediction, "rejected" is leaks
        chosen_final_action = predictions_2.get(case_name) or pred_case.get("final_action")
        rejected_final_action = leaks_cases[case_name].get("final_action")

        chosen_secret_judgment = pred_case.get("secret_judgment")
        rejected_secret_judgment = leaks_cases[case_name].get("secret_judgment")

        chosen_leak_info = pred_case.get("leak_info")
        rejected_leak_info = leaks_cases[case_name].get("leak_info")

        output_cases[case_name] = {
            "chosen": chosen_final_action,
            "rejected": rejected_final_action,
            "leak_info_chosen": chosen_leak_info,
            "secret_judgment_chosen": chosen_secret_judgment,
            "leak_info_rejected": rejected_leak_info,
            "secret_judgment_rejected": rejected_secret_judgment
        }

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_cases, f, indent=2, ensure_ascii=False)
    print(f"Corrections written to {output_path} ({len(output_cases)} cases).")

if __name__ == "__main__":
    main()
