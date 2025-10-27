""" Johan C
Collect leaks from predictions and judgements.

Output JSON format (example):
{
  "main1": {
    "final_action": "Thought: ...",           # Text of the prediction or answer.
    "leak_info": true,                        # Boolean indicating whether it leaked information.
    "secret_judgment": [                      # List of [info, is_secret] pairs for judgment.
      ["Mr. Smithson is accused of embezzling...", true],
      ...
    ]
  },
  ...
}

"""
import csv
import json
import sys
import os
import argparse

def write_json(leaks_dict, outpath):
    print(f"Writing leaks to: {outpath}")
    with open(outpath, "w", encoding="utf-8") as out_f:
        json.dump(leaks_dict, out_f, ensure_ascii=False, indent=2)

def write_csv(leaks_dict, outpath):
    print(f"Writing leaks to: {outpath}")
    if not leaks_dict:
        print("No leaks to write to CSV.")
        return
    # Get fieldnames from any dict value (all have the same structure)
    first_example = next(iter(leaks_dict.values()))
    fields = ["name"] + list(first_example.keys())
    def serialize(name, entry):
        row = entry.copy()
        row["name"] = name
        row["secret_judgment"] = json.dumps(row["secret_judgment"], ensure_ascii=False)
        return row
    with open(outpath, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()
        for name, entry in leaks_dict.items():
            writer.writerow(serialize(name, entry))

def main():
    parser = argparse.ArgumentParser(description="Collect leaks from predictions and judgements.")
    parser.add_argument("--predictions", type=str, default="data_pipeline/predictions/qwen2-7b-instr0.csv", help="Path to the input predictions CSV file.")
    parser.add_argument("--judgements", type=str, default="data_pipeline/predictions/qwen2-7b-instr0.json", help="Path to the input judgements JSON file.")
    parser.add_argument("--leaks_json", type=str, default="data_pipeline/leaks/qwen2-7b-instr0.json", help="Path to write the output JSON with leaks.")
    parser.add_argument("--leaks_csv", type=str, default="data_pipeline/leaks/qwen2-7b-instr0.csv", help="Path to write the output CSV with leaks.")
    args, unknown = parser.parse_known_args()

    PREDICTIONS_PATH = args.predictions
    JUDGEMENTS_PATH = args.judgements
    OUTPUT_PATH_JSON = args.leaks_json
    OUTPUT_PATH_CSV = args.leaks_csv

    # Allow user to specify output format (json/csv/both) via command line argument
    out_format = "json"
    for a in unknown:
        if a.lower() in {"json", "csv", "both"}:
            out_format = a.lower()
            break

    if out_format not in {"json", "csv", "both"}:
        if len(unknown) > 0:
            print(f"Unknown format: {unknown}, valid options are json/csv/both.")
            sys.exit(1)

    print(f"Loading JSON data from: {JUDGEMENTS_PATH}")
    with open(JUDGEMENTS_PATH, "r", encoding="utf-8") as jf:
        json_data = json.load(jf)

    leaks_dict = {}
    print(f"Reading CSV data from: {PREDICTIONS_PATH}")
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            name = row["name"]
            if name in json_data and json_data[name].get("leak_info", False):
                leaks_dict[name] = {
                    "final_action": row["final_action"],
                    "leak_info": json_data[name].get("leak_info"),
                    "secret_judgment": json_data[name].get("secret_judgment"),
                }

    print(f"Total leaks found: {len(leaks_dict)}")
    if out_format == "json":
        write_json(leaks_dict, OUTPUT_PATH_JSON)
    elif out_format == "csv":
        write_csv(leaks_dict, OUTPUT_PATH_CSV)
    elif out_format == "both":
        write_json(leaks_dict, OUTPUT_PATH_JSON)
        write_csv(leaks_dict, OUTPUT_PATH_CSV)
    print("Done.")

if __name__ == "__main__":
    main()
