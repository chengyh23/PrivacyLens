import os
import json
import random
import numpy as np

# Refined logic to support both 2-way (train/test) and 3-way (train/val/test) splits more robustly.

def split_privacylens(json_path, out_prefix="main_data", split_ratio=[0.8, 0.1, 0.1]):
    """
    Splits main_data.json into train/validation/test sets for future ML work.

    Args:
        json_path (str): Path to the main_data.json file.
        out_prefix (str): Prefix for output files.
        split_ratio (list): [train, val, test] or [train, test] as proportions.
    """

    split_ratio = list(split_ratio)
    n = len(split_ratio)
    assert n in [2, 3], "split_ratio must be of length 2 (train/test) or 3 (train/val/test)"
    total = sum(split_ratio)
    split_ratio = [r/total for r in split_ratio]  # Normalize if sum != 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)
    indices = list(range(len(data)))
    random.shuffle(indices)

    num_total = len(data)
    if n == 3:
        train_ratio, val_ratio, test_ratio = split_ratio
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        num_test = num_total - num_train - num_val  # Remainder goes to test
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]
    elif n == 2:
        train_ratio, test_ratio = split_ratio
        num_train = int(num_total * train_ratio)
        num_test = num_total - num_train
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        val_indices = []

    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]
    val = [data[i] for i in val_indices] if val_indices else []

    out_root = os.path.dirname(json_path)
    with open(f"{out_root}/{out_prefix}_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    if n == 3:
        with open(f"{out_root}/{out_prefix}_val.json", "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
    with open(f"{out_root}/{out_prefix}_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"Total: {num_total} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Wrote: {out_prefix}_train.json"
          + (f", {out_prefix}_val.json" if n == 3 else "")
          + f", {out_prefix}_test.json")



if __name__ == "__main__":
    split_privacylens("data/main_data.json", split_ratio=[0.8,0.2])
