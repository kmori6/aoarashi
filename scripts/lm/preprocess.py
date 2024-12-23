import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--valid_rate", type=float, default=0.05)
    parser.add_argument("--test_rate", type=float, default=0.05)
    args = parser.parse_args()
    with open(f"{args.data_dir}/librispeech-lm-norm.txt", "r") as f:
        lines = f.readlines()
    num_samples = len(lines)
    num_train_samples = int(num_samples * (1 - args.valid_rate - args.test_rate))
    subsets = {
        "train": lines[:num_train_samples],
        "valid": lines[num_train_samples : int(num_samples * (1 - args.test_rate))],
        "test": lines[int(num_samples * (1 - args.test_rate)) :],
    }
    text_list = []
    for key in subsets.keys():
        dic = {}
        subset = subsets[key]
        for i, line in tqdm(enumerate(subset), total=len(subset), desc=key):
            sample_id = str(i).zfill(8)
            text = line.strip().lower()
            if len(text) == 0:
                continue
            dic[sample_id] = {"text": text}
            if key == "train":
                text_list.append(text + "\n")
        os.makedirs(Path(args.out_dir), exist_ok=True)
        with open(Path(args.out_dir) / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.out_dir) / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(text_list)


if __name__ == "__main__":
    main()
