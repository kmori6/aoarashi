import argparse
import os
from pathlib import Path

from g2p_en import G2p
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(Path(args.out_dir), exist_ok=True)
    with open(args.text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    g2p = G2p()
    token_list = ["<pad>", "<unk>"]
    phoneme_set = set()
    for text in tqdm(lines):
        # phonemes without word separators
        phones = [p for p in g2p(text.strip()) if p != " "]
        phoneme_set |= set(phones)
    with open(Path(args.out_dir) / "token.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(token_list + sorted(phoneme_set)))


if __name__ == "__main__":
    main()
