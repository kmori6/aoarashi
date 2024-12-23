import argparse
import os
from pathlib import Path

from sentencepiece import SentencePieceTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_name", type=str, default="tokenizer")
    args = parser.parse_args()
    os.makedirs(Path(args.out_dir), exist_ok=True)
    SentencePieceTrainer.train(
        input=args.text_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        model_prefix=f"{args.out_dir}/{args.model_name}",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


if __name__ == "__main__":
    main()
