import argparse
import os

import torch
import torchaudio
import yaml

from aoarashi.tasks.tts.model import Model
from aoarashi.utils.tokenizer import PhonemeTokenizer, normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--token_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_text", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = PhonemeTokenizer(args.token_path)
    model = Model(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device("cpu"))["model_state_dict"])
    model.eval()
    token = tokenizer.encode(normalize_text(args.input_text))
    wav = model.synthesize(torch.tensor(token, dtype=torch.long))  # (1, 1, sample)
    torchaudio.save(f"{args.out_dir}/output.wav", wav[:, 0, :], sample_rate=22050)


if __name__ == "__main__":
    main()
