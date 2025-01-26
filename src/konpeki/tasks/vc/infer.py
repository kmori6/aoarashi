import argparse
import os

import torch
import torchaudio
import yaml

from konpeki.tasks.vc.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--wav_path", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    model = Model(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device("cpu"))["model_state_dict"])
    model.eval()
    wav, sample_rate = torchaudio.load(args.wav_path)
    assert sample_rate == config["model"]["sample_rate"]
    wav = model.synthesize(wav.squeeze())  # (1, 1, sample)
    torchaudio.save(f"{args.out_dir}/output.wav", wav[:, 0, :], sample_rate=config["model"]["sample_rate"])


if __name__ == "__main__":
    main()
