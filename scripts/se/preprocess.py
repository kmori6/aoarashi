import argparse
import json
import os
from glob import glob
from pathlib import Path

import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--valid_spks", type=int, default=2)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    args = parser.parse_args()
    for key in ["train", "valid", "test"]:
        if key in ["train", "valid"]:
            base_dir = "trainset_28spk"
        else:
            base_dir = "testset"
        spks = sorted(
            set([wav.split("/")[-1].split("_")[0] for wav in glob(f"{args.data_dir}/clean_{base_dir}_wav/*.wav")])
        )
        if key == "train":
            spks = spks[: -args.valid_spks]
        elif key == "valid":
            spks = spks[-args.valid_spks :]
        dic = {}
        for spk in tqdm(spks):
            for clean_wav, noisy_wav in zip(
                glob(f"{args.data_dir}/clean_{base_dir}_wav/{spk}_*.wav"),
                glob(f"{args.data_dir}/noisy_{base_dir}_wav/{spk}_*.wav"),
            ):
                utt_id = clean_wav.split("/")[-1].split(".")[0]
                clean_info = torchaudio.info(clean_wav)
                noisy_info = torchaudio.info(noisy_wav)
                assert clean_info.num_channels == 1
                assert noisy_info.num_channels == 1
                assert clean_info.sample_rate == 48000
                assert noisy_info.sample_rate == 48000
                clean_length = clean_info.num_frames
                noisy_length = noisy_info.num_frames
                assert clean_length == noisy_length
                if clean_length < args.min_duration * 48000 or args.max_duration * 48000 < clean_length:
                    continue
                sample = {
                    "clean_audio_path": clean_wav,
                    "noisy_audio_path": noisy_wav,
                    "audio_length": clean_length,
                }
                dic[utt_id] = sample
        os.makedirs(Path(args.out_dir), exist_ok=True)
        with open(Path(args.out_dir) / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
