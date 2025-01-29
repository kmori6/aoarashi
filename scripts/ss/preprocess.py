import argparse
import json
import os
from glob import glob
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="min")
    parser.add_argument("--sample_rate", type=str, default="8k")
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=5.0)
    args = parser.parse_args()
    for key in ["train", "dev", "test"]:
        dic = {}
        for csv_path in glob(
            f"{args.data_dir}/Libri2Mix/wav{args.sample_rate}/{args.mode}/metadata/mixture_{key}*_mix_both.csv"
        ):
            df = pd.read_csv(csv_path)
            for _, row in tqdm(df.iterrows(), total=len(df)):
                utt_id = row["mixture_ID"]
                mix_audio_path = row["mixture_path"]
                src1_audio_path = row["source_1_path"]
                src2_audio_path = row["source_2_path"]
                audio_length = int(row["length"])
                audio_info = torchaudio.info(mix_audio_path)
                assert audio_info.num_channels == 1
                assert audio_length == audio_info.num_frames
                if audio_length < args.min_duration * 8000 or args.max_duration * 8000 < audio_length:
                    continue
                sample = {
                    "mix_audio_path": mix_audio_path,
                    "src1_audio_path": src1_audio_path,
                    "src2_audio_path": src2_audio_path,
                    "audio_length": audio_length,
                }
                dic[utt_id] = sample
        os.makedirs(Path(args.out_dir), exist_ok=True)
        file_name = f"{key}.json" if key != "dev" else "valid.json"
        with open(Path(args.out_dir) / file_name, "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
