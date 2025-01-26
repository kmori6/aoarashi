import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commonvoice_data_dir", type=str, required=True)
    parser.add_argument("--covost2_data_dir", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="ja")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    args = parser.parse_args()
    covost2_df = pd.read_csv(
        Path(args.covost2_data_dir) / f"covost_v2.{args.src_lang}_{args.tgt_lang}.tsv",
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )
    text_list = []
    for key in ["train", "dev", "test"]:
        cv_df = pd.read_csv(
            Path(args.commonvoice_data_dir) / f"{key}.tsv",
            sep="\t",
            header=0,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )
        df = covost2_df[covost2_df["path"].isin(cv_df["path"])]
        assert df["split"].unique() == key
        assert len(df) == len(cv_df)
        dic = {}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            utt_id = row["path"].split(".")[0]
            audio_path = f"{args.commonvoice_data_dir}/clips/{row['path']}"
            # We skip the audio file if it is empty
            if os.path.getsize(audio_path) == 0:
                continue
            text = row["translation"].strip().lower()
            audio_info = torchaudio.info(audio_path)
            assert audio_info.num_channels == 1
            assert audio_info.sample_rate == 48000
            audio_length = audio_info.num_frames
            if key in ["train", "valid"] and (
                audio_length < args.min_duration * 48000 or args.max_duration * 48000 < audio_length
            ):
                continue
            sample = {"audio_path": audio_path, "audio_length": audio_length, "text": text}
            dic[utt_id] = sample
            if key == "train":
                text_list.append(text + "\n")
        os.makedirs(Path(args.out_dir), exist_ok=True)
        file_name = f"{key}.json" if key != "dev" else "valid.json"
        with open(Path(args.out_dir) / file_name, "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.out_dir) / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(text_list)


if __name__ == "__main__":
    main()
