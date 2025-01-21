import argparse
import json
import os
from pathlib import Path

import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--src_spk", type=str, default="bdl")
    parser.add_argument("--tgt_spk", type=str, default="slt")
    parser.add_argument("--valid_samples", type=int, default=50)
    parser.add_argument("--test_samples", type=int, default=50)
    args = parser.parse_args()

    src_base_dir = Path(args.data_dir) / f"cmu_us_{args.src_spk}_arctic" / "wav"
    tgt_base_dir = Path(args.data_dir) / f"cmu_us_{args.tgt_spk}_arctic" / "wav"
    src_files = sorted(os.listdir(src_base_dir))
    tgt_files = sorted(os.listdir(tgt_base_dir))
    dic = {"train": {}, "valid": {}, "test": {}}
    for i, (src_file, tgt_file) in tqdm(enumerate(zip(src_files, tgt_files)), total=len(src_files)):
        src_audio_path = src_base_dir / src_file
        tgt_audio_path = tgt_base_dir / tgt_file
        src_utt_id = src_file.split(".")[0]
        tgt_utt_id = tgt_file.split(".")[0]
        assert src_utt_id == tgt_utt_id, f"{src_utt_id} != {tgt_utt_id}"
        audio_info = torchaudio.info(src_audio_path)
        assert audio_info.num_channels == 1
        assert audio_info.sample_rate == 16000
        audio_length = audio_info.num_frames
        sample = {
            "src_audio_path": str(src_audio_path),
            "tgt_audio_path": str(tgt_audio_path),
            "audio_length": audio_length,
        }
        if i < len(src_files) - args.valid_samples - args.test_samples:
            dic["train"][src_utt_id] = sample
        elif i < len(src_files) - args.test_samples:
            dic["valid"][src_utt_id] = sample
        else:
            dic["test"][src_utt_id] = sample

    os.makedirs(Path(args.out_dir), exist_ok=True)
    for key in ["train", "valid", "test"]:
        with open(Path(args.out_dir) / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic[key], f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
