import json
from typing import Any

import torchaudio
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        self.sample_list = sorted(data_dict.items(), key=lambda x: x[1]["audio_length"], reverse=True)

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, sample_dict = self.sample_list[index]
        sample = {
            "src_audio": torchaudio.load(sample_dict["src_audio_path"])[0].squeeze(),
            "tgt_audio": torchaudio.load(sample_dict["tgt_audio_path"])[0].squeeze(),
        }
        return sample
