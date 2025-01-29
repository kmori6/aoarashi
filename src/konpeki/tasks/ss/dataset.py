import json

import torch
import torchaudio
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, json_path: str, sample_rate: int = 8000):
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        self.sample_list = sorted(data_dict.items(), key=lambda x: x[1]["audio_length"], reverse=True)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _, sample_dict = self.sample_list[index]
        mix_audio, sample_rate = torchaudio.load(sample_dict["mix_audio_path"])
        assert sample_rate == self.sample_rate
        src1_audio, _ = torchaudio.load(sample_dict["src1_audio_path"])
        src2_audio, _ = torchaudio.load(sample_dict["src2_audio_path"])
        sample = {
            "mix_audio": mix_audio.squeeze(),
            "src1_audio": src1_audio.squeeze(),
            "src2_audio": src2_audio.squeeze(),
        }
        return sample
