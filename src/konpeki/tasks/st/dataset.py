import json
from typing import Any

import torchaudio
from torch.utils.data import Dataset as BaseDataset
from torchaudio.functional import resample


class Dataset(BaseDataset):
    def __init__(self, json_path: str, sample_rate: int = 16000):
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        self.sample_rate = sample_rate
        self.sample_list = sorted(data_dict.items(), key=lambda x: x[1]["audio_length"], reverse=True)

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, sample_dict = self.sample_list[index]
        audio, sample_rate = torchaudio.load(sample_dict["audio_path"])
        assert sample_rate != self.sample_rate
        audio = resample(audio, sample_rate, self.sample_rate)
        sample = {"audio": audio.squeeze(), "text": sample_dict["text"]}
        return sample
