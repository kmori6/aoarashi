import json
from typing import Any

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        self.sample_list = sorted(data_dict.items(), key=lambda x: len(x[1]["text"]), reverse=True)

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, sample_dict = self.sample_list[index]
        sample = {"text": sample_dict["text"]}
        return sample
