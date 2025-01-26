from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from konpeki.utils.tokenizer import Tokenizer


class CollateFn:
    def __init__(self, tokenizer: Tokenizer, blank_token_id: int):
        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_list, audio_length_list, token_list, target_list, target_length_list = [], [], [], [], []
        for sample in sample_list:
            audio_list.append(sample["audio"])
            audio_length_list.append(len(sample["audio"]))
            target = self.tokenizer.encode(sample["text"])
            token = [self.blank_token_id] + target
            token_list.append(torch.tensor(token, dtype=torch.long))
            target_list.append(torch.tensor(target, dtype=torch.long))
            target_length_list.append(len(target))
        batch = {
            "audio": pad_sequence(audio_list, True, 0.0),
            "audio_length": torch.tensor(audio_length_list, dtype=torch.long),
            "token": pad_sequence(token_list, True, self.blank_token_id),
            "target": pad_sequence(target_list, True, self.blank_token_id),
            "target_length": torch.tensor(target_length_list, dtype=torch.long),
        }
        return batch
