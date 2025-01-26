from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from konpeki.utils.tokenizer import Tokenizer


class CollateFn:
    def __init__(
        self, tokenizer: Tokenizer, bos_token_id: int, eos_token_id: int, pad_token_id: int, ignore_token_id: int = -100
    ):
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_list, target_list = [], []
        for sample in sample_list:
            base_token = self.tokenizer.encode(sample["text"])
            token = [self.bos_token_id] + base_token
            target = base_token + [self.eos_token_id]
            token_list.append(torch.tensor(token, dtype=torch.long))
            target_list.append(torch.tensor(target, dtype=torch.long))
        batch = {
            "token": pad_sequence(token_list, True, self.pad_token_id),
            "target": pad_sequence(target_list, True, self.ignore_token_id),
        }
        return batch
