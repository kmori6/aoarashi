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
        enc_token_list, enc_length_list, dec_token_list, dec_length_list, tgt_token_list = [], [], [], [], []
        for sample in sample_list:
            enc_token = self.tokenizer.encode(sample["src_text"])
            enc_token_list.append(torch.tensor(enc_token, dtype=torch.long))
            enc_length_list.append(len(enc_token))
            base_token = self.tokenizer.encode(sample["tgt_text"])
            dec_token = [self.bos_token_id] + base_token
            tgt_token = base_token + [self.eos_token_id]
            dec_token_list.append(torch.tensor(dec_token, dtype=torch.long))
            tgt_token_list.append(torch.tensor(tgt_token, dtype=torch.long))
            dec_length_list.append(len(dec_token))
        batch = {
            "enc_token": pad_sequence(enc_token_list, True, self.pad_token_id),
            "enc_token_length": torch.tensor(enc_length_list, dtype=torch.long),
            "dec_token": pad_sequence(dec_token_list, True, self.pad_token_id),
            "dec_token_length": torch.tensor(dec_length_list, dtype=torch.long),
            "tgt_token": pad_sequence(tgt_token_list, True, self.ignore_token_id),
        }
        return batch
