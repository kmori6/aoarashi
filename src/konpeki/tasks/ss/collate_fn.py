from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


class CollateFn:
    def __init__(self, stride: int):
        self.stride = stride

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        mix_audio_list, mix_audio_length_list, src1_audio_list, src2_audio_list = [], [], [], []
        for sample in sample_list:
            mix_audio = sample["mix_audio"]
            src1_audio = sample["src1_audio"]
            src2_audio = sample["src2_audio"]
            assert len(mix_audio) == len(src1_audio)
            assert len(mix_audio) == len(src2_audio)
            residual = len(mix_audio) % self.stride
            if residual > 0:
                # L = len(mix_audio) / stride * stride
                mix_audio = mix_audio[:-residual]
                src1_audio = src1_audio[:-residual]
                src2_audio = src2_audio[:-residual]
            mix_audio_list.append(mix_audio)
            mix_audio_length_list.append(len(mix_audio))
            src1_audio_list.append(src1_audio)
            src2_audio_list.append(src2_audio)
        batch = {
            "mix_audio": pad_sequence(mix_audio_list, True, 0.0),
            "mix_audio_length": torch.tensor(mix_audio_length_list, dtype=torch.long),
            "src1_audio": pad_sequence(src1_audio_list, True, 0.0),
            "src2_audio": pad_sequence(src2_audio_list, True, 0.0),
        }
        return batch
