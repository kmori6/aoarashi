from typing import Any

import torch
from librosa.util import normalize
from torch.nn.utils.rnn import pad_sequence


class CollateFn:
    def __init__(self, hop_length: int = 256):
        self.hop_length = hop_length

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        src_audio_list, src_audio_length, tgt_audio_list, tgt_audio_length = [], [], [], []
        for sample in sample_list:
            # normalize audio volume following to HiFi-GAN manner for tanh output
            src_audio = torch.from_numpy(0.95 * normalize(sample["src_audio"].numpy())).to(torch.float32)
            tgt_audio = torch.from_numpy(0.95 * normalize(sample["tgt_audio"].numpy())).to(torch.float32)
            assert -1.0 < src_audio.min() and src_audio.max() < 1.0 and -1.0 < tgt_audio.min() and tgt_audio.max() < 1.0
            residual = len(tgt_audio) % self.hop_length
            if residual > 0:
                # L = len(tgt_audio) / hop_length * hop_length
                tgt_audio = tgt_audio[:-residual]
            src_audio_list.append(src_audio)
            src_audio_length.append(len(src_audio))
            tgt_audio_list.append(tgt_audio)
            tgt_audio_length.append(len(tgt_audio))
        batch = {
            "src_audio": pad_sequence(src_audio_list, True, 0.0),
            "src_audio_length": torch.tensor(src_audio_length, dtype=torch.long),
            "tgt_audio": pad_sequence(tgt_audio_list, True, 0.0),
            "tgt_audio_length": torch.tensor(tgt_audio_length, dtype=torch.long),
        }
        return batch
