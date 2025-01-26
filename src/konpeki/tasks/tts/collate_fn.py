from typing import Any

import torch
from librosa.util import normalize
from torch.nn.utils.rnn import pad_sequence

from konpeki.utils.tokenizer import PhonemeTokenizer


class CollateFn:
    def __init__(self, tokenizer: PhonemeTokenizer, hop_length: int = 256, pad_token_id: int = 0):
        self.tokenizer = tokenizer
        self.hop_length = hop_length
        self.pad_token_id = pad_token_id

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_list, audio_length, token_list, token_length_list = [], [], [], []
        for sample in sample_list:
            # normalize audio volume following to HiFi-GAN manner for tanh output
            audio = torch.from_numpy(0.95 * normalize(sample["audio"].numpy())).to(torch.float32)
            assert -1.0 < audio.min() and audio.max() < 1.0
            residual = len(audio) % self.hop_length
            if residual > 0:
                # L = len(audio) / hop_length * hop_length
                audio = audio[:-residual]
            audio_list.append(audio)
            audio_length.append(len(audio))
            token = self.tokenizer.encode(sample["text"])
            token_list.append(torch.tensor(token, dtype=torch.long))
            token_length_list.append(len(token))
        batch = {
            "audio": pad_sequence(audio_list, True, 0.0),
            "audio_length": torch.tensor(audio_length, dtype=torch.long),
            "token": pad_sequence(token_list, True, self.pad_token_id),
            "token_length": torch.tensor(token_length_list, dtype=torch.long),
        }
        return batch
