import numpy as np
import pyworld as pw
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Pitch(nn.Module):
    """Extract fundamental frequency (F0) from waveform."""

    def __init__(self, sample_rate: int, hop_size: int, eps: float = 1e-12):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.eps = eps

    def forward(self, audio: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, sample).
            length (torch.Tensor): Input length (batch,).

        Returns:
            torch.Tensor: Output tensor (batch, sequence).
        """
        xs = []
        for b in range(audio.shape[0]):
            audio_numpy = audio[b, : length[b]].cpu().numpy().astype(np.float64)
            _f0, t = pw.dio(
                audio_numpy,
                fs=self.sample_rate,
                frame_period=self.hop_size / self.sample_rate * 1000,  # to make nearly equal frames to STFT
            )  # raw pitch extractor
            x = pw.stonemask(audio_numpy, _f0, t, self.sample_rate)
            # linear interpolation to fill the unvoiced frame use nn.functional.interpolate
            mask = x > 0
            mask[0] = mask[-1] = True
            x = np.interp(np.arange(len(x)), t[mask], x[mask])
            # transform the resulting pitch contour to logarithmic scale
            x = np.log(x + self.eps)
            # normalize it to zero mean and unit variance for each utterance
            x = (x - x.mean()) / (x.std() + self.eps)
            x = torch.from_numpy(x).to(device=audio.device, dtype=audio.dtype)
            xs.append(x)
        x = pad_sequence(xs, batch_first=True, padding_value=0.0)
        return x
