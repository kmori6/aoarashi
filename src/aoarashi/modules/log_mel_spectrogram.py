import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks

from aoarashi.utils.mask import sequence_mask


class LogMelSpectrogram(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int, eps: float = 1e-8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, dtype=torch.float32))
        self.register_buffer(
            "fbank", melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=0, f_max=8000, n_mels=n_mels, sample_rate=16000)
        )  # (n_freqs, n_mels)
        self.eps = eps

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Input sequence (batch, sample).
            length (torch.Tensor): Input length (batch,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output sequence (batch, frame, n_mels).
        """
        x = torch.stft(
            speech,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=speech.dtype, device=speech.device),
            return_complex=True,
        )  # (batch, n_freqs, frame)
        mask = sequence_mask(1 + length // self.hop_length)  # (batch, frame)
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch, frame, n_freqs)
        x = torch.matmul(x, self.fbank)  # (batch, frame, n_mels)
        x = torch.log(x + self.eps)
        x = x.masked_fill(~mask[:, :, None], 0.0)
        return x, mask
