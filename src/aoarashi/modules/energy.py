import torch
import torch.nn as nn

from aoarashi.utils.mask import sequence_mask


class Energy(nn.Module):
    """L2-norm of the amplitude of each short-time Fourier transform (STFT) frame."""

    def __init__(self, fft_size: int, hop_size: int, window_size: int, eps: float = 1e-12):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size
        self.register_buffer("window", torch.hann_window(window_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Speech tensor (batch, sample).
            length (torch.Tensor): Speech length tensor (batch,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Energy tensor (batch, sample // hop_size + 1).
                torch.Tensor: Mask tensor (batch, sample // hop_size + 1).
        """
        x = torch.stft(
            speech,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window.to(dtype=speech.dtype, device=speech.device),
            return_complex=True,
        )  # (batch, freq_size, frame)
        mask = sequence_mask(1 + length // self.hop_size)  # (batch, frame)
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch, frame, freq_size)
        x = torch.linalg.norm(x, ord=2, dim=-1)  # (batch, frame)
        x = x.masked_fill(~mask, 0.0)
        return x, mask
