import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks

from aoarashi.utils.mask import sequence_mask


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        window_size: int,
        mel_size: int,
        sample_rate: int,
        eps: float = 1e-12,
        from_linear: bool = False,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size
        self.register_buffer("window", torch.hann_window(window_size, dtype=torch.float32))
        self.register_buffer(
            "fbank",
            melscale_fbanks(
                n_freqs=fft_size // 2 + 1, f_min=0.0, f_max=sample_rate / 2, n_mels=mel_size, sample_rate=sample_rate
            ),
        )  # (freq_size, mel_size)
        self.from_linear = from_linear
        self.eps = eps

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Speech tensor (batch, sample).
            length (torch.Tensor): Speech length tensor (batch,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Log mel-spectrogram tensor (batch, sample // hop_size + 1, mel_size).
                torch.Tensor: Mask tensor (batch, sample // hop_size + 1).
        """
        x = torch.stft(
            speech,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window.to(dtype=speech.dtype, device=speech.device),
            center=True,
            return_complex=True,
        )  # (batch, freq_size, frame)
        mask = sequence_mask(1 + length // self.hop_size)  # (batch, frame)
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch, frame, freq_size)
        if self.from_linear:
            x = torch.sqrt(x)
        x = torch.matmul(x, self.fbank)  # (batch, frame, n_mels)
        x = torch.log(x + self.eps)
        x = x.masked_fill(~mask[:, :, None], 0.0)
        return x, mask
