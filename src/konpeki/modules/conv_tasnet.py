"""Conv-TasNet modules.

Proposed in Y. Luo et al., "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation,"
IEEE TASLP, vol. 27, no. 8, pp. 1256-1266, 2019.

"""

from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn


def si_snr(s_hat: torch.Tensor, s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """

    Args:
        s_hat (torch.Tensor): estimated waveform tensor (batch_size, sample_length).
        s (torch.Tensor): Original waveform tensor (batch_size, sample_length).

    Returns:
        torch.Tensor: Scale-invariant source-to-noise ratio (batch_size,).
    """
    s_target = (
        torch.sum(s_hat * s, dim=-1, keepdim=True) * s / (torch.linalg.norm(s, ord=2, dim=-1, keepdim=True) ** 2 + eps)
    )  # (batch_size, sample_length)
    e_noise = s_hat - s_target  # (batch_size, sample_length)
    si_snr = 10 * torch.log10(
        torch.linalg.norm(s_target, ord=2, dim=-1) ** 2 / torch.linalg.norm(e_noise, ord=2, dim=1) ** 2 + eps
    )  # (batch_size,)
    return si_snr


class ScaleInvariantSourceToNoiseRatioLoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, s_hat: torch.Tensor, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            s_hat (torch.Tensor): estimated waveform tensor (batch_size, num_speakers, sample_length).
            s (torch.Tensor): Original waveform tensor (batch_size, num_speakers, sample_length).
            mask (torch.Tensor): Mask tensor (batch_size, sample_length).

        Returns:
            torch.Tensor: Scale-invariant source-to-noise ratio loss.
        """
        s_hat = s_hat.masked_fill(~mask[:, None, :], 0.0)
        s = s.masked_fill(~mask[:, None, :], 0.0)
        # permutation consideration
        candidate_list = []
        for permutation in permutations(range(s.shape[1])):
            si_snr_list = []
            for src, tgt in enumerate(permutation):
                si_snr_value = si_snr(s_hat[:, src, :], s[:, tgt, :])  # (batch_size,)
                si_snr_list.append(si_snr_value)
            si_snr_value = torch.stack(si_snr_list, dim=-1)  # (batch_size, num_speakers)
            candidate = torch.sum(si_snr_value, dim=-1) / s.shape[1]  # (batch_size,)
            candidate_list.append(candidate)
        candidate = torch.stack(candidate_list, dim=-1)  # (batch_size, num_permutations)
        probable, _ = torch.max(candidate, dim=-1)  # (batch_size,)
        loss = -probable.mean()
        return loss


class GlobalLayerNormalization(nn.LayerNorm):
    """The feature is normalized over both the channel and the time dimensions.

    gLN(F) = (F - E[F]) / sqrt(Var[F] + eps) * gamma + beta
    E[F] = 1 / NT sum_{n=1}^{N} sum_{t=1}^{T} F_{nt}
    Var[F] = 1 / NT sum_{n=1}^{N} sum_{t=1}^{T} (F_{nt} - E[F])^2

    """

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (torch.Tensor): Input tensor (B, N, T).

        Returns:
            torch.Tensor: Normalized tensor (B, N, T).
        """
        mean = torch.mean(x, dim=(1, 2), keepdim=True)  # (B, 1, 1)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps) * self.weight[None, :, None] + self.bias[None, :, None]
        return x


class Encoder(nn.Module):
    def __init__(self, filter_size: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            1, filter_size, kernel_size, stride=stride, padding=(kernel_size - stride) // 2, bias=False
        )
        self.activation = nn.ReLU()

    def forward(self, audio: torch.Tensor, audio_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            audio: (torch.Tensor): Mixture waveform tensor (batch_size, sample_length).
            audio_length: (torch.Tensor): Length of mixture waveform tensor (batch_size,).

        Returns:
            torch.Tensor: Embedding tensor (batch_size, filter_size, sample_length // stride).
            torch.Tensor: Frame length tensor (batch_size,).
        """
        x = self.conv(audio[:, None, :])  # (batch_size, filter_size, frame_length)
        x = self.activation(x)
        feat_length = audio_length // self.stride
        return x, feat_length


class Decoder(nn.Module):
    def __init__(self, filter_size: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            filter_size, 1, kernel_size, stride=stride, padding=(kernel_size - stride) // 2, bias=False
        )

    def forward(self, x: torch.Tensor, feat_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: (torch.Tensor): Embedding tensor (batch_size, filter_size, frame_length).
            audio_length: (torch.Tensor): Length of mixture waveform tensor (batch_size,).

        Returns:
            torch.Tensor: Waveform tensor (batch_size, frame_length * stride).
            torch.Tensor: Length of waveform tensor (batch_size,).
        """
        x = self.conv(x)  # (batch_size, 1, sample_length)
        audio_length = feat_length * self.stride
        return x, audio_length


class ConvBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dilation: int, output_conv: bool = True):
        super().__init__()
        self.pointwise_conv = nn.Conv1d(input_size, hidden_size, 1)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=hidden_size,
        )
        self.skip_conv = nn.Conv1d(hidden_size, input_size, 1)
        self.output_conv: Optional[nn.Conv1d] = None
        if output_conv:
            self.output_conv = nn.Conv1d(hidden_size, input_size, 1)
        self.gln1 = GlobalLayerNormalization(hidden_size)
        self.gln2 = GlobalLayerNormalization(hidden_size)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: (torch.Tensor): Acoustic embedding tensor (batch_size, input_size, frame_length).


        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Residual acoustic embedding tensor (batch_size, input_size, frame_length).
                torch.Tensor: Skip-connection embedding tensor (batch_size, input_size, frame_length).
        """
        x_res = x
        x = self.pointwise_conv(x)  # (batch_size, hidden_size, frame_length)
        x = self.prelu(x)
        x = self.gln1(x)
        x = self.depthwise_conv(x)
        x = self.prelu(x)
        x = self.gln2(x)
        x_skip = self.skip_conv(x)  # (batch_size, input_size, frame_length)
        if self.output_conv is not None:
            x = x_res + self.output_conv(x)  # (batch_size, input_size, frame_length)
        return x, x_skip


class Separation(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_speakers: int,
        bottleneck_size: int,
        convolutional_size: int,
        kernel_size: int,
        num_blocks: int,
        num_repeats: int,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.layer_norm = nn.LayerNorm(input_size)
        self.input_conv = nn.Conv1d(input_size, bottleneck_size, 1)
        self.blocks = nn.ModuleList([])
        for r in range(num_repeats):
            for b in range(num_blocks):
                self.blocks.append(
                    ConvBlock(
                        input_size=bottleneck_size,
                        hidden_size=convolutional_size,
                        kernel_size=kernel_size,
                        dilation=2**b,
                        output_conv=r != num_repeats - 1 or b != num_blocks - 1,  # Last block
                    )
                )
        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(bottleneck_size, input_size * num_speakers, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (torch.Tensor): Acoustic embedding tensor (batch_size, input_size, frame_length).

        Returns:
            torch.Tensor: Separated acoustic embedding tensor (batch_size, num_speakers, input_size, frame_length).
        """
        b, n, t = x.shape
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # (batch_size, input_size, frame_length)
        x = self.input_conv(x)  # (batch_size, bottleneck_size, frame_length)
        skips = []
        for block in self.blocks:
            x, skip = block(x)  # (batch_size, bottleneck_size, frame_length)
            skips.append(skip)
        x = torch.stack(skips, dim=1)  # (batch_size, num_blocks, bottleneck_size, frame_length)
        x = torch.sum(x, dim=1)  # (batch_size, bottleneck_size, frame_length)
        x = self.prelu(x)
        x = self.output_conv(x)  # (batch_size, input_size * num_speakers, frame_length)
        x = x.view(b, self.num_speakers, n, t)  # (batch_size, num_speakers, input_size, frame_length)
        x = torch.sigmoid(x)
        return x
