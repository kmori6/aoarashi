"""Conformer modules.

Proposed in A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition,"
in Interspeech, 2020, pp. 5036-5040.

"""

import torch
import torch.nn as nn

from konpeki.modules.transformer import FeedForward, RelativePositionalMultiHeadAttention
from konpeki.utils.mask import sequence_mask


class ConvolutionSubsampling(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, output_size, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output sequence (batch_size, frame_size', output_size').
                torch.Tensor: Mask sequence (batch_size, frame_size').
                where:
                    - frame_size' = (frame_size - 1) // 2 - 1) // 2.
                    - output_size' = output_size * ((input_size - 1) // 2 - 1) // 2.
        """
        x = x[:, None, :, :]  # (batch_size, 1, frame_length, input_size)
        x = self.conv1(x)  # (batch_size, output_size, (frame_length - 1) // 2, (input_size - 1) // 2)
        x = self.activation(x)
        # (batch_size, output_size, ((frame_length - 1) // 2 - 1) // 2, ((input_size - 1) // 2 - 1) // 2)
        x = self.conv2(x)
        x = self.activation(x)
        # (batch_size, ((frame_length - 1) // 2 - 1) // 2, output_size * ((input_size - 1) // 2 - 1) // 2)
        x = x.transpose(1, 2).flatten(2, -1)
        return x, sequence_mask(((mask.sum(-1) - 1) // 2 - 1) // 2)


class FeedForwardModule(FeedForward):
    def __init__(self, input_size: int, dropout_rate: float):
        super().__init__(input_size, 4 * input_size, dropout_rate, activation=nn.SiLU())
        self.layernorm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = super().forward(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttentionModule(RelativePositionalMultiHeadAttention):
    def __init__(self, input_size: int, num_heads: int, dropout_rate: float):
        super().__init__(input_size, num_heads, dropout_rate)
        self.layernorm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length, frame_length).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = super().forward(x, x, x, mask)
        x = self.dropout(x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_size)
        self.pointwise_conv1 = nn.Conv1d(input_size, 2 * input_size, kernel_size=1, stride=1, padding=0)
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            input_size, input_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=input_size
        )
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.swish_activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = x.transpose(1, 2)  # (batch_size, input_size, frame_length)
        x = self.pointwise_conv1(x)  # (batch_size, 2 * input_size, frame_length)
        x = self.glu_activation(x)  # (batch_size, input_size, frame_length)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (batch_size, frame_length, input_size)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, input_size: int, num_heads: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(input_size, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(input_size, num_heads, dropout_rate)
        self.conv = ConvolutionModule(input_size, kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(input_size, dropout_rate)
        self.layernorm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, 1, 1, frame_length).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length, input_size).
        """
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, mask)
        x = x + self.conv(x)
        x = self.layernorm(x + 0.5 * self.ffn2(x))
        return x


class Conformer(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int, kernel_size: int, num_blocks: int, dropout_rate: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.convolution_subsampling = ConvolutionSubsampling(hidden_size)
        self.linear = nn.Linear(hidden_size * (((input_size - 1) // 2 - 1) // 2), hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.conformer_blocks = nn.ModuleList(
            [ConformerBlock(hidden_size, num_heads, kernel_size, dropout_rate) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length', hidden_size).
            torch.Tensor: Mask tensor (batch_size, frame_length').
            where frame_length' = (frame_length - 1) // 2 - 1) // 2.
        """
        x, mask = self.convolution_subsampling(
            x, mask
        )  # (batch_size, frame_length', hidden_size * (((input_size - 1) // 2 - 1) // 2)
        mask = mask[:, None, :].expand(-1, x.shape[1], -1)  # (batch_size, frame_length', frame_length')
        x = self.linear(x)  # (batch_size, frame_length', hidden_size)
        x = self.dropout(x)
        for block in self.conformer_blocks:
            x = block(x, mask[:, None, None, :])
        return x, mask
