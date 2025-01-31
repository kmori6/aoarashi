"""WaveNet modules.

Proposed in A. v. d. Oord et al., "WaveNet: a Generative model for raw audio," arXiv preprint arXiv:1609.03499, 2016.

"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, dilation: int, residual: bool):
        super().__init__()
        self.sigmoid_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size,
            stride=1,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False,
        )
        self.tanh_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size,
            stride=1,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False,
        )
        self.skip_conv = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.residual_conv = None
        if residual:
            self.residual_conv = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, input_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - torch.Tensor: Output tensor (batch_size, input_size, frame_length).
                - torch.Tensor: Skip connection tensor (batch_size, input_size, frame_length).
        """
        x_res = x
        x = torch.tanh(self.tanh_conv(x)) * torch.sigmoid(self.sigmoid_conv(x))
        x_skip = self.skip_conv(x)
        if self.residual_conv is not None:
            x = x_res + self.residual_conv(x)
        return x, x_skip
