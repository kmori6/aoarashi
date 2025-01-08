"""EATS modules.

Proposed in J. Donahue et al., "End-to-end adversarial text-to-speech," in ICLR, 2021.

"""

import torch
import torch.nn as nn


class GaussianResampling(nn.Module):
    def __init__(self, variance: float = 10.0):
        super().__init__()
        self.variance = variance

    def forward(self, x: torch.Tensor, d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        w^n_t = exp(-((t - c_n)/sigma)**2)}{sum^N_{m=1} exp(-((t - c^m)/sigma)**2)}

        Args:
            x (torch.Tensor): Token embedding tensor (batch, sequence, d_model).
            d (torch.Tensor): Each token length tensor (batch, sequence).
            mask (torch.Tensor): Token mask tensor (batch, sequence).

        Returns:
            torch.Tensor: Resampled token embedding tensor (batch, frame, d_model).
        """
        e = torch.cumsum(d, dim=-1)  # (batch, sequence)
        c = e - d / 2  # (batch, sequence)
        frame_length = int(torch.max(e[:, -1]))
        # NOTE: frame index starts from 1 following the token index definition in the paper (t = {1, ..., N})
        t = torch.arange(1, frame_length + 1, device=x.device, dtype=x.dtype)  # (frame)
        logits = -((t[None, :, None] - c[:, None, :]) ** 2) / self.variance  # (batch, frame, sequence)
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        w = torch.softmax(logits, dim=-1)  # (batch, frame, sequence)
        x = torch.matmul(w, x)  # (batch, frame, d_model)
        return x
