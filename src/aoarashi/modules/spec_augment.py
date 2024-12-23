import math
import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment module.

    Proposed in D. S. Park et al., "SpecAugment on large scale datasets," in ICASSP, 2020, pp. 6879-6883.

    """

    def __init__(self, num_time_masks: int, num_freq_masks: int, time_mask_ratio: float, max_freq_mask_size: int):
        super().__init__()
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_mask_ratio = time_mask_ratio
        self.max_freq_mask_size = max_freq_mask_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, n_mels).

        Returns:
            torch.Tensor: Masked spectrogram (batch, frame, n_mels).
        """
        _, frame, n_mels = x.shape
        # frequency masking
        f = random.randint(0, self.max_freq_mask_size)
        f0 = torch.randint(0, n_mels - f, (self.num_freq_masks,))
        x[..., [range(_f0, _f0 + f) for _f0 in f0]] = 0.0
        # time masking
        max_time_mask_size = math.floor(self.time_mask_ratio * frame)
        t = random.randint(0, max_time_mask_size)
        t0 = torch.randint(0, frame - t, (self.num_time_masks,))
        x[:, [range(_t0, _t0 + t) for _t0 in t0], :] = 0.0
        return x
