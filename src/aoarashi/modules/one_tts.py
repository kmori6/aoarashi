"""One-TTS alignment module.

Proposed in R. Badlani et al., "One tts alignment to rule them all," in ICASSP, 2022, pp. 6092-6096.

"""

import torch
import torch.nn as nn

from aoarashi.modules.glow_tts import monotonic_alignment_search


def hard_alignment(log_a_soft: torch.Tensor, token_length: torch.Tensor, feat_length: torch.Tensor) -> torch.Tensor:
    """Convert soft alignment to hard alignment.

    Args:
        log_a_soft (torch.Tensor): Log-probability soft alignment tensor (batch, sequence, frame).
        token_length (torch.Tensor): Token length tensor (batch).
        feat_length (torch.Tensor): Frame length tensor (batch).

    Returns:
        torch.Tensor: Hard alignment tensor (batch, frame, sequence).
    """
    batch_size, sequence_size, frame_size = log_a_soft.shape
    a_hard_list = []
    for b in range(batch_size):
        _log_a_soft = log_a_soft[b, : token_length[b], : feat_length[b]]  # (sequence', frame')
        alignment = torch.from_numpy(
            monotonic_alignment_search(_log_a_soft.detach().cpu().numpy(), int(token_length[b]), int(feat_length[b]))
        )  # (frame')
        a_hard = torch.zeros_like(_log_a_soft)  # (sequence', frame')
        a_hard[alignment, torch.arange(int(feat_length[b]))] = 1
        a_hard = nn.functional.pad(
            a_hard, (0, frame_size - int(feat_length[b]), 0, sequence_size - int(token_length[b])), value=0
        )  # (sequence, frame)
        a_hard_list.append(a_hard)
    a_hard = torch.stack(a_hard_list, dim=0)  # (batch, sequence, frame)
    return a_hard


class AlignmentModule(nn.Module):
    def __init__(self, text_size: int, feat_size: int):
        super().__init__()
        self.text_conv = nn.Sequential(
            nn.Conv1d(text_size, text_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(text_size, text_size, kernel_size=1, padding=0),
        )
        self.feat_conv = nn.Sequential(
            nn.Conv1d(feat_size, text_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(text_size, text_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(text_size, text_size, kernel_size=1, padding=0),
        )

    def forward(self, h: torch.Tensor, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            h (torch.Tensor): Token embedding tensor (batch, sequence, text_size).
            m (torch.Tensor): Mel-spectrogram tensor (batch, frame, feat_size).
            mask (torch.Tensor): Token mask tensor (batch, sequence).

        Returns:
            torch.Tensor: Log-probability soft alignment tensor (batch, sequence, frame).
        """
        h = self.text_conv(h.transpose(1, 2)).transpose(1, 2)  # (batch, sequence, text_size)
        m = self.feat_conv(m.transpose(1, 2)).transpose(1, 2)  # (batch, frame, text_size)
        # softmax normalized across text domain
        dist = torch.linalg.norm(h[:, :, None, :] - m[:, None, :, :], ord=2, dim=-1)  # (batch, sequence, frame)
        dist = dist.masked_fill(~mask[:, :, None], 0.0)
        log_a_soft = torch.log_softmax(-dist, dim=1)  # (batch, sequence, frame)
        return log_a_soft
