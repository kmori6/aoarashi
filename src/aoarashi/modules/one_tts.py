"""One-TTS modules.

Proposed in R. Badlani et al., "One tts alignment to rule them all," in ICASSP, 2022, pp. 6092-6096.

"""

import torch
import torch.nn as nn

from aoarashi.modules.glow_tts import monotonic_alignment_search


class AlignmentModule(nn.Module):
    def __init__(self, d_text: int, d_feat: int):
        super().__init__()
        self.text_conv = nn.Sequential(
            nn.Conv1d(d_text, d_text, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_text, d_text, kernel_size=1, padding=0),
        )
        self.feat_conv = nn.Sequential(
            nn.Conv1d(d_feat, d_text, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_text, d_text, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_text, d_text, kernel_size=1, padding=0),
        )

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        token_length: torch.Tensor,
        feat_length: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Token embedding tensor (batch, sequence, d_text).
            x (torch.Tensor): Mel-spectrogram tensor (batch, frame, d_feat).
            token_length (torch.Tensor): Token length tensor (batch).
            feat_length (torch.Tensor): Frame length tensor (batch).
            mask (torch.Tensor): Token mask tensor (batch, sequence).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.Tensor: Duration tensor (batch, sequence).
                torch.Tensor: Log-probability soft alignment tensor (batch, frame, sequence).
                torch.Tensor: Hard alignment tensor (batch, frame, sequence).
        """
        h = self.text_conv(h.transpose(1, 2)).transpose(1, 2)  # (batch, sequence, d_text)
        m = self.feat_conv(m.transpose(1, 2)).transpose(1, 2)  # (batch, frame, d_text)
        # softmax normalized across text domain
        dist = torch.linalg.norm(h[:, :, None, :] - m[:, None, :, :], ord=2, dim=-1)  # (batch, sequence, frame)
        dist = dist.masked_fill(~mask[:, :, None], 0.0)
        log_a_soft = torch.log_softmax(-dist, dim=1)  # (batch, sequence, frame)
        batch_size, sequence_size, frame_size = log_a_soft.shape
        a_hard_list = []
        for b in range(batch_size):
            _log_a_soft = log_a_soft[b, : token_length[b], : feat_length[b]]  # (sequence', frame')
            alignment = torch.from_numpy(
                monotonic_alignment_search(
                    _log_a_soft.detach().cpu().numpy(), int(token_length[b]), int(feat_length[b])
                )
            )  # (frame')
            a_hard = torch.zeros_like(_log_a_soft)  # (sequence', frame')
            a_hard[alignment, torch.arange(int(feat_length[b]))] = 1
            a_hard = nn.functional.pad(
                a_hard, (0, frame_size - int(feat_length[b]), 0, sequence_size - int(token_length[b])), value=0
            )  # (sequence, frame)
            a_hard_list.append(a_hard)
        a_hard = torch.stack(a_hard_list, dim=0)  # (batch, sequence, frame)
        d = torch.sum(a_hard, dim=-1)  # (batch, sequence)
        return d, log_a_soft, a_hard
