"""RAD-TTS modules.

Proposed in K. Shih et al., "RAD-TTS: parallel flow-based tts with robust alignment learning and diverse synthesis,"
in ICML Workshop, 2021.

"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import betabinom

from aoarashi.modules.glow_tts import monotonic_alignment_search


def hard_alignment(log_a_soft: torch.Tensor, token_length: torch.Tensor, feat_length: torch.Tensor) -> torch.Tensor:
    """Convert hard alignment from log-probability soft alignment

    Args:
        log_a_soft (torch.Tensor): Log-probability soft alignment tensor (batch_size, sequence_length, frame_length).
        token_length (torch.Tensor): Token length tensor (batch_size).
        feat_length (torch.Tensor): Feature length tensor (batch_size).

    Returns:
        torch.Tensor: Hard alignment tensor (batch_size, sequence_length, frame_length).
    """
    batch_size, sequence_length, frame_length = log_a_soft.shape
    a_hard_list = []
    for b in range(batch_size):
        _log_a_soft = log_a_soft[b, : token_length[b], : feat_length[b]]  # (sequence_length', frame_length')
        alignment = torch.from_numpy(
            monotonic_alignment_search(_log_a_soft.detach().cpu().numpy(), int(token_length[b]), int(feat_length[b]))
        )  # (frame_length')
        a_hard = torch.zeros_like(_log_a_soft)  # (sequence_length', frame_length')
        a_hard[alignment, torch.arange(int(feat_length[b]))] = 1
        a_hard = nn.functional.pad(
            a_hard, (0, frame_length - int(feat_length[b]), 0, sequence_length - int(token_length[b])), value=0
        )  # (sequence_length, frame_length)
        a_hard_list.append(a_hard)
    a_hard = torch.stack(a_hard_list, dim=0)  # (batch_size, sequence_length, frame_length)
    return a_hard


def beta_binomial_prior(sequence_length: int, frame_length: int, omega: float = 1.0) -> np.ndarray:
    """Static 2D beta-binomial alignment prior.

    Args:
        sequence_length (int): Text token length.
        frame_length (int): Mel-spectrogram frame length.
        omega (float): Hyperparameter for beta function.

    Returns:
        np.ndarray: Beta-binomial prior log-probability (sequence_length, frame_length).
    """
    log_probs = []
    k = np.arange(sequence_length)
    for i in range(1, frame_length + 1):
        # NOTE: N -> N - 1 in the original paper formula to match the token length
        rv = betabinom(sequence_length - 1, omega * i, omega * (frame_length - i + 1))
        log_probs.append(rv.logpmf(k))
    log_prior = np.stack(log_probs, axis=1)  # (sequence_length, frame_length)
    return log_prior


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

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        mask: torch.Tensor,
        token_length: torch.Tensor,
        feat_length: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            h (torch.Tensor): Token embedding tensor (batch_size, sequence_length, text_size).
            m (torch.Tensor): Mel-spectrogram tensor (batch_size, frame_length, feat_size).
            mask (torch.Tensor): Token mask tensor (batch_size, sequence_length).

        Returns:
            torch.Tensor: Log-probability soft alignment tensor (batch_size, sequence_length, frame_length).
        """
        h = self.text_conv(h.transpose(1, 2)).transpose(1, 2)  # (batch_size, sequence_length, text_size)
        m = self.feat_conv(m.transpose(1, 2)).transpose(1, 2)  # (batch_size, frame_length, text_size)
        # softmax normalized across text domain
        dist = torch.linalg.norm(
            h[:, :, None, :] - m[:, None, :, :], ord=2, dim=-1
        )  # (batch_size, sequence_length, frame_length)
        logit = -dist
        logit = logit.masked_fill(~mask[:, :, None], float("-inf"))
        log_a_soft = torch.log_softmax(logit, dim=1)  # (batch_size, sequence_length, frame_length)
        # apply static 2d beta-binomial prior for fast convergence
        log_prior = torch.full_like(log_a_soft, fill_value=float("-inf"))
        for b in range(h.shape[0]):
            log_prior[b, : int(token_length[b]), : int(feat_length[b])] = torch.from_numpy(
                beta_binomial_prior(int(token_length[b]), int(feat_length[b]))
            ).to(device=h.device, dtype=h.dtype)
        log_a_soft = log_prior + log_a_soft
        return log_a_soft


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_log_prob: float = -1.0):
        super().__init__()
        self.blank_log_prob = blank_log_prob
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, x: torch.Tensor, token_length: torch.Tensor, feat_length: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Log-probability soft alignment tensor (batch_size, sequence_length, frame_length).
            token_length (torch.Tensor): Text length tensor (batch_size).
            feat_length (torch.Tensor): Frame length tensor (batch_size).

        Returns:
            torch.Tensor: Loss value.
        """
        losses = []
        for b in range(x.shape[0]):
            log_prob = x[b, : int(token_length[b]), : int(feat_length[b])]  # (sequence_length', frame_length')
            # pad logarithmic domain 0 values for ctc algorithm at 0 index
            log_prob = torch.cat(
                [torch.full_like(log_prob[:1, :], fill_value=self.blank_log_prob), log_prob], dim=0
            )  # (sequence_length' + 1, frame_length')
            log_prob = torch.log_softmax(log_prob, dim=0).transpose(0, 1)  # (frame_length', sequence_length' + 1)
            target = torch.arange(1, int(token_length[b]) + 1, device=x.device, dtype=torch.long)  # (sequence_length',)
            loss = self.ctc_loss_fn(
                log_prob[:, None, :],  # (frame_length', 1, sequence_length' + 1)
                target[None, :],  # (1, sequence_length')
                input_lengths=feat_length[b : b + 1],
                target_lengths=token_length[b : b + 1],
            )
            losses.append(loss)
        loss = torch.stack(losses).mean()
        return loss


class BinalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_a_soft: torch.Tensor, a_hard: torch.Tensor) -> torch.Tensor:
        """

        Args:
            log_as_soft (torch.Tensor): Log-probability soft alignment tensor
                (batch_size, sequence_length, frame_length).
            as_hard (torch.Tensor): Hard alignment tensor (batch_size, sequence_length, frame_length).

        Returns:
            torch.Tensor: Loss value.
        """
        assert log_a_soft.shape == a_hard.shape
        loss = -log_a_soft.masked_select(a_hard == 1).mean()
        return loss
