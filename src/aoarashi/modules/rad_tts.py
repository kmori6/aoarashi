"""RAD-TTS modules.

Proposed in K. Shih et al., "RAD-TTS: parallel flow-based tts with robust alignment learning and diverse synthesis,"
in ICML Workshop, 2021.

"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import betabinom


def static_prior(sequence: int, frame: int, omega: float = 1.0) -> np.ndarray:
    """Static 2D beta-binomial alignment prior.

    Args:
        sequence (int): Text token length.
        frame (int): Mel-spectrogram length.
        omega (float): Hyperparameter for beta function.

    Returns:
        np.ndarray: Beta-binomial prior log-probability (sequence, frame).
    """
    log_probs = []
    k = np.arange(sequence)
    for i in range(1, frame + 1):
        # NOTE: N -> N - 1 in the original paper formula to match the token length
        rv = betabinom(sequence - 1, omega * i, omega * (frame - i + 1))
        log_probs.append(rv.logpmf(k))
    log_prior = np.stack(log_probs, axis=1)  # (sequence, frame)
    return log_prior


class ForwardSumLoss(nn.Module):
    def __init__(self, log_blank_prob: float = -1.0):
        super().__init__()
        self.log_blank_prob = log_blank_prob
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, x: torch.Tensor, token_length: torch.Tensor, feat_length: torch.Tensor) -> torch.Tensor:
        """Alignment learning objective with forward sum algorithm.

        Args:
            x (torch.Tensor): Log-probability soft alignment tensor (batch, sequence, frame).
            token_length (torch.Tensor): Text length tensor (batch).
            feat_length (torch.Tensor): Frame length tensor (batch).
            eps (float): Constant blank probability.

        Returns:
            torch.Tensor: Loss value.
        """
        losses = []
        for b in range(x.shape[0]):
            log_prob = x[b, : int(token_length[b]), : int(feat_length[b])]  # (sequence', frame')
            # apply static 2d beta-binomial prior for fast convergence
            log_prior = torch.from_numpy(static_prior(int(token_length[b]), int(feat_length[b]))).to(
                device=x.device, dtype=x.dtype
            )  # (sequence', frame')
            log_prob = log_prior + log_prob
            # pad logarithmic domain 0 values for ctc algorithm at 0 index
            log_prob = torch.cat(
                [torch.full_like(log_prob[:1, :], fill_value=self.log_blank_prob), log_prob], dim=0
            )  # (sequence' + 1, frame')
            log_prob = torch.log_softmax(log_prob, dim=0).transpose(0, 1)[:, None, :]  # (frame', 1, sequence' + 1)
            target = torch.arange(1, int(token_length[b]) + 1, device=x.device, dtype=torch.long)[
                None, :
            ]  # (1, sequence')
            loss = self.ctc_loss_fn(
                log_prob, target, input_lengths=feat_length[b : b + 1], target_lengths=token_length[b : b + 1]
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
            log_as_soft (torch.Tensor): Log-probability soft alignment tensor (batch, sequence, frame).
            as_hard (torch.Tensor): Hard alignment tensor (batch, sequence, frame).

        Returns:
            torch.Tensor: Loss value.
        """
        assert log_a_soft.shape == a_hard.shape
        loss = -log_a_soft.masked_select(a_hard == 1).mean()
        return loss
