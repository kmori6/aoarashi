"""RAD-TTS modules.

Proposed in K. Shih et al., "RAD-TTS: parallel flow-based tts with robust alignment learning and diverse synthesis,"
in ICML Workshop, 2021.

"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import betabinom


def static_prior(token_length: int, feat_length: int, omega: float = 1.0):
    """Static 2D beta-binomial alignment prior.

    Args:
        token_length (int): Token length.
        feat_length (int): Mel-spectrogram length.
        omega (float): Hyperparameter for beta function.

    Returns:
        np.ndarray: Beta-binomial prior probability (token_length, feat_length).
    """
    # NOTE: start k from 1 to follow the feature index t = {1, 2, ..., feat_length}
    k = np.arange(1, token_length + 1)[:, None]  # (token_length, 1)
    t = np.arange(1, feat_length + 1)[None, :]  # (1, feat_length)
    prob = betabinom.pmf(
        k=k, n=token_length, a=omega * t, b=omega * (feat_length - t + 1)
    )  # (token_length, feat_length)
    return prob


def forward_sum_loss(
    x: torch.Tensor,
    token_length: torch.Tensor,
    feat_length: torch.Tensor,
    blank_eps: float = -1,
    log_eps: float = 1e-12,
):
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
        # apply static 2d prior for fast convergence
        prior = torch.from_numpy(static_prior(int(token_length[b]), int(feat_length[b]))).to(
            device=x.device, dtype=x.dtype
        )  # (sequence', frame')
        log_prior = torch.log(prior + log_eps)  # (sequence', frame')
        log_prob = log_prior + x[b, : int(token_length[b]), : int(feat_length[b])]  # (sequence', frame')
        # pad logarithmic domain 0 values for ctc algorithm at 0 index
        log_prob = torch.cat(
            [torch.full_like(log_prob[:1, :], fill_value=blank_eps), log_prob], dim=0
        )  # (1 + sequence, frame)
        log_prob = torch.log_softmax(log_prob.transpose(0, 1), dim=1)[:, None, :]  # (frame, 1, 1 + sequence)
        target = torch.arange(1, int(token_length[b]) + 1, device=x.device, dtype=torch.long)[None, :]  # (1, sequence')
        loss = nn.functional.ctc_loss(
            log_prob,
            target,
            input_lengths=feat_length[b : b + 1],
            target_lengths=token_length[b : b + 1],
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        losses.append(loss)
    loss = torch.stack(losses).mean()
    return loss


def bin_loss(log_a_soft: torch.Tensor, a_hard: torch.Tensor):
    """Alignment learning objective with binary loss.

    Args:
        log_as_soft (torch.Tensor): Log-probability soft alignment tensor (batch, sequence, frame).
        as_hard (torch.Tensor): Hard alignment tensor (batch, sequence, frame).

    Returns:
        torch.Tensor: Loss value.
    """
    loss = -log_a_soft.masked_select(a_hard == 1).mean()
    return loss
