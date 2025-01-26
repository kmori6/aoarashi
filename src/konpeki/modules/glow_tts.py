"""Glow-TTS modules.

Proposed in J. Kim et al., "Glow-TTS: a generative flow for text-to-speech via monotonic alignment search,"
in NeurIPS, 2020, pp. 8067-8077.

"""

import numpy as np


def monotonic_alignment_search(value: np.ndarray, token_length: int, feat_length: int) -> np.ndarray:
    """Monotonic alignment search algorithm with cpu.

    Args:
        value (np.ndarray): Log-likelihood matrix (token_length, feat_length).
        token_length (int): Token length.
        feat_length (int): Frame length.

    Returns:
        torch.Tensor: Most probable monotonic alignment (feat_length).

    Examples:
        >>> value = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.2, -0.3, -0.4, -0.5], [-0.3, -0.4, -0.5, -0.6]])
        >>> monotonic_alignment_search(value, token_length=3, feat_length=4)
        tensor([0, 0, 1, 2])

    """
    assert value.shape == (token_length, feat_length)
    # Initialize Q[:, :] <- -infilnity, a cache to store the maximum log-likelihood calculations
    Q = np.full_like(value, fill_value=-np.inf)  # (sequence, frame)
    # Compute the first row Q[1, j] <- \Sigma_{k=1}^{j} value[1, i]
    Q[0, :] = np.cumsum(value[0, :], axis=0)
    for j in range(1, feat_length):
        for i in range(1, min(j + 1, token_length)):
            # Q[i, j] <- max(Q[i-1, j-1], Q[i, j-1]) + value[i, j]
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + value[i, j]
    # Initialize A*(feat_length) <- token_length
    A_star = np.full((feat_length,), fill_value=token_length - 1, dtype=np.int64)
    for j in reversed(range(feat_length - 1)):
        # A*(j) <- argmax_{i \in {A*(j+1)-1, A*(j+1)}} Q[i, j]
        mask = np.zeros(token_length, dtype=bool)
        mask[A_star[j + 1] - 1] = mask[A_star[j + 1]] = True
        A_star[j] = np.argmax(np.where(~mask, -np.inf, Q[:, j]))
    return A_star
