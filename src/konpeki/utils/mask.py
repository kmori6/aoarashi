import torch


def sequence_mask(length: torch.Tensor) -> torch.Tensor:
    """Create a sequence mask.

    Args:
        length (torch.Tensor): Sequence length (batch,).

    Returns:
        torch.Tensor: Mask tensor (batch, max_length).

    Examples:
        >>> sequence_mask(torch.tensor([3, 5]))
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True]])
    """
    return torch.arange(max(length), device=length.device) < length[:, None]


def causal_mask(length: torch.Tensor) -> torch.Tensor:
    """Create a causal mask.

    Args:
        length (torch.Tensor): Sequence length (batch,).

    Returns:
        torch.Tensor: Mask tensor (batch, max_length, max_length).

    Examples:
        >>> causal_mask(torch.tensor([2, 3]))
        tensor([[[ True, False, False],
                [ True,  True, False],
                [ True,  True, False]],

                [[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]]])
    """
    max_len = max(length)
    return (
        sequence_mask(length)[:, None, :]
        & torch.ones(1, max_len, max_len, dtype=torch.bool, device=length.device).tril()
    )
