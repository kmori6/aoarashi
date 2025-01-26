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


def streaming_mask(length: torch.Tensor, chunk_size: int, history_size: int) -> torch.Tensor:
    """Create a streaming mask.

    Proposed in X. Chen et al., "Developing realtime streaming transformer transducer for speech recognition
    on large-scale dataset," in ICASSP, 2021, pp. 5904-5908.

    Args:
        length (torch.Tensor): Sequence length (batch_size,).
        chunk_size (int): Chunk size.
        history_size (int): History size.

    Examples:
        >>> streaming_mask(torch.tensor([3, 5]), 2, 1)
        tensor([[[ True,  True, False, False, False],
                 [ True,  True, False, False, False],
                 [ False,  True, True, True, False],
                 [ False,  True, True, True, False],
                 [ False,  False, False, True, True]],

                [[ True,  True, False, False, False],
                 [ True,  True, False, False, False],
                 [ False,  True, True, True, False],
                 [ False,  True, True, True, False],
                 [ False,  False, False, True, True]]])

    Returns:
        torch.Tensor: Mask tensor (batch_size, max_sequence_length, max_sequence_length).
    """
    max_length = max(length)
    start = torch.arange(-history_size, max_length, chunk_size).repeat_interleave(chunk_size)[:max_length]
    end = torch.arange(chunk_size, max_length + chunk_size, chunk_size).repeat_interleave(chunk_size)[:max_length]
    idx = torch.arange(max_length)
    mask = (start[:, None] <= idx) & (idx < end[:, None])
    return mask.to(length.device)
