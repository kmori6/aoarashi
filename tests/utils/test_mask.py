import pytest
import torch

from konpeki.utils.mask import causal_mask, sequence_mask, streaming_mask


@pytest.mark.parametrize(
    "length, desired",
    [
        (torch.tensor([2, 3]), torch.tensor([[True, True, False], [True, True, True]])),
        (torch.tensor([1, 1, 1]), torch.tensor([[True], [True], [True]])),
        (torch.tensor([1, 3, 2]), torch.tensor([[True, False, False], [True, True, True], [True, True, False]])),
    ],
)
def test_sequence_mask(length: torch.Tensor, desired: torch.Tensor):
    actual = sequence_mask(length)
    assert torch.equal(actual, desired)


@pytest.mark.parametrize(
    "length, desired",
    [
        (
            torch.tensor([2, 3]),
            torch.tensor(
                [
                    [[True, False, False], [True, True, False], [True, True, False]],
                    [[True, False, False], [True, True, False], [True, True, True]],
                ]
            ),
        ),
        (
            torch.tensor([3, 3]),
            torch.tensor(
                [
                    [[True, False, False], [True, True, False], [True, True, True]],
                    [[True, False, False], [True, True, False], [True, True, True]],
                ]
            ),
        ),
    ],
)
def test_causal_mask(length: torch.Tensor, desired: torch.Tensor):
    actual = causal_mask(length)
    assert torch.equal(actual, desired)


@pytest.mark.parametrize(
    "length, chunk_size, history_size, desired",
    [
        (
            torch.tensor([3, 6]),
            2,
            0,
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, False, True, True, False, False],
                    [False, False, True, True, False, False],
                    [False, False, False, False, True, True],
                    [False, False, False, False, True, True],
                ]
            ),
        ),
        (
            torch.tensor([2, 6]),
            2,
            1,
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, False, False],
                    [False, False, False, True, True, True],
                    [False, False, False, True, True, True],
                ]
            ),
        ),
        (
            torch.tensor([5, 3]),
            2,
            0,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [False, False, True, True, False],
                    [False, False, True, True, False],
                    [False, False, False, False, True],
                ]
            ),
        ),
        (
            torch.tensor([2, 5]),
            2,
            1,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, False, False, True, True],
                ]
            ),
        ),
    ],
)
def test_streaming_mask(length: torch.Tensor, chunk_size: int, history_size: int, desired: torch.Tensor):
    actual = streaming_mask(length, chunk_size, history_size)
    assert torch.equal(actual, desired)
