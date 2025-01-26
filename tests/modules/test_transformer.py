import math

import pytest
import torch

from konpeki.modules.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    left_shift,
    sinusoidal_positional_encoding,
)


def test_sinusoidal_positional_encoding():
    hidden_size = 4
    sequence_length = 2
    pe = sinusoidal_positional_encoding(hidden_size, sequence_length)
    desired = torch.zeros(sequence_length, hidden_size)
    desired[0, 0] = math.sin(0.0)
    desired[0, 1] = math.cos(0.0)
    desired[0, 2] = math.sin(0.0)
    desired[0, 3] = math.cos(0.0)
    desired[1, 0] = math.sin(1.0)
    desired[1, 1] = math.cos(1.0)
    desired[1, 2] = math.sin(1.0 / (10000.0 ** (2.0 / hidden_size)))
    desired[1, 3] = math.cos(1.0 / (10000.0 ** (2.0 / hidden_size)))
    assert pe.shape == (sequence_length, hidden_size)
    assert torch.allclose(pe, desired)


@pytest.mark.parametrize(
    "M, L, desired",
    [
        (
            5,
            5,
            torch.tensor(
                [[[10, 0, 0, 0, 0], [11, 10, 0, 0, 0], [12, 11, 10, 0, 0], [13, 12, 11, 10, 0], [14, 13, 12, 11, 10]]]
            ),
        ),
        (
            5,
            3,
            torch.tensor([[[12, 11, 10, 0, 0], [13, 12, 11, 10, 0], [14, 13, 12, 11, 10]]]),
        ),
    ],
)
def test_left_shift(M: int, L: int, desired: torch.Tensor):
    x = torch.arange(M + L - 1, -1, -1).repeat(1, L, 1) + 10
    y = left_shift(x)
    assert torch.equal(y, desired)


def test_positional_encoding():
    batch_size = 2
    sequence_length = 4
    hidden_size = 8
    module = PositionalEncoding(hidden_size, sequence_length)
    x = torch.randn(batch_size, sequence_length, hidden_size, requires_grad=True)
    y = module(x)
    # shape check
    assert y.shape == (1, sequence_length, hidden_size)
    # gradient check
    x = x + y
    x.sum().backward()


def test_multi_head_attention():
    batch_size = 2
    target_sequence_length = 3
    source_sequence_length = 2
    hidden_size = 8
    num_heads = 2
    module = MultiHeadAttention(hidden_size, num_heads, dropout_rate=0.1)
    x = torch.randn(batch_size, target_sequence_length, hidden_size, requires_grad=True)
    y = torch.randn(batch_size, source_sequence_length, hidden_size, requires_grad=True)
    mask = torch.tensor(
        [
            [[True, False], [True, False], [True, False]],
            [[True, True], [True, True], [True, True]],
        ],
    )
    z = module(x, y, y, mask)
    # shape check
    assert z.shape == (batch_size, target_sequence_length, hidden_size)
    # gradient check
    z.sum().backward()


def test_feed_forward():
    batch_size = 2
    sequence_length = 4
    hidden_size = 8
    module = FeedForward(input_size=hidden_size, hidden_size=2 * hidden_size, dropout_rate=0.1)
    x = torch.randn(batch_size, sequence_length, hidden_size, requires_grad=True)
    y = module(x)
    # shape check
    assert y.shape == (batch_size, sequence_length, hidden_size)
    # gradient check
    y.sum().backward()


def test_encoder_layer():
    batch_size = 2
    sequence_length = 3
    hidden_size = 8
    num_heads = 2
    module = EncoderLayer(d_model=hidden_size, num_heads=num_heads, d_ff=2 * hidden_size, dropout_rate=0.1)
    x = torch.randn(batch_size, sequence_length, hidden_size, requires_grad=True)
    mask = torch.tensor(
        [
            [[True, True, False], [True, True, False], [True, True, False]],
            [[True, True, True], [True, True, True], [True, True, True]],
        ],
    )
    y = module(x, mask)
    # shape check
    assert y.shape == (batch_size, sequence_length, hidden_size)
    # gradient check
    y.sum().backward()


def test_encoder():
    batch_size = 2
    sequence_length = 3
    hidden_size = 8
    num_heads = 2
    num_layers = 2
    module = Encoder(
        num_layers=num_layers, d_model=hidden_size, num_heads=num_heads, d_ff=2 * hidden_size, dropout_rate=0.1
    )
    x = torch.randn(batch_size, sequence_length, hidden_size, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    y = module(x, mask)
    # shape check
    assert y.shape == (batch_size, sequence_length, hidden_size)
    # gradient check
    y.sum().backward()


def test_decoder_layer():
    batch_size = 2
    target_sequence_length = 3
    source_sequence_length = 2
    hidden_size = 8
    num_heads = 2
    module = DecoderLayer(d_model=hidden_size, num_heads=num_heads, d_ff=2 * hidden_size, dropout_rate=0.1)
    x = torch.randn(batch_size, target_sequence_length, hidden_size, requires_grad=True)
    y = torch.randn(batch_size, source_sequence_length, hidden_size, requires_grad=True)
    source_mask = torch.tensor(
        [
            [[True, False], [True, False], [True, False]],
            [[True, True], [True, True], [True, True]],
        ],
    )
    target_mask = torch.tensor(
        [
            [[True, False, False], [True, True, False], [True, True, False]],
            [[True, False, False], [True, True, False], [True, True, True]],
        ],
    )
    z = module(y, x, source_mask, target_mask)
    # shape check
    assert z.shape == (batch_size, target_sequence_length, hidden_size)
    # gradient check
    z.sum().backward()


def test_decoder():
    batch_size = 2
    target_sequence_length = 3
    source_sequence_length = 2
    hidden_size = 8
    num_heads = 2
    num_layers = 2
    module = Decoder(
        num_layers=num_layers, d_model=hidden_size, num_heads=num_heads, d_ff=2 * hidden_size, dropout_rate=0.1
    )
    x = torch.randn(batch_size, target_sequence_length, hidden_size, requires_grad=True)
    y = torch.randn(batch_size, source_sequence_length, hidden_size, requires_grad=True)
    source_mask = torch.tensor([[True, False], [True, True]])
    target_mask = torch.tensor(
        [
            [[True, False, False], [True, True, False], [True, True, False]],
            [[True, False, False], [True, True, False], [True, True, True]],
        ],
    )
    z = module(y, x, source_mask, target_mask)
    # shape check
    assert z.shape == (batch_size, target_sequence_length, hidden_size)
    # gradient check
    z.sum().backward()
