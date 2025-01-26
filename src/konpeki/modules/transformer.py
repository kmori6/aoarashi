"""Transformer modules.

Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

"""

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(d_model: int, max_length: int) -> torch.Tensor:
    """Sinusoidal positional encoding.

    PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
    PE_{(pos, 2i + 1)} = cos(pos/10000^{2i/d_model})

    Args:
        d_model (int): Hidden state dimension.
        max_length (int): Maximum sequence length.

    Returns:
        torch.Tensor: Sinusoidal positional encoding (max_length, d_model).
    """
    pos = torch.arange(max_length, dtype=torch.float32)[:, None]  # (max_length, 1)
    theta = pos / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))  # (max_length, d_model / 2)
    pe = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1).flatten(1, -1)  # (max_length, d_model)
    return pe


def left_shift(x: torch.Tensor):
    """Shift positional tensor for relative positional calculation.

    Args:
        x (torch.Tensor): Embedding tensor (*, L, M + L).

    Returns:
        torch.Tensor: Left-shifted tensor (*, L, M).

    Examples:
        >>> M = L = 5
        >>> x = torch.arange(M + L - 1, -1, -1).repeat(1, L, 1) + 10
        >>> x
        tensor([[[19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]]])
        >>> left_shift(x)
        tensor([[[10,  0,  0,  0,  0],
                 [11, 10,  0,  0,  0],
                 [12, 11, 10,  0,  0],
                 [13, 12, 11, 10,  0],
                 [14, 13, 12, 11, 10]]])
    """
    L = x.shape[-2]
    M = x.shape[-1] - L
    assert M > 0
    return x.flatten(-2, -1)[..., 2 * L - 1 :].unfold(-1, size=M, step=M + L - 1).tril(M - L)  # (*, L, M)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_length: int = 4096):
        super().__init__()
        assert hidden_size % 2 == 0
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.register_buffer("pe", sinusoidal_positional_encoding(hidden_size, max_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Positional encoding (1, sequence_length, hidden_size).
        """
        assert len(x.shape) == 3 and x.shape[-1] == self.hidden_size and x.shape[1] <= self.max_length
        return self.pe[None, : x.shape[1], :].to(dtype=x.dtype, device=x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.h = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.d_k = hidden_size // num_heads
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        # linear matrix calculation fashion based on M. Xu et al.,
        # "Conformer-based speech recognition on extreme edge-computing devices," in NAACL, 2024, pp. 131-139.
        # KQV: (b, *, h x d_k) -> linear -> (b, *, h x d_k) -> reshape -> (b, *, h, d_k) -> transpose -> (b, h, *, d_k)
        b, t, s = q.shape[0], q.shape[1], k.shape[1]
        q = self.w_q(q).view(b, t, self.h, self.d_k).transpose(1, 2)  # (b, h, t, d_k)
        k = self.w_k(k).view(b, s, self.h, self.d_k).transpose(1, 2)  # (b, h, s, d_k)
        v = self.w_v(v).view(b, s, self.h, self.d_k).transpose(1, 2)  # (b, h, s, d_k)
        # scaled dot-product attention by pytorch function
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        x = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask[:, None, :, :],
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False,  # NOTE: mask should be already causal
            scale=1 / math.sqrt(self.d_k),
            enable_gqa=False,
        )  # (b, h, t, d_k)
        x = x.transpose(1, 2).flatten(2, -1)  # (b, t, h x d_k)
        x = self.w_o(x)
        return x


class RelativePositionalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with relative positional encoding.

    Proposed in Z. Dai et al., "Transformer-XL: attentive language models beyond a fixed-length context,"
    in ACL, 2019, pp. 2978-2988.

    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__(hidden_size, num_heads, dropout_rate)
        self.pe = PositionalEncoding(hidden_size)
        self.w_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_u = nn.Parameter(torch.empty(self.h, self.d_k), requires_grad=True)
        self.b_v = nn.Parameter(torch.empty(self.h, self.d_k), requires_grad=True)
        # NOTE: initialize parameters same as "Linear"
        # https://pytorch.org/docs/main/generated/torch.nn.Linear.html#torch.nn.Linear
        nn.init.uniform_(self.b_u, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
        nn.init.uniform_(self.b_v, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        # linear matrix calculation fashion based on M. Xu et al.,
        # "Conformer-based speech recognition on extreme edge-computing devices," in NAACL, 2024, pp. 131-139.
        # KQV: (b, *, h x d_k) -> linear -> (b, *, h x d_k) -> reshape -> (b, *, h, d_k) -> transpose -> (b, h, *, d_k)
        b, t, s = q.shape[0], q.shape[1], k.shape[1]
        q = self.w_q(q).view(b, t, self.h, self.d_k).transpose(1, 2)  # (b, h, t, d_k)
        k = self.w_k(k).view(b, s, self.h, self.d_k).transpose(1, 2)  # (b, h, s, d_k)
        v = self.w_v(v).view(b, s, self.h, self.d_k).transpose(1, 2)  # (b, h, s, d_k)
        # positional encoding where t = 2s - 1, ..., 0 (reverse order)
        p = self.pe(k.new_ones(1, 2 * s, 1)).flip(1)  # (1, 2s, h x d_k)
        p = self.w_p(p).view(1, 2 * s, self.h, self.d_k).transpose(1, 2)  # (1, h, 2s, d_k)
        # attention score in section 3.3 and appendix B
        ac = torch.matmul(q + self.b_u[None, :, None, :], k.transpose(2, 3))  # (b, h, t, s)
        bd = torch.matmul(q + self.b_v[None, :, None, :], p.transpose(2, 3))  # (b, h, t, 2s)
        bd = left_shift(bd)  # (b, h, t, s)
        x = (ac + bd) / math.sqrt(self.d_k)  # (b, h, t, s)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)  # (b, h, t, d_k)
        x = x.transpose(1, 2).flatten(2, -1)  # (b, t, h x d_k)
        x = self.w_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, input_size).
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Mask tensor (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        x = self.layer_norm1(x + self.dropout1(self.mha(x, x, x, mask)))
        x = self.layer_norm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder embedding sequence tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder embedding sequence tesor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder mask tensor
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder mask tensor (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output sequence tensor (batch_size, target_sequence_length, d_model).
        """
        x_dec = self.layer_norm1(x_dec + self.dropout1(self.masked_mha(x_dec, x_dec, x_dec, mask_dec)))
        x_dec = self.layer_norm2(x_dec + self.dropout2(self.mha(x_dec, x_enc, x_enc, mask_enc)))
        x_dec = self.layer_norm3(x_dec + self.dropout3(self.ffn(x_dec)))
        return x_dec


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Sequence mask tensor (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        mask = mask[:, None, :].expand(-1, x.shape[1], -1)  # (batch_size, sequence_length, sequence_length)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder embedding tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder embedding tensor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder sequence mask tensor
                (batch_size, source_sequence_length).
            mask_dec (torch.Tensor): Decoder causal mask tensor
                (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output sequence (batch, seq2, d_model).
        """
        mask_enc = mask_enc[:, None, :].expand(-1, x_dec.shape[1], -1)
        for layer in self.layers:
            x_dec = layer(x_enc, x_dec, mask_enc, mask_dec)
        return x_dec
