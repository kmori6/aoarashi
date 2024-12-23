"""Transformer modules.

Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.max_length = max_length
        self._init_encoding()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d_model={self.d_model}, max_length={self.max_length})"

    def _init_encoding(self):
        """Initialize positional encoding.

        PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})
        PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})

        """
        pos = torch.arange(self.max_length, dtype=torch.float32)[:, None]  # (max_length, 1)
        theta = pos / 10000 ** (
            torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model
        )  # (max_length, d_model / 2)
        pe = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1).flatten(1)  # (max_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """

        Args:
            length (int): Sequence length.
            dtype (torch.dtype): Positional encoding dtype.
            device (torch.device): Positional encoding device.

        Returns:
            torch.Tensor: Positional encoding (length, d_model).
        """
        return self.pe[:length, :].to(dtype=dtype, device=device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.w_q = nn.Parameter(torch.empty(self.h, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.empty(self.h, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.empty(self.h, d_model, self.d_k))
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        # NOTE: initialize parameters same as "Linear"
        # https://pytorch.org/docs/main/generated/torch.nn.Linear.html#torch.nn.Linear
        nn.init.uniform_(self.w_q, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))
        nn.init.uniform_(self.w_k, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))
        nn.init.uniform_(self.w_v, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + f"  (w_q): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_k): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_v): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_o): Linear(in_features={self.d_model}, out_features={self.d_model}, bias=False)\n"
            + ")"
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query sequence (batch, seq1, d_model).
            k (torch.Tensor): Key sequence (batch, seq2, d_model).
            v (torch.Tensor): Value sequence (batch, seq2, d_model).
            mask (torch.Tensor): Mask sequence (batch, 1, 1 or seq1, seq2).

        Returns:
            torch.Tensor: Output sequence (batch, seq1, d_model).
        """
        q = torch.matmul(q[:, None, :, :], self.w_q)  # (batch, head, seq1, d_k)
        k = torch.matmul(k[:, None, :, :], self.w_k)  # (batch, head, seq2, d_k)
        v = torch.matmul(v[:, None, :, :], self.w_v)  # (batch, head, seq2, d_k)
        x = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)  # (batch, head, seq1, seq2)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)  # (batch, head, seq1, d_k)
        x = x.transpose(1, 2).flatten(2, 3)  # (batch, seq1, d_model)
        x = self.w_o(x)
        return x


class RelativePositionalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with relative position embedding.

    Proposed in Z. Dai et al., "Transformer-XL: attentive language models beyond a fixed-length context,"
    in ACL, 2019, pp. 2978-2988.

    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__(d_model, num_heads)
        self.pe = PositionalEncoding(d_model)
        self.w_p = nn.Parameter(torch.empty(self.h, d_model, self.d_k))
        self.b_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.b_v = nn.Parameter(torch.empty(self.h, self.d_k))
        # NOTE: initialize parameters same as "Linear"
        # https://pytorch.org/docs/main/generated/torch.nn.Linear.html#torch.nn.Linear
        nn.init.uniform_(self.w_p, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))
        nn.init.uniform_(self.b_u, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))
        nn.init.uniform_(self.b_v, -math.sqrt(1 / d_model), math.sqrt(1 / d_model))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + f"  (w_q): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_k): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_v): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (w_p): Parameter(num_heads={self.h}, in_features={self.d_model}, out_features={self.d_k})\n"
            + f"  (b_u): Parameter(num_heads={self.h}, out_features={self.d_k})\n"
            + f"  (b_v): Parameter(num_heads={self.h}, out_features={self.d_k})\n"
            + f"  (w_o): Linear(in_features={self.d_model}, out_features={self.d_model}, bias=False)\n"
            + ")"
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query sequence (batch, seq1, d_model).
            k (torch.Tensor): Key sequence (batch, seq2, d_model).
            v (torch.Tensor): Value sequence (batch, seq2, d_model).
            mask (torch.Tensor): Mask sequence (batch, 1, 1 or seq1, seq2).

        Returns:
            torch.Tensor: Output sequence (batch, seq1, d_model).
        """
        p = self.pe(2 * k.shape[1], k.dtype, k.device).flip(0)  # (1, 2 * seq2, d_model)
        p = torch.matmul(p[:, None, :, :], self.w_p)  # (batch, head, 2 * seq2, d_k)
        s1, s2 = q.shape[1], k.shape[1]
        q = torch.matmul(q[:, None, :, :], self.w_q)  # (batch, head, seq1, d_k)
        k = torch.matmul(k[:, None, :, :], self.w_k)  # (batch, head, seq2, d_k)
        v = torch.matmul(v[:, None, :, :], self.w_v)  # (batch, head, seq2, d_k)
        # attention score in section 3.3 and appendix B
        ac = torch.matmul(q + self.b_u[None, :, None, :], k.transpose(2, 3))  # (batch, head, seq1, seq2)
        bd = torch.matmul(q + self.b_v[None, :, None, :], p.transpose(2, 3))  # (batch, head, seq1, 2 * seq2)
        bd = bd.flatten(2)[..., s1 + s2 - 1 :].unfold(-1, s2, 2 * s2 - 1).tril(s2 - s1)  # (batch, head, seq1, seq2)
        x = (ac + bd) / math.sqrt(self.d_k)  # (batch, head, seq1, seq2)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)  # (batch, head, seq1, d_k)
        x = x.transpose(1, 2).flatten(2, 3)  # (batch, seq1, d_model)
        x = self.w_o(x)
        return x


class RotaryPositionalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with rotary position embedding.

    Proposed in J. Su et al., "RoFormer: enhanced transformer with rotary position embedding,"
    arXiv preprint arXiv:2104.09864, 2021.


    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__(d_model, num_heads)
        self.pe = PositionalEncoding(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query sequence (batch, seq1, d_model).
            k (torch.Tensor): Key sequence (batch, seq2, d_model).
            v (torch.Tensor): Value sequence (batch, seq2, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1 or seq1, seq2).

        Returns:
            torch.Tensor: Output sequence (batch, seq1, d_model).
        """
        q = torch.matmul(q[:, None, :, :], self.w_q)  # (batch, head, seq1, d_k)
        k = torch.matmul(k[:, None, :, :], self.w_k)  # (batch, head, seq2, d_k)
        v = torch.matmul(v[:, None, :, :], self.w_v)  # (batch, head, seq2, d_k)
        # rotary position embedding described in the section 3.4.2
        p_q = self.pe(q.shape[2], q.dtype, q.device)[None, None, :, :]  # (1, 1, seq1, d_k)
        p_qsin, p_qcos = p_q[..., 0::2].repeat_interleave(2, dim=-1), p_q[..., 1::2].repeat_interleave(2, dim=-1)
        p_k = self.pe(k.shape[2], q.dtype, q.device)[None, None, :, :]  # (1, 1, seq2, d_k)
        p_ksin, p_kcos = p_k[..., 0::2].repeat_interleave(2, dim=-1), p_k[..., 1::2].repeat_interleave(2, dim=-1)
        q = q * p_qcos + torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1).flatten(3) * p_qsin
        k = k * p_kcos + torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1).flatten(3) * p_ksin
        x = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)  # (batch, head, seq1, seq2)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)  # (batch, head, seq1, d_k)
        x = x.transpose(1, 2).flatten(2, 3)  # (batch, seq1, d_model)
        x = self.w_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, seq, d_model).

        Returns:
            torch.Tensor: Output sequence (batch, seq, d_model).
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, seq, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, seq).

        Returns:
            torch.Tensor: Output sequence (batch, seq, d_model).
        """
        x = self.layer_norm1(x + self.dropout1(self.mha(x, x, x, mask)))
        x = self.layer_norm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder sequence (batch, seq1, d_model).
            x_dec (torch.Tensor): Decoder sequence (batch, seq2, d_model).
            mask_enc (torch.Tensor): Encoder mask (batch, 1, 1, seq1).
            mask_dec (torch.Tensor): Decoder mask (batch, 1, seq2, seq2).

        Returns:
            torch.Tensor: Output sequence (batch, seq2, d_model).
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
            x (torch.Tensor): Input sequence (batch, seq, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, seq).

        Returns:
            torch.Tensor: Output sequence (batch, seq, d_model).
        """
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
            x_enc (torch.Tensor): Encoder sequence (batch, seq1, d_model).
            x_dec (torch.Tensor): Decoder sequence (batch, seq2, d_model).
            mask_enc (torch.Tensor): Encoder mask (batch, 1, 1, seq1).
            mask_dec (torch.Tensor): Decoder mask (batch, 1, seq2, seq2).

        Returns:
            torch.Tensor: Output sequence (batch, seq2, d_model).
        """
        for layer in self.layers:
            x_dec = layer(x_enc, x_dec, mask_enc, mask_dec)
        return x_dec
