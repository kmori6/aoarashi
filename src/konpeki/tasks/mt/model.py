import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from konpeki.modules.transformer import Decoder, Encoder, PositionalEncoding
from konpeki.utils.mask import causal_mask, sequence_mask


@dataclass
class Hypothesis:
    token: list[int]
    total_score: float


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout_rate: float,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        label_smoothing: float,
        ignore_token_id: int = -100,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.ignore_token_id = ignore_token_id
        self.scale = math.sqrt(d_model)
        self.input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.output_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout_rate)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # share the same weight matrix
        self.output_embedding = self.input_embedding
        self.linear.weight = self.input_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(
        self,
        enc_token: torch.Tensor,
        enc_token_length: torch.Tensor,
        dec_token: torch.Tensor,
        dec_token_length: torch.Tensor,
        tgt_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            enc_token (torch.Tensor): Encoder input token sequence (batch, seq1).
            enc_token_length (torch.Tensor): Encoder input token length (batch,).
            dec_token (torch.Tensor): Decoder input token sequence (batch, seq2).
            dec_token_length (torch.Tensor): Decoder input token length (batch,).
            tgt_token (torch.Tensor): Target token sequence (batch, seq2).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - torch.Tensor: Loss value.
                - dict[str, torch.Tensor]: Training statistics dictionary.
        """
        # encoder
        mask_enc = sequence_mask(enc_token_length)[:, None, None, :]  # (batch, 1, 1, seq1)
        x_enc = self.input_embedding(enc_token)  # (batch, seq1, d_model)
        x_enc = x_enc * self.scale + self.positional_encoding(x_enc.shape[1], x_enc.dtype, x_enc.device)[None, :, :]
        x_enc = self.dropout1(x_enc)
        x_enc = self.encoder(x_enc, mask_enc)
        # decoder
        x_dec = self.output_embedding(dec_token)  # (batch, seq2, d_model)
        x_dec = x_dec * self.scale + self.positional_encoding(x_dec.shape[1], x_dec.dtype, x_dec.device)
        x_dec = self.dropout2(x_dec)
        mask_dec = causal_mask(dec_token_length)[:, None, :, :]  # (batch, 1, seq2, seq2)
        x_dec = self.decoder(x_enc, x_dec, mask_enc, mask_dec)
        x_dec = self.linear(x_dec)  # (batch, seq2, vocab_size)
        # loss
        loss = self.loss_fn(x_dec.flatten(0, 1), tgt_token.flatten()) / x_dec.shape[0]
        mask_valid = tgt_token != self.ignore_token_id
        acc = (x_dec.argmax(-1)[mask_valid] == tgt_token[mask_valid]).sum() / mask_valid.sum()
        return loss, {"loss": loss.item(), "acc": acc.item()}

    @torch.no_grad()
    def translate(
        self, enc_token: torch.Tensor, beam_size: int, length_buffer: int, length_penalty: float
    ) -> Hypothesis:
        # encode
        mask_enc = torch.ones_like(enc_token, dtype=torch.bool)[:, None, None, :]  # (1, 1, 1, seq1)
        x_enc = self.input_embedding(enc_token)  # (1, seq1, d_model)
        x_enc = x_enc * self.scale + self.positional_encoding(x_enc.shape[1], x_enc.dtype, x_enc.device)
        x_enc = self.encoder(x_enc, mask_enc)
        # decode
        hyps = [Hypothesis(token=[self.bos_token_id], total_score=0.0)]
        # NOTE: add a minimum sample to prevent an empty hypothesis
        final_hyps = [Hypothesis(token=[self.bos_token_id, self.eos_token_id], total_score=float("-inf"))]
        for _ in range(enc_token.shape[1] + length_buffer):
            best_hyps = []
            for hyp in hyps:
                token_dec = torch.tensor([hyp.token], dtype=torch.long, device=enc_token.device)  # (1, seq2)
                mask_dec = torch.ones(
                    1, 1, len(hyp.token), len(hyp.token), dtype=torch.bool, device=enc_token.device
                )  # (1, 1, seq2, seq2)
                x_dec = self.output_embedding(token_dec)  # (batch, seq2, d_model)
                x_dec = x_dec * self.scale + self.positional_encoding(x_dec.shape[1], x_dec.dtype, x_dec.device)
                x_dec = self.decoder(x_enc, x_dec, mask_enc, mask_dec)
                x_dec = self.linear(x_dec)  # (1, seq2, vocab_size)
                scores = torch.log_softmax(x_dec[0, -1, :], dim=-1)  # (vocab_size,)
                # NOTE: limit the number of candidates to reduce computation
                for score, k in zip(*torch.topk(scores, beam_size)):
                    best_hyps.append(Hypothesis(token=hyp.token + [int(k)], total_score=hyp.total_score + score.item()))
                best_hyps = sorted(best_hyps, key=lambda x: x.total_score, reverse=True)[:beam_size]
            for best_hyp in best_hyps:
                if best_hyp.token[-1] == self.eos_token_id:
                    final_hyps.append(best_hyp)
            if len(final_hyps) > beam_size:
                break
            hyps = best_hyps
        return max(final_hyps, key=lambda x: x.total_score / (len(x.token) ** length_penalty))
