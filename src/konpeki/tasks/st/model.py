import math

import torch
import torch.nn as nn
from torchaudio.transforms import RNNTLoss

from konpeki.modules.conformer import ConvolutionSubsampling
from konpeki.modules.log_mel_spectrogram import LogMelSpectrogram
from konpeki.modules.rnn_transducer import (
    JointNetwork,
    PredictionNetwork,
    Sequence,
    beam_search,
)
from konpeki.modules.transformer import EncoderLayer, PositionalEncoding
from konpeki.utils.mask import streaming_mask


class Model(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        window_size: int,
        vocab_size: int,
        mel_size: int,
        min_freq: int,
        max_freq: int,
        sample_rate: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_blocks: int,
        hidden_size: int,
        num_layers: int,
        joint_size: int,
        chunk_size: int,
        history_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.blank_token_id = vocab_size - 1
        self.chunk_size = chunk_size
        self.history_size = history_size
        self.frontend = LogMelSpectrogram(
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            mel_size=mel_size,
            min_freq=min_freq,
            max_freq=max_freq,
            sample_rate=sample_rate,
        )
        self.embedding = ConvolutionSubsampling(d_model)
        self.linear = nn.Linear(d_model * (((mel_size - 1) // 2 - 1) // 2), d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_blocks)])
        self.predictor = PredictionNetwork(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            blank_token_id=self.blank_token_id,
        )
        self.joiner = JointNetwork(
            vocab_size=vocab_size,
            encoder_size=d_model,
            predictor_size=hidden_size,
            hidden_size=joint_size,
            dropout_rate=dropout_rate,
        )
        self.loss_fn = RNNTLoss(blank=self.blank_token_id, reduction="mean", fused_log_softmax=False)

    def forward(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        token: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            audio (torch.Tensor): Speech tensor (batch_size, sample_length).
            audio_length (torch.Tensor): Speech length tensor (batch_size,).
            token (torch.Tensor): Prediction network input token tensor
                with blank token at the head position (batch_size, sequence_length + 1).
            target (torch.Tensor): Target token tensor (batch_size, sequence_length).
            target_length (torch.Tensor): Target token length tensor (batch_size,).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Loss.
                dict[str, torch.Tensor]: Statistics.
        """
        b = audio.shape[0]
        x_enc, mask = self.frontend(audio, audio_length)  # (batch_size, frame_length, mel_size)
        x_enc, mask = self.embedding(x_enc, mask)  # (batch_size, frame_length', mel_size')
        x_enc = self.linear(x_enc)  # (batch_size, frame_length', d_model)
        x_enc = x_enc * self.scale + self.positional_encoding(x_enc)
        attn_mask = streaming_mask(mask.sum(-1), chunk_size=self.chunk_size, history_size=self.history_size)
        attn_mask = attn_mask[None, :, :].expand(b, -1, -1)  # (batch_size, frame_length', frame_length')
        for layer in self.encoder:
            x_enc = layer(x_enc, attn_mask)
        x_dec, *_ = self.predictor(
            token, *self.predictor.init_state(b, x_enc.device)
        )  # (batch_size, sequence_length, hidden_size)
        x_rnnt = self.joiner(
            x_enc[:, :, None, :], x_dec[:, None, :, :]
        )  # (batch_size, frame_length', sequence_length, vocab_size)
        #  loss
        frame_length = mask.sum(-1)
        loss = self.loss_fn(torch.log_softmax(x_rnnt, dim=-1), target.int(), frame_length.int(), target_length.int())
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def translate(
        self,
        audio_chunk: torch.Tensor,
        beam_size: int,
        chunk_size: int,
        history_size: int,
        initial_set: list = [],
        end_chunk: bool = False,
    ) -> list[Sequence]:
        """

        Args:
            audio_chunk (torch.Tensor): Speech tensor (sample_length,).
            beam_size (int): Beam size.
            initial_set (list): Initial set of sequences for beam search.
            end_chunk (bool): If True, return the highest log probability sequence at the end of the chunk.

        Returns: tuple[torch.Tensor, list[Sequence]]:
            torch.Tensor: Encoder embedding tensor (1, frame_length, d_model).
            Sequence: Decoded sequence with highest log probability.
        """
        audio_chunk = audio_chunk[None, :]
        audio_length = torch.tensor([audio_chunk.shape[1]], dtype=torch.long, device=audio_chunk.device)
        x, mask = self.frontend(audio_chunk, audio_length)  # (1, frame_length, mel_size)
        x, mask = self.embedding(x, mask)  # (1, frame_length', mel_size')
        assert x.shape[1] == history_size + chunk_size - 1
        x = self.linear(x)  # (1, frame_length', d_model)
        x = x * self.scale + self.positional_encoding(x)
        mask = mask[:, None, :].expand(-1, x.shape[1], -1)  # (1, frame_length', frame_length')
        for layer in self.encoder:
            x = layer(x, mask)
        x = x[:, history_size - 1 :, :]  # (1, chunk_size, d_model)
        assert x.shape[1] == chunk_size
        hyp = beam_search(
            x,
            prediction_network=self.predictor.eval(),
            joint_network=self.joiner.eval(),
            beam_width=beam_size,
            blank_token_id=self.blank_token_id,
            initial_set=initial_set,
            return_highest=end_chunk,
        )
        return hyp
