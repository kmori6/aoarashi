import math

import torch
import torch.nn as nn

from konpeki.modules.fastspeech2 import (
    ConvolutionEncoder,
    FastSpeech2Decoder,
    FastSpeech2Encoder,
    VarianceAdapter,
    random_extract_segment,
)
from konpeki.modules.hifi_gan import Generator
from konpeki.modules.rad_tts import AlignmentModule, hard_alignment
from konpeki.modules.transformer import PositionalEncoding
from konpeki.utils.mask import sequence_mask


class Jets(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        mel_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        ff_kernel_size: int,
        adapter_kernel_size: int,
        adapter_dropout_rate: float,
        dropout_rate: float,
        pad_token_id: int,
        segment_size: int,
    ):
        super().__init__()
        self.alignment_module = AlignmentModule(d_model, mel_size)
        self.encoder = FastSpeech2Encoder(
            vocab_size, d_model, d_ff, num_layers, num_heads, dropout_rate, ff_kernel_size, pad_token_id
        )
        self.decoder = FastSpeech2Decoder(d_model, d_model, d_ff, num_layers, num_heads, dropout_rate, ff_kernel_size)
        self.variance_adapter = VarianceAdapter(d_model, adapter_kernel_size, adapter_dropout_rate)
        self.generator = Generator(d_model)
        self.segment_size = segment_size

    def forward(
        self,
        token: torch.Tensor,
        token_length: torch.Tensor,
        feat: torch.Tensor,
        feat_length: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
    ):
        """
        Args:
            token (torch.Tensor): Token sequence tensor (batch, sequence).
            token_length (torch.Tensor): Token mask tensor (batch, sequence).
            feat (torch.Tensor): Feature tensor (batch, frame, n_mels).
            feat_length (torch.Tensor): Feature length tensor (batch).
            pitch (torch.Tensor): Pitch tensor (batch, frame).
            energy (torch.Tensor): Energy tensor (batch, frame).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
                dict[str, torch.Tensor]: Statistics.
        """
        token_mask = sequence_mask(token_length)  # (batch, sequence)
        feat_mask = sequence_mask(feat_length)  # (batch, frame)
        # transformer encoder
        x = self.encoder(token, token_mask)  # (batch, sequence, d_model)
        # alignment module
        log_a_soft = self.alignment_module(
            h=x, m=feat, mask=token_mask, token_length=token_length, feat_length=feat_length
        )  # (batch, sequence, frame)
        a_hard = hard_alignment(
            log_a_soft=log_a_soft, token_length=token_length, feat_length=feat_length
        )  # (batch, frame, sequence)
        duration = torch.sum(a_hard, dim=-1)  # (batch, sequence)
        # variance adapter
        x, d_hat, p_hat, e_hat = self.variance_adapter(
            x=x, duration=duration, pitch=pitch, energy=energy, mask=token_mask
        )  # (batch, frame, d_model)
        # transformer decoder
        x = self.decoder(x, feat_mask)  # (batch, frame, n_mels)
        # generator
        x = x.transpose(1, 2)  # (batch, n_mels, frame)
        x, start_frame = random_extract_segment(x, feat_length, self.segment_size)  # (batch, n_mels, segment)
        fake_wav = self.generator(x)  # (batch, 1, sample)
        return fake_wav, start_frame, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard

    @torch.no_grad()
    def generate(self, token: torch.Tensor, token_length: torch.Tensor):
        """Generate mel spectrogram from token sequence.

        Args:
            token (torch.Tensor): Token sequence tensor (batch, sequence).
            token_length (torch.Tensor): Token length tensor (batch).

        Returns:
            torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
        """
        token_mask = sequence_mask(token_length)  # (batch, sequence)
        # transformer encoder
        x = self.encoder(token, token_mask)  # (batch, sequence, d_model)
        # variance adapter
        x = self.variance_adapter.inference(x, token_mask)  # (batch, frame, d_model)
        # transformer decoder
        feat_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)  # (batch, frame)
        x = self.decoder(x, feat_mask)  # (batch, frame, n_mels)
        # generator
        x = x.transpose(1, 2)  # (batch, n_mels, frame)
        wav = self.generator(x)  # (batch, 1, sample)
        return wav


class JetsVc(nn.Module):
    def __init__(
        self,
        input_size: int,
        mel_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        ff_kernel_size: int,
        adapter_kernel_size: int,
        adapter_dropout_rate: float,
        dropout_rate: float,
        segment_size: int,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.alignment_module = AlignmentModule(d_model, mel_size)
        self.encoder = ConvolutionEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            kernel_size=ff_kernel_size,
        )
        self.decoder = ConvolutionEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            kernel_size=ff_kernel_size,
        )
        self.variance_adapter = VarianceAdapter(d_model, adapter_kernel_size, adapter_dropout_rate)
        self.generator = Generator(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(d_model)
        self.segment_size = segment_size

    def forward(
        self,
        src_feat: torch.Tensor,
        src_length: torch.Tensor,
        tgt_feat: torch.Tensor,
        tgt_length: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
    ):
        """
        Args:
            src_feat (torch.Tensor): Source feature tensor (batch_size, src_frame_length, input_size).
            src_length (torch.Tensor): Source feature length tensor (batch_size).
            tgt_feat (torch.Tensor): Target feature tensor (batch_size, tgt_frame_length, input_size).
            tgt_length (torch.Tensor): Target feature length tensor (batch_size).
            pitch (torch.Tensor): Pitch tensor (batch, frame).
            energy (torch.Tensor): Energy tensor (batch, frame).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
                dict[str, torch.Tensor]: Statistics.
        """
        src_mask = sequence_mask(src_length)  # (batch_size, src_frame_length)
        tgt_mask = sequence_mask(tgt_length)  # (batch_size, tgt_frame_length)
        # transformer encoder
        x = self.embedding(src_feat)  # (batch_size, src_frame_length, d_model)
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout1(x)
        x = self.encoder(x, src_mask)  # (batch_size, src_frame_length, d_model)
        # alignment module
        log_a_soft = self.alignment_module(
            h=x, m=tgt_feat, mask=src_mask, token_length=src_length, feat_length=tgt_length
        )  # (batch_size, src_frame_length, tgt_frame_length)
        a_hard = hard_alignment(
            log_a_soft=log_a_soft, token_length=src_length, feat_length=tgt_length
        )  # (batch_size, src_frame_length, tgt_frame_length)
        duration = torch.sum(a_hard, dim=-1)  # (batch_size, src_frame_length)
        # variance adapter
        x, d_hat, p_hat, e_hat = self.variance_adapter(
            x=x, duration=duration, pitch=pitch, energy=energy, mask=src_mask
        )  # (batch_size, tgt_frame_length, d_model)
        # transformer decoder
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout2(x)
        x = self.decoder(x, tgt_mask)  # (batch_size, tgt_frame_length, d_model)
        # generator
        x = x.transpose(1, 2)  # (batch_size, d_model, tgt_frame_length)
        x, start_frame = random_extract_segment(x, tgt_length, self.segment_size)  # (batch_size, d_model, segment_size)
        fake_wav = self.generator(x)  # (batch_size, 1, sample_length)
        return fake_wav, start_frame, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard

    @torch.no_grad()
    def generate(self, feat: torch.Tensor, feat_length: torch.Tensor):
        """Generate mel spectrogram from token sequence.

        Args:
            token (torch.Tensor): Token sequence tensor (batch, sequence).
            token_length (torch.Tensor): Token length tensor (batch).

        Returns:
            torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
        """
        src_mask = sequence_mask(feat_length)  # (batch_size, src_frame_length)
        # transformer encoder
        x = self.embedding(feat)  # (batch_size, src_frame_length, d_model)
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout1(x)
        x = self.encoder(x, src_mask)  # (batch_size, src_frame_length, d_model)
        # variance adapter
        x = self.variance_adapter.inference(x, src_mask)  # (batch, frame, d_model)
        # transformer decoder
        tgt_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)  # (batch, frame)
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout2(x)
        x = self.decoder(x, tgt_mask)  # (batch_size, tgt_frame_length, d_model)
        # generator
        x = x.transpose(1, 2)  # (batch, n_mels, frame)
        wav = self.generator(x)  # (batch, 1, sample)
        return wav
