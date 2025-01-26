import math
from typing import Optional

import torch
import torch.nn as nn

from konpeki.modules.energy import Energy
from konpeki.modules.fastspeech2 import VarianceLoss, extract_segment
from konpeki.modules.hifi_gan import (
    Discriminator,
    DiscriminatorAdversarialLoss,
    FeatureMatchingLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from konpeki.modules.jets import JetsVc
from konpeki.modules.log_mel_spectrogram import LogMelSpectrogram
from konpeki.modules.pitch import Pitch
from konpeki.modules.rad_tts import BinalizationLoss, ForwardSumLoss
from konpeki.utils.mask import sequence_mask


class Model(nn.Module):
    def __init__(
        self,
        mel_size: int,
        fft_size: int,
        window_size: int,
        hop_size: int,
        sample_rate: int,
        min_freq: float,
        max_freq: float,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        ff_kernel_size: int,
        adapter_kernel_size: int,
        reduction_factor: int,
        adapter_dropout_rate: float,
        dropout_rate: float,
        segment_size: int,
        lambda_fm: float,
        lambda_mel: float,
        lambda_align: float,
        lambda_var: float,
    ):
        super().__init__()
        self.generator = JetsVc(
            input_size=mel_size * reduction_factor,
            mel_size=mel_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_kernel_size=ff_kernel_size,
            adapter_kernel_size=adapter_kernel_size,
            adapter_dropout_rate=adapter_dropout_rate,
            dropout_rate=dropout_rate,
            segment_size=segment_size,
        )
        self.discriminator = Discriminator()

        # external modules
        self.mel_spectrogram = LogMelSpectrogram(
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            mel_size=mel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            from_linear=True,
        )
        self.pitch = Pitch(sample_rate=sample_rate, hop_size=hop_size)
        self.energy = Energy(fft_size=fft_size, hop_size=hop_size, window_size=window_size)
        self.upsample_scale = hop_size
        self.reduction_factor = reduction_factor
        self.segment_size = segment_size
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.lambda_align = lambda_align
        self.lambda_var = lambda_var

        self.bin_loss_fn = BinalizationLoss()
        self.fs_loss_fn = ForwardSumLoss()
        self.gen_adv_loss_fn = GeneratorAdversarialLoss()
        self.disc_adv_loss_fn = DiscriminatorAdversarialLoss()
        self.fm_loss_fn = FeatureMatchingLoss()
        self.mel_loss_fn = MelSpectrogramLoss(
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            mel_size=mel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        self.var_loss_fn = VarianceLoss()

        # NOTE: cache for discriminator training
        self.cache: Optional[dict[str, torch.Tensor]] = None

    def forward(
        self,
        src_audio: torch.Tensor,
        src_audio_length: torch.Tensor,
        tgt_audio: torch.Tensor,
        tgt_audio_length: torch.Tensor,
        discriminator_training: bool = False,
        eps: float = 1e-12,
    ):
        """
        Args:
            src_audio (torch.Tensor): Source waveform tensor (batch_size, src_sample_length).
            src_audio_length (torch.Tensor): Source waveform length tensor (batch_size,).
            tgt_audio (torch.Tensor): Target waveform tensor (batch_size, tgt_sample_length).
            tgt_audio_length (torch.Tensor): Target waveform length tensor (batch_size,).
            discriminator_training (bool): Whether training discriminator.
            eps (float): Epsilon value for numerical stability.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
                dict[str, torch.Tensor]: Statistics.
        """
        if discriminator_training:
            return self._forward_discriminator()

        # preprocessing
        # NOTE: ensure audio length is multiple of hop length so that the generator can reconstruct same length audio
        assert torch.all(tgt_audio_length % self.mel_spectrogram.hop_size == 0)
        src_feat, src_mask = self.mel_spectrogram(
            src_audio, src_audio_length
        )  # (batch_size, src_frame_length, mel_size)
        tgt_feat, tgt_mask = self.mel_spectrogram(
            tgt_audio, tgt_audio_length
        )  # (batch_size, tgt_frame_length, mel_size)
        energy, energy_mask = self.energy(tgt_audio, tgt_audio_length)  # (batch_size, src_frame_length)
        assert torch.equal(tgt_mask, energy_mask)
        pitch = self.pitch(tgt_audio, tgt_audio_length)  # (batch_size, src_frame_length')
        if pitch.shape[1] < energy.shape[1]:
            pitch = torch.cat([pitch, torch.zeros_like(pitch[:, : energy.shape[1] - pitch.shape[1]])], dim=1)
        else:
            pitch = pitch[:, : energy.shape[1]]
        assert energy.shape == pitch.shape
        # NOTE: normalize energy to follow pitch preprocessing and value scales
        src_feat = (src_feat - src_feat.mean(dim=1)[:, None, :]) / (src_feat.std(dim=1)[:, None, :] + eps)
        src_feat = src_feat.masked_fill(~src_mask[:, :, None], 0.0)
        tgt_feat = (tgt_feat - tgt_feat.mean(dim=1)[:, None, :]) / (tgt_feat.std(dim=1)[:, None, :] + eps)
        tgt_feat = tgt_feat.masked_fill(~tgt_mask[:, :, None], 0.0)
        energy = (energy - energy.mean(dim=1)[:, None]) / (energy.std(dim=1)[:, None] + eps)
        energy = energy.masked_fill(~energy_mask, 0.0)
        # NOTE: trim the last mel spectrogram frame to match length to reconstruct same length audio
        # L / hop_size + 1 -> L / hop_size
        tgt_feat, tgt_mask = tgt_feat[:, :-1, :], tgt_mask[:, :-1]  # (batch_size, tgt_frame_length, mel_size)
        energy = energy[:, :-1]  # (batch_size, tgt_frame_length)
        pitch = pitch[:, :-1]  # (batch_size, tgt_frame_length)
        src_length = src_mask.sum(dim=1)  # (batch_size,)
        tgt_length = tgt_mask.sum(dim=1)  # (batch_size,)

        # apply reduction factor
        batch_size, src_frame_length, mel_size = src_feat.shape
        new_frame_length = math.ceil(src_frame_length / self.reduction_factor)
        src_feat = nn.functional.pad(
            src_feat, (0, 0, 0, self.reduction_factor * new_frame_length - src_frame_length), "constant", 0.0
        )
        src_feat = src_feat.view(
            batch_size, new_frame_length, self.reduction_factor * mel_size
        )  # (batch_size, ceil(src_frame_length / reduction_factor), mel_size * reduction_factor)
        src_length = torch.ceil(src_length / self.reduction_factor).to(torch.long)
        src_mask = sequence_mask(src_length)  # (batch_size, ceil(src_frame_length / reduction_factor))

        # generator forward
        # NOTE: hop length equals to upsampling factor of Hifi-GAN generator
        fake_wav, start_frame, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard = self.generator(
            src_feat=src_feat,
            src_length=src_length,
            tgt_feat=tgt_feat,
            tgt_length=tgt_length,
            pitch=pitch,
            energy=energy,
        )
        tgt_audio = extract_segment(
            x=tgt_audio[:, None, :],
            start_frame=start_frame * self.upsample_scale,
            segment_size=self.segment_size * self.upsample_scale,
        )  # (batch_size, 1, segment_size * upsample_scale)
        assert fake_wav.shape[2] == self.upsample_scale * self.segment_size
        assert fake_wav.shape == tgt_audio.shape, f"{fake_wav.shape} != {tgt_audio.shape}"
        self.cache = {"audio": tgt_audio, "fake_wav": fake_wav}

        # loss calculation
        fake_xs, fake_feats = self.discriminator(fake_wav)
        _, real_feats = self.discriminator(tgt_audio)

        loss_adv = self.gen_adv_loss_fn(fake_xs)
        loss_fm = self.fm_loss_fn(real_feats=real_feats, fake_feats=fake_feats)
        loss_mel = self.mel_loss_fn(
            fake_wav=fake_wav.squeeze(1).to(torch.float32),
            real_wav=tgt_audio.squeeze(1).to(torch.float32),
            length=torch.tensor(
                [self.upsample_scale * self.segment_size] * len(fake_wav), dtype=torch.long, device=fake_wav.device
            ),
        )
        d_loss, p_loss, e_loss = self.var_loss_fn(
            d_hat=d_hat,
            d=duration,
            p_hat=p_hat,
            p=pitch,
            e_hat=e_hat,
            e=energy,
            token_mask=src_mask,
            feat_mask=tgt_mask,
        )
        loss_var = d_loss + p_loss + e_loss
        loss_fs = self.fs_loss_fn(x=log_a_soft, token_length=src_length, feat_length=tgt_length)
        loss_bin = self.bin_loss_fn(log_a_soft=log_a_soft, a_hard=a_hard)
        loss_align = loss_fs + loss_bin
        loss = (
            loss_adv
            + self.lambda_fm * loss_fm
            + self.lambda_mel * loss_mel
            + self.lambda_align * loss_align
            + self.lambda_var * loss_var
        )
        stats = {
            "loss": loss.item(),
            "loss_adv": loss_adv.item(),
            "loss_fm": loss_fm.item(),
            "loss_mel": loss_mel.item(),
            "loss_fs": loss_fs.item(),
            "loss_bin": loss_bin.item(),
            "loss_d": d_loss.item(),
            "loss_p": p_loss.item(),
            "loss_e": e_loss.item(),
        }
        return loss, stats

    def _forward_discriminator(self):
        audio = self.cache["audio"]
        fake_wav = self.cache["fake_wav"]
        self.cache = None
        # NOTE: detach all gradient computation in the generator
        fake_xs, _ = self.discriminator(fake_wav.detach())
        real_xs, _ = self.discriminator(audio)
        loss = self.disc_adv_loss_fn(real_xs=real_xs, fake_xs=fake_xs)
        stats = {"loss": loss.item()}
        return loss, stats

    @torch.no_grad()
    def synthesize(self, audio: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from token sequence.

        Args:
            audio (torch.Tensor): Audio tensor (sample_length,).

        Returns:
            torch.Tensor: Synthesized waveform tensor (batch, sample).
        """
        audio_length = torch.tensor([len(audio)], dtype=torch.long, device=audio.device)
        audio = audio[None, :]  # (1, sample_length)
        feat, feat_mask = self.mel_spectrogram(audio, audio_length)  # (1, frame_length, mel_size)
        feat_length = feat_mask.sum(dim=1)  # (1,)

        # apply reduction factor
        _, frame_length, mel_size = feat.shape
        new_frame_length = math.ceil(frame_length / self.reduction_factor)
        feat = nn.functional.pad(
            feat, (0, 0, 0, self.reduction_factor * new_frame_length - frame_length), "constant", 0.0
        )
        feat = feat.view(
            1, new_frame_length, self.reduction_factor * mel_size
        )  # (1, ceil(frame_length / reduction_factor), mel_size * reduction_factor)
        feat_length = torch.ceil(feat_length / self.reduction_factor).to(torch.long)
        wav = self.generator.generate(feat, feat_length)
        return wav
