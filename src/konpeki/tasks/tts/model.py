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
from konpeki.modules.jets import Jets
from konpeki.modules.log_mel_spectrogram import LogMelSpectrogram
from konpeki.modules.pitch import Pitch
from konpeki.modules.rad_tts import BinalizationLoss, ForwardSumLoss
from konpeki.utils.mask import sequence_mask


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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
        adapter_dropout_rate: float,
        dropout_rate: float,
        pad_token_id: int,
        segment_size: int,
        lambda_fm: float,
        lambda_mel: float,
        lambda_align: float,
        lambda_var: float,
    ):
        super().__init__()
        self.generator = Jets(
            vocab_size=vocab_size,
            mel_size=mel_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_kernel_size=ff_kernel_size,
            adapter_kernel_size=adapter_kernel_size,
            adapter_dropout_rate=adapter_dropout_rate,
            dropout_rate=dropout_rate,
            pad_token_id=pad_token_id,
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
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        token: torch.Tensor,
        token_length: torch.Tensor,
        discriminator_training: bool = False,
        eps: float = 1e-12,
    ):
        """
        Args:
            audio (torch.Tensor): Waveform tensor (batch, sample).
            audio_length (torch.Tensor): Audio length tensor (batch).
            token (torch.Tensor): Token sequence tensor (batch, sequence).
            token_length (torch.Tensor): Token length tensor (batch).
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
        assert torch.all(audio_length % self.mel_spectrogram.hop_size == 0)
        token_mask = sequence_mask(token_length)  # (batch, sequence)
        feat, feat_mask = self.mel_spectrogram(audio, audio_length)  # (1, L / hop_size + 1, n_mels)
        energy, energy_mask = self.energy(audio, audio_length)  # (batch, L / hop_size + 1)
        assert torch.equal(feat_mask, energy_mask)
        pitch = self.pitch(audio, audio_length)  # (batch, frame')
        if pitch.shape[1] < energy.shape[1]:
            pitch = torch.cat([pitch, torch.zeros_like(pitch[:, : energy.shape[1] - pitch.shape[1]])], dim=1)
        else:
            pitch = pitch[:, : energy.shape[1]]
        assert energy.shape == pitch.shape
        # NOTE: normalize energy to follow pitch preprocessing and value scales
        feat = (feat - feat.mean(dim=1)[:, None, :]) / (feat.std(dim=1)[:, None, :] + eps)
        feat = feat.masked_fill(~feat_mask[:, :, None], 0.0)
        energy = (energy - energy.mean(dim=1)[:, None]) / (energy.std(dim=1)[:, None] + eps)
        energy = energy.masked_fill(~energy_mask, 0.0)
        # NOTE: trim the last mel spectrogram frame to match length to reconstruct same length audio
        # L / hop_size + 1 -> L / hop_size
        feat, feat_mask = feat[:, :-1, :], feat_mask[:, :-1]  # (batch, frame, n_mels)
        energy = energy[:, :-1]  # (batch, frame)
        pitch = pitch[:, :-1]  # (batch, frame)
        feat_length = feat_mask.sum(dim=1)  # (batch,)

        # generator forward
        # NOTE: hop length equals to upsampling factor of Hifi-GAN generator
        fake_wav, start_frame, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard = self.generator(
            token=token,
            token_length=token_length,
            feat=feat,
            feat_length=feat_length,
            pitch=pitch,
            energy=energy,
        )
        audio = extract_segment(
            x=audio[:, None, :],
            start_frame=start_frame * self.upsample_scale,
            segment_size=self.segment_size * self.upsample_scale,
        )  # (batch_size, 1, segment_size * upsample_scale)
        assert fake_wav.shape[2] == self.upsample_scale * self.segment_size
        assert fake_wav.shape == audio.shape, f"{fake_wav.shape} != {audio.shape}"
        self.cache = {"audio": audio, "fake_wav": fake_wav}

        # loss calculation
        fake_xs, fake_feats = self.discriminator(fake_wav)
        _, real_feats = self.discriminator(audio)

        loss_adv = self.gen_adv_loss_fn(fake_xs)
        loss_fm = self.fm_loss_fn(real_feats=real_feats, fake_feats=fake_feats)
        loss_mel = self.mel_loss_fn(
            fake_wav=fake_wav.squeeze(1).to(torch.float32),
            real_wav=audio.squeeze(1).to(torch.float32),
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
            token_mask=token_mask,
            feat_mask=feat_mask,
        )
        loss_var = d_loss + p_loss + e_loss
        loss_fs = self.fs_loss_fn(x=log_a_soft, token_length=token_length, feat_length=feat_length)
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
    def synthesize(self, token: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from token sequence.

        Args:
            token (torch.Tensor): Token sequence tensor (sequence).

        Returns:
            torch.Tensor: Synthesized waveform tensor (batch, sample).
        """
        token_length = torch.tensor([len(token)], dtype=torch.long, device=token.device)
        token = token[None, :]
        wav = self.generator.generate(token, token_length)
        return wav
