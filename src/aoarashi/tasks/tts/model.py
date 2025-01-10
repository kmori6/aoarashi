import torch
import torch.nn as nn

from aoarashi.modules.energy import Energy
from aoarashi.modules.fastspeech2 import (
    FastSpeech2Decoder,
    FastSpeech2Encoder,
    VarianceAdapter,
    variance_loss,
)
from aoarashi.modules.hifi_gan import (
    Discriminator,
    Generator,
    discriminator_adversarial_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    mel_spectrogram_loss,
)
from aoarashi.modules.log_mel_spectrogram import LogMelSpectrogram
from aoarashi.modules.one_tts import AlignmentModule
from aoarashi.modules.pitch import Pitch
from aoarashi.modules.rad_tts import bin_loss, forward_sum_loss
from aoarashi.utils.mask import sequence_mask


class Jets(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_mel: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        ff_kernel_size: int,
        adapter_kernel_size: int,
        adapter_dropout_rate: float,
        dropout_rate: float,
        pad_token_id: int,
        random_slice_size: int,
    ):
        super().__init__()
        self.alignment_module = AlignmentModule(d_model, d_mel)
        self.encoder = FastSpeech2Encoder(
            vocab_size, d_model, d_ff, num_layers, num_heads, dropout_rate, ff_kernel_size, pad_token_id
        )
        self.decoder = FastSpeech2Decoder(d_model, d_mel, d_ff, num_layers, num_heads, dropout_rate, ff_kernel_size)
        self.variance_adapter = VarianceAdapter(d_model, adapter_kernel_size, adapter_dropout_rate)
        self.generator = Generator(d_mel)
        self.random_slice_size = random_slice_size

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
        x = self.encoder(token, token_mask[:, None, None, :])  # (batch, sequence, d_model)
        # alignment module
        duration, log_a_soft, a_hard = self.alignment_module(
            x, feat, token_length, feat_length, token_mask
        )  # (batch, sequence)
        # variance adapter
        x, d_hat, p_hat, e_hat = self.variance_adapter(
            x=x, duration=duration, pitch=pitch, energy=energy, mask=token_mask
        )  # (batch, frame, d_model)
        # transformer decoder
        x = self.decoder(x, feat_mask[:, None, None, :])  # (batch, frame, n_mels)
        # generator
        x = x.transpose(1, 2)  # (batch, n_mels, frame)
        # randomly slice mel spectrogram tensor within the feature length
        if x.shape[2] > self.random_slice_size:
            random_tail = torch.clamp(token_length - self.random_slice_size, min=1)
            start = torch.stack([torch.randint(0, tail, (1,)) for tail in random_tail], dim=0).flatten()  # (batch,)
            x = torch.stack(
                [x[b, :, s : s + self.random_slice_size] for b, s in enumerate(start)], dim=0
            )  # (batch, n_mels, random_slice_size)
        else:
            start = None
        fake_wav = self.generator(x)  # (batch, 1, sample)
        return fake_wav, start, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard

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
        x = self.encoder(token, token_mask[:, None, None, :])  # (batch, sequence, d_model)
        # variance adapter
        x = self.variance_adapter.inference(x, token_mask)  # (batch, frame, d_model)
        # transformer decoder
        feat_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)  # (batch, frame)
        x = self.decoder(x, feat_mask[:, None, None, :])  # (batch, frame, n_mels)
        # generator
        x = x.transpose(1, 2)  # (batch, n_mels, frame)
        wav = self.generator(x)  # (batch, 1, sample)
        return wav


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_mel: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        ff_kernel_size: int,
        adapter_kernel_size: int,
        adapter_dropout_rate: float,
        dropout_rate: float,
        pad_token_id: int,
        random_slice_size: int,
        lambda_fm: float,
        lambda_mel: float,
        lambda_align: float,
        lambda_var: float,
    ):
        super().__init__()
        self.generator = Jets(
            vocab_size=vocab_size,
            d_mel=d_mel,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_kernel_size=ff_kernel_size,
            adapter_kernel_size=adapter_kernel_size,
            adapter_dropout_rate=adapter_dropout_rate,
            dropout_rate=dropout_rate,
            pad_token_id=pad_token_id,
            random_slice_size=random_slice_size,
        )
        self.discriminator = Discriminator()

        # external modules
        self.mel_spectrogram = LogMelSpectrogram(
            fft_size=1024, hop_size=256, window_size=1024, mel_size=d_mel, sample_rate=22050
        )
        self.pitch = Pitch(sample_rate=22050, hop_size=256)
        self.energy = Energy(fft_size=1024, hop_size=256, window_size=1024)

        self.random_slice_size = random_slice_size
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.lambda_align = lambda_align
        self.lambda_var = lambda_var

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
            eps (float): Small value for numerical stability.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Mel spectrogram tensor (batch, frame, mel_bins).
                dict[str, torch.Tensor]: Statistics.
        """
        # NOTE: ensure audio length is multiple of hop length so that the generator can reconstruct same length audio
        assert torch.all(audio_length % self.mel_spectrogram.hop_size == 0)
        # process inputs
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
        energy = (energy - energy.mean(dim=1, keepdim=True)) / (energy.std(dim=1, keepdim=True) + eps)
        energy = energy.masked_fill(~energy_mask, 0.0)
        # NOTE: trim the last mel spectrogram frame to match length to reconstruct same length audio
        # L / hop_size + 1 -> L / hop_size
        feat, feat_mask = feat[:, :-1, :], feat_mask[:, :-1]  # (batch, frame, n_mels)
        energy = energy[:, :-1]  # (batch, frame)
        pitch = pitch[:, :-1]  # (batch, frame)
        feat_length = feat_mask.sum(dim=1)  # (batch,)

        # forward models
        fake_wav, start, d_hat, p_hat, e_hat, duration, pitch, energy, log_a_soft, a_hard = self.generator(
            token=token, token_length=token_length, feat=feat, feat_length=feat_length, pitch=pitch, energy=energy
        )
        # NOTE: hop length equals to upsampling factor of Hifi-GAN generator
        audio = torch.stack(
            [
                audio[
                    b, s * self.mel_spectrogram.hop_size : (s + self.random_slice_size) * self.mel_spectrogram.hop_size
                ]
                for b, s in enumerate(start)
            ],
        )
        audio = audio[:, None, :]  # (batch, 1, random_slice_size * hop_size)
        assert fake_wav.shape == audio.shape, f"{fake_wav.shape} != {audio.shape}"

        # loss calculation
        if discriminator_training:
            # NOTE: detach all gradient computation in the generator
            fake_xs, _ = self.discriminator(fake_wav.detach())
            real_xs, _ = self.discriminator(audio)
            loss = discriminator_adversarial_loss(real_xs=real_xs, fake_xs=fake_xs)
            stats = {"loss": loss.item()}
            return loss, stats
        else:
            fake_xs, fake_feats = self.discriminator(fake_wav)
            # NOTE: disable gradient computation in the discriminator from real audio
            with torch.no_grad():
                _, real_feats = self.discriminator(audio)

        # NOTE: stft does not support bfloat16
        wav_length = torch.tensor([fake_wav.shape[1]] * len(fake_wav), dtype=torch.long, device=fake_wav.device)
        fake_mel, _ = self.mel_spectrogram(fake_wav[:, 0, :].to(torch.float32), wav_length)  # (batch, frame, mel_bins)
        real_mel, _ = self.mel_spectrogram(audio[:, 0, :].to(torch.float32), wav_length)  # (batch, frame, mel_bins)
        loss_adv = generator_adversarial_loss(fake_xs)
        loss_fm = feature_matching_loss(real_feats=real_feats, fake_feats=fake_feats)
        loss_mel = mel_spectrogram_loss(fake_mel=fake_mel, real_mel=real_mel)
        # NOTE: convert duration into logarithmic domain for ease of prediction (FastSpeech)
        duration = torch.log(duration + eps)
        d_loss, p_loss, e_loss = variance_loss(
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
        loss_fs = forward_sum_loss(x=log_a_soft, token_length=token_length, feat_length=feat_length)
        loss_bin = bin_loss(log_a_soft=log_a_soft, a_hard=a_hard)
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
