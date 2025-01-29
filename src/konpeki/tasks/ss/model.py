import torch
import torch.nn as nn

from konpeki.modules.conv_tasnet import (
    Decoder,
    Encoder,
    ScaleInvariantSourceToNoiseRatioLoss,
    Separation,
)
from konpeki.utils.mask import sequence_mask


class Model(nn.Module):
    def __init__(
        self,
        autoencoder_filter_size: int,
        autoencoder_kernel_size: int,
        autoencoder_stride: int,
        num_speakers: int,
        bottleneck_size: int,
        convolutional_size: int,
        kernel_size: int,
        num_blocks: int,
        num_repeats: int,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.encoder = Encoder(
            filter_size=autoencoder_filter_size,
            kernel_size=autoencoder_kernel_size,
            stride=autoencoder_stride,
        )
        self.separation = Separation(
            input_size=autoencoder_filter_size,
            num_speakers=num_speakers,
            bottleneck_size=bottleneck_size,
            convolutional_size=convolutional_size,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
        )
        self.decoder = Decoder(
            filter_size=autoencoder_filter_size,
            kernel_size=autoencoder_kernel_size,
            stride=autoencoder_stride,
        )
        self.loss_fn = ScaleInvariantSourceToNoiseRatioLoss()

    def forward(
        self,
        mix_audio: torch.Tensor,
        mix_audio_length: torch.Tensor,
        src1_audio: torch.Tensor,
        src2_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            mix_audio (torch.Tensor): Mixed audio tensor (batch_size, sample_length).
            mix_audio_length (torch.Tensor): Mixed audio length tensor (batch_size).
            src1_audio (torch.Tensor): Source speaker 1 audio tensor (batch_size, sample_length).
            src2_audio (torch.Tensor): Source speaker 2 audio tensor (batch_size, sample_length).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Loss.
                dict[str, torch.Tensor]: Statistics.
        """
        x, length = self.encoder(mix_audio, mix_audio_length)  # (batch_size, filter_size, frame_length)
        b, c, t = x.shape
        mask = self.separation(x)  # (batch_size, num_speakers, filter_size, frame_length)
        x = x[:, None, :, :] * mask  # (batch_size, num_speakers, filter_size, frame_length)
        x = x.view(b * self.num_speakers, c, t)  # (batch_size * num_speakers, filter_size, frame_length)
        x, length = self.decoder(x, length)  # (batch_size * num_speakers, sample_length)
        x = x.view(b, self.num_speakers, -1)  # (batch_size, num_speakers, sample_length)
        assert x.shape[2] == src1_audio.shape[1], f"{x.shape[2]} != {src1_audio.shape[1]}"
        assert x.shape[2] == src2_audio.shape[1], f"{x.shape[2]} != {src2_audio.shape[1]}"
        target = torch.stack([src1_audio, src2_audio], dim=1)  # (batch_size, num_speakers, sample_length)
        mask = sequence_mask(length)  # (batch_size, sample_length)
        loss = self.loss_fn(x, target, mask)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def separate(self, mix_audio: torch.Tensor) -> torch.Tensor:
        """

        Args:
            mix_audio (torch.Tensor): Mixed audio tensor (sample_length,).

        Returns:
            torch.Tensor: Separated audio tensor (batch_size, num_speakers, sample_length).
        """
        mix_audio_length = torch.tensor([len(mix_audio)], dtype=torch.long)
        mix_audio = mix_audio[None, :]  # (1, sample_length)
        x, length = self.encoder(mix_audio, mix_audio_length)  # (1, filter_size, frame_length)
        _, c, t = x.shape
        mask = self.separation(x)  # (1, num_speakers, filter_size, frame_length)
        x = x[:, None, :, :] * mask  # (1, num_speakers, filter_size, frame_length)
        x = x.view(self.num_speakers, c, t)  # (num_speakers, filter_size, frame_length)
        x, _ = self.decoder(x, length)  # (num_speakers, sample_length)
        return x
