"""HiFi-GAN Modules (version 1).

Proposed in J. Kong et al., "HiFi-GAN: generative adversarial networks
for efficient and high fidelity speech synthesis," in NeurIPS, 2020, pp. 17022-17033.

"""

import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from konpeki.modules.log_mel_spectrogram import LogMelSpectrogram


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, fake_xs: list[torch.Tensor]) -> torch.Tensor:
        """

        L_adv = sum_{k=1}^K E_s[(D_k(G(s)) - 1)^2]
            - G: generator
            - D_k: k-th discriminator
            - K: number of discriminators
            - s: input condition

        Args:
            fake_xs (list[torch.Tensor]): Discriminator output tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Generator adversarial loss.
        """
        losses = []
        for fake_x in fake_xs:
            assert len(fake_x.shape) == 2  # (batch, *)
            loss = self.mse_loss_fn(fake_x, torch.ones_like(fake_x))
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, real_xs: list[torch.Tensor], fake_xs: list[torch.Tensor]) -> torch.Tensor:
        """

        L_adv = sum_{k=1}^K E_(x,s)[(D_k(x) - 1)^2 + (D_k(G(s)))^2]
            - G: generator
            - D_k: k-th discriminator
            - K: number of discriminators
            - x: ground truth audio
            - s: input condition

        Args:
            real_xs (list[torch.Tensor]): Discriminator output tensors from the ground truth audio (batch, *).
            fake_xs (list[torch.Tensor]): Discriminator output tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Discriminator adversarial loss.
        """
        losses = []
        for real_x, fake_x in zip(real_xs, fake_xs):
            assert real_x.shape == fake_x.shape and len(real_x.shape) == 2  # (batch, *)
            loss = self.mse_loss_fn(real_x, torch.ones_like(real_x)) + self.mse_loss_fn(
                fake_x, torch.zeros_like(fake_x)
            )
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, real_feats: list[list[torch.Tensor]], fake_feats: list[list[torch.Tensor]]) -> torch.Tensor:
        """

        L_FM(G; D) = sum_{k=1}^K E_(x,s)[sum_{t=1}^T 1/N_i |D^i_t(x) - D^i_t(G(s))|]

        Args:
            real_feats (list[torch.Tensor]): Discriminator feature tensors from the ground truth audio (batch, *).
            fake_feats (list[torch.Tensor]): Discriminator feature tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Feature matching loss.
        """
        losses = []
        for real_disc_feats, fake_disc_feats in zip(real_feats, fake_feats):
            assert len(real_disc_feats) == len(fake_disc_feats)
            for real_layer_feats, fake_layer_feats in zip(real_disc_feats, fake_disc_feats):
                assert real_layer_feats.shape == fake_layer_feats.shape
                # NOTE: L1 loss averaged over the batch and feature size
                loss = self.l1_loss_fn(real_layer_feats, fake_layer_feats)
                losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        window_size: int,
        mel_size: int,
        sample_rate: int,
        min_freq: float,
        max_freq: float,
    ):
        super().__init__()
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
        self.l1_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, fake_wav: torch.Tensor, real_wav: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """

        Args:
            fake_wav (torch.Tensor): Waveform tensor from the generator (batch, sample).
            real_wav (torch.Tensor): Waveform tensor from the ground truth audio (batch, sample).
            length (torch.Tensor): Waveform length tensor (batch).

        Returns:
            torch.Tensor: Mel-spectrogram loss.
        """
        assert fake_wav.shape == real_wav.shape and length.max() == fake_wav.shape[1]
        fake_mel, _ = self.mel_spectrogram(fake_wav, length)  # (batch, frame, mel_size)
        real_mel, _ = self.mel_spectrogram(real_wav, length)  # (batch, frame, mel_size)
        loss = self.l1_loss_fn(fake_mel, real_mel)
        return loss


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilations: list[int], negative_slope: float):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert len(dilations) == 3
        self.stacks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    weight_norm(
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size,
                            stride=1,
                            padding=(dilation * (kernel_size - 1)) // 2,
                            dilation=dilation,
                        )
                    ),
                    nn.LeakyReLU(negative_slope),
                    weight_norm(
                        nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
                    ),
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor of shape (batch, hidden_size, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, hidden_size, length).
        """
        for stack in self.stacks:
            x_res = x
            x = stack(x)
            x = x_res + x
        return x


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(
        self, hidden_dim: int, negative_slope: float, kernel_sizes: list[int], dilations_list: list[list[int]]
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(hidden_dim, kernel_size, dilations, negative_slope)
                for kernel_size, dilations in zip(kernel_sizes, dilations_list)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor of shape (batch, hidden_size, frame).

        Returns:
            torch.Tensor: Output tensor of shape (batch, hidden_size, frame).
        """
        xs = [block(x) for block in self.blocks]  # list of (batch, hidden_size, frame)
        x = torch.stack(xs, dim=1)  # (batch, block, hidden_size, frame)
        x = torch.mean(x, dim=1)  # (batch, hidden_size, frame)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        residual_kernel_sizes: list[int] = [3, 7, 11],
        dilations_list: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.conv_input = weight_norm(nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3))
        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    weight_norm(
                        nn.ConvTranspose1d(
                            hidden_size // 2 ** (i - 1),
                            hidden_size // 2**i,
                            kernel_size=kernel_size,
                            stride=kernel_size // 2,
                            padding=kernel_size // 4,
                        )
                    ),
                    MultiReceptiveFieldFusion(
                        hidden_size // 2**i, negative_slope, residual_kernel_sizes, dilations_list
                    ),
                )
                for i, kernel_size in enumerate(upsample_kernel_sizes, start=1)
            ]
        )
        self.leaky_relu = nn.LeakyReLU()
        self.conv_out = weight_norm(
            nn.Conv1d(hidden_size // (2 ** len(upsample_kernel_sizes)), 1, kernel_size=7, padding=3)
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Mel-spectrogram tensor (batch, mel_size, frame).

        Returns:
            torch.Tensor: Waveform tensor (batch, 1, 256 * frame).
        """
        x = self.conv_input(x)  # (batch, hidden_size, frame)
        for upsample in self.upsamples:
            x = upsample(x)
        x = self.leaky_relu(x)  # (batch, hidden_size / 2^4, frame)
        x = self.conv_out(x)  # (batch, 1, frame)
        x = self.tanh(x)
        return x


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, negative_slope: float):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                # NOTE: set padding size to 1 to make L_out = L_in // 3
                nn.Sequential(
                    weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
            ]
        )
        self.layer = nn.Sequential(
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), padding=(2, 0))),
            nn.LeakyReLU(negative_slope),
        )
        self.conv_out = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                torch.Tensor: Logit tensor (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *, *).
        """
        feats = []
        # pad and reshape (batch, 1, sample) -> (batch, 1, ceil(sample / period), period)
        b, _, t = x.shape
        new_t = math.ceil(t / self.period)
        # pad only the tail of the last dimension
        x = torch.nn.functional.pad(x, (0, self.period * new_t - t), "reflect")
        x = x.view(b, 1, new_t, self.period)
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.layer(x)  # (batch, 1024, *, *)
        feats.append(x)
        x = self.conv_out(x)  # (batch, 1, *, *)
        feats.append(x)
        x = torch.flatten(x, 1, -1)  # (batch, *)
        return x, feats


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale: int, negative_slope: float):
        super().__init__()
        self.scale = scale
        if scale > 1:
            assert scale % 2 == 0
            self.avg_pool = nn.ModuleList(
                [
                    nn.AvgPool1d(
                        kernel_size=4,
                        stride=2,
                        padding=1,  # padding to make L_out = L_in // 2
                    )
                    for _ in range(scale // 2)
                ]
            )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                    nn.LeakyReLU(negative_slope),
                ),
            ]
        )
        self.conv_out = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                torch.Tensor: Logit tensor (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *).
        """
        feats = []
        if self.scale > 1:
            for avg_pool in self.avg_pool:
                x = avg_pool(x)  # (batch, 1, sample // *)
        for layer in self.layers:
            x = layer(x)  # (batch, *, *)
            feats.append(x)
        x = self.conv_out(x)  # (batch, 1, *)
        feats.append(x)
        x = torch.flatten(x, 1, -1)  # (batch, *)
        return x, feats


class Discriminator(nn.Module):
    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11], scales: list[int] = [1]):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDiscriminator(period, negative_slope=0.1) for period in periods])
        self.msd = nn.ModuleList([ScaleDiscriminator(scale, negative_slope=0.1) for scale in scales])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
                list[torch.Tensor]: Logit tensors (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *).
        """
        x_list, feats_list = [], []
        for mpd in self.mpd:
            x_o, feats = mpd(x)
            x_list.append(x_o)
            feats_list.append(feats)
        for msd in self.msd:
            x_o, feats = msd(x)
            x_list.append(x_o)
            feats_list.append(feats)
        return x_list, feats_list
