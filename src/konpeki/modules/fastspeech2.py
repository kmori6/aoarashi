"""FastSpeech2 modules.

Proposed in Y. Ren et al., "FastSpeech 2: fast and high-quality end-to-end text to speech," in ICLR, 2021.

"""

import math

import torch
import torch.nn as nn

from konpeki.modules.eats import GaussianResampling
from konpeki.modules.transformer import Encoder, FeedForward, PositionalEncoding


def extract_segment(x: torch.Tensor, start_frame: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Extract a segment from the input tensor based on the start frame and segment size.

    Args:
        x (torch.Tensor): Sequence tensor (batch_size, hidden_size, frame_length).
        start_frame (torch.Tensor): Start frame tensor (batch_size).
        segment_size (int): Segment size.

    Returns:
        torch.Tensor: Extracted segment tensor (batch_size, hidden_size, segment_size).
    """
    batch_size, hidden_size, frame_size = x.shape
    assert frame_size >= segment_size
    x_new = torch.zeros(batch_size, hidden_size, segment_size, device=x.device, dtype=x.dtype)
    for b in range(batch_size):
        x_new[b, :, :] = x[b, :, start_frame[b] : start_frame[b] + segment_size]
    return x_new


def random_extract_segment(
    x: torch.Tensor, feat_length: torch.Tensor, segment_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly extract a segment from the input tensor.

    Args:
        x (torch.Tensor): Sequence tensor (batch_size, hidden_size, frame_length).
        feat_length (torch.Tensor): Feature length tensor (batch_size).
        segment_size (int): Segment size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: Extracted segment tensor (batch_size, hidden_size, segment_size).
            torch.Tensor: Start frame tensor (batch_size).
    """
    batch_size, _, frame_size = x.shape
    assert frame_size >= segment_size
    start_frame = torch.randint(0, frame_size - segment_size + 1, (batch_size,), device=x.device)
    # NOTE: do not use padding frames
    start_frame = torch.min(start_frame, torch.clamp(feat_length - segment_size, min=0))
    assert torch.all(start_frame >= 0)
    x_new = extract_segment(x, start_frame, segment_size)
    return x_new, start_frame


class VarianceLoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss(reduction="none")
        self.eps = eps

    def forward(
        self,
        d_hat: torch.Tensor,
        d: torch.Tensor,
        p_hat: torch.Tensor,
        p: torch.Tensor,
        e_hat: torch.Tensor,
        e: torch.Tensor,
        token_mask: torch.Tensor,
        feat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            d_hat (torch.Tensor): Predicted duration in logarithmic domain tensor (batch_size, sequence_length).
            d (torch.Tensor): Target duration tensor (batch_size, sequence_length).
            p_hat (torch.Tensor): Predicted pitch tensor (batch_size, frame_length).
            p (torch.Tensor): Target pitch tensor (batch_size, frame_length).
            e_hat (torch.Tensor): Predicted energy tensor (batch_size, frame_length).
            e (torch.Tensor): Target energy tensor (batch_size, frame_length).
            token_mask (torch.Tensor): Token embedding mask tensor (batch_size, sequence_length).
            feat_mask (torch.Tensor): Mel-spectrogram mask tensor (batch_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.Tensor: Duration loss.
                torch.Tensor: Pitch loss.
                torch.Tensor: Energy loss.

        """
        assert d_hat.shape == d.shape, f"{d_hat.shape} != {d.shape}"
        assert p_hat.shape == p.shape, f"{p_hat.shape} != {p.shape}"
        assert e_hat.shape == e.shape, f"{e_hat.shape} != {e.shape}"
        # NOTE: convert duration into logarithmic domain for ease of prediction (FastSpeech)
        d = torch.log(d + self.eps)
        duration_loss = self.mse_loss_fn(d_hat, d)
        duration_loss = duration_loss.masked_fill(~token_mask, 0.0).sum() / token_mask.sum()
        pitch_loss = self.mse_loss_fn(p_hat, p)
        pitch_loss = pitch_loss.masked_fill(~feat_mask, 0.0).sum() / feat_mask.sum()
        energy_loss = self.mse_loss_fn(e_hat, e)
        energy_loss = energy_loss.masked_fill(~feat_mask, 0.0).sum() / feat_mask.sum()
        return duration_loss, pitch_loss, energy_loss


class Predictor(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(input_size, input_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, *, input_size).

        Returns:
            torch.Tensor: Predicted tensor (batch_size, *).
        """
        x = torch.transpose(x, 1, 2)  # (batch_size, input_size, *)
        x = self.conv1(x)  # (batch_size, input_size, *)
        x = self.relu(x)
        x = self.layer_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.layer_norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)
        x = self.linear(x.transpose(1, 2)).squeeze(2)  # (batch_size, *)
        return x


class ConvolutionFeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.w_1 = nn.Conv1d(input_size, hidden_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = nn.Conv1d(hidden_size, input_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, input_size).
        """
        x = torch.transpose(x, 1, 2)  # (batch_size, input_size, sequence_length)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = torch.transpose(x, 1, 2)  # (batch_size, sequence_length, input_size)
        return x


class ConvolutionEncoder(Encoder):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float, kernel_size: int):
        super().__init__(d_model, num_heads, d_ff, num_layers, dropout_rate)
        for name, module in self.layers.named_modules():
            if isinstance(module, FeedForward):
                new_module = ConvolutionFeedForward(d_model, d_ff, kernel_size, dropout_rate)
                # get the parent module
                parent_key, child_key = name.rsplit(".", 1)
                parent_module = self.layers.get_submodule(parent_key)
                setattr(parent_module, child_key, new_module)


class FastSpeech2Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        kernel_size: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = ConvolutionEncoder(d_model, num_heads, d_ff, num_layers, dropout_rate, kernel_size)

    def forward(self, token: torch.Tensor, mask: torch.Tensor):
        """

        Args:
            token (torch.Tensor): Token sequence tensor (batch_size, sequence_length).
            mask (torch.Tensor): Token mask tensor (batch_size, sequence_length).

        Returns:
            torch.Tensor: Token embedding tensor (batch_size, sequence_length, d_model).
        """
        x = self.embedding(token)  # (batch_size, sequence_length, d_model)
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout(x)
        h = self.encoder(x, mask)
        return h


class FastSpeech2Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_size: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        kernel_size: int,
    ):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = ConvolutionEncoder(d_model, num_heads, d_ff, num_layers, dropout_rate, kernel_size)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """

        Args:
            token (torch.Tensor): Audio feature tensor (batch_size, frame_length, d_model).
            mask (torch.Tensor): Audio feature mask tensor (batch_size, frame_length).

        Returns:
            torch.Tensor: Audio embedding tensor (batch_size, frame_length, output_size).
        """
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout(x)
        x = self.decoder(x, mask)
        x = self.linear(x)  # (batch_size, frame_length, output_size)
        return x


class VarianceAdapter(nn.Module):
    """Variance adapter module.

    Based on T. Hayashi et al., "Non-autoregressive sequence-to-sequence voice conversion,"
    in ICASSP, 2021, pp. 7068-7072.

    """

    def __init__(self, input_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.duration_predictor = Predictor(input_size, kernel_size, dropout_rate)
        self.pitch_predictor = Predictor(input_size, kernel_size, dropout_rate)
        self.energy_predictor = Predictor(input_size, kernel_size, dropout_rate)
        self.pitch_embedding = nn.Conv1d(1, input_size, kernel_size=1, stride=1)
        self.energy_embedding = nn.Conv1d(1, input_size, kernel_size=1, stride=1)
        self.length_regulator = GaussianResampling()

    def forward(
        self, x: torch.Tensor, duration: torch.Tensor, pitch: torch.Tensor, energy: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Token embedding tensor (batch_size, sequence_length, input_size).
            duration (torch.Tensor): Duration tensor (batch_size, sequence_length).
            pitch (torch.Tensor): Pitch tensor (batch_size, frame_length).
            energy (torch.Tensor): Energy tensor (batch_size, frame_length).
            mask (torch.Tensor): Token embedding mask tensor (batch_size, sequence_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.Tensor: Token embedding tensor (batch_size, frame_length, input_size).
                torch.Tensor: Predicted duration tensor in logarithmic domain (batch_size, sequence_length).
                torch.Tensor: Predicted pitch tensor (batch_size, frame_length).
                torch.Tensor: Energy tensor (batch_size, frame_length).
        """
        d_hat = self.duration_predictor(x)  # (batch_size, sequence_length)
        # NOTE: use ground truth duration for training
        x = self.length_regulator(x=x, d=duration, mask=mask)  # (batch_size, frame_length, input_size)
        p_hat = self.pitch_predictor(x)  # (batch_size, frame_length)
        e_hat = self.energy_predictor(x)  # (batch_size, frame_length)
        # NOTE: use ground truth pitch and energy for training
        p = self.pitch_embedding(pitch[:, None, :]).transpose(1, 2)  # (batch_size, frame_length, input_size)
        e = self.energy_embedding(energy[:, None, :]).transpose(1, 2)  # (batch_size, frame_length, input_size)
        x = x + p + e
        return x, d_hat, p_hat, e_hat

    @torch.no_grad()
    def inference(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Token embedding tensor (batch, sequence, d_model).
            mask (torch.Tensor): Token mask tensor (batch, sequence).

        Returns:
            torch.Tensor: Token embedding tensor (batch, frame, d_model).
        """
        d_hat = self.duration_predictor(x)  # (batch, sequence)
        # NOTE: use predicted duration for inference
        # NOTE: duration is logarithm domain
        d_hat = torch.exp(d_hat)
        x = self.length_regulator(x=x, d=d_hat, mask=mask)  # (batch, frame, d_model)
        p_hat = self.pitch_predictor(x)  # (batch, frame)
        e_hat = self.energy_predictor(x)  # (batch, frame)
        # NOTE: use predicted pitch and energy for inference
        p = self.pitch_embedding(p_hat[:, None, :]).transpose(1, 2)  # (batch, frame, d_model)
        e = self.energy_embedding(e_hat[:, None, :]).transpose(1, 2)  # (batch, frame, d_model)
        x = x + p + e
        return x
