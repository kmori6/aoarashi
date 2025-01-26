"""RNN Transducer module.

Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks," in ICASSP, 2013, pp. 6645-6649.

"""

import torch
import torch.nn as nn


class PredictionNetwork(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, dropout_rate: float, blank_token_id: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=blank_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        return h, c

    def forward(
        self, token: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            token (torch.Tensor): Token tensor (batch_size, sequence_length).
            state (torch.Tensor): Hidden state (h, c) tensor (num_layers, batch_size, hidden_size).

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                torch.Tensor: Hidden state tensor (batch_size, sequence_length, hidden_size).
                tuple[torch.Tensor, torch.Tensor]: Hidden state (h, c) tensor (num_layers, batch_size, hidden_size).
        """
        x = self.embed(token)  # (batch_size, sequence_length, hidden_size)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, state)
        return x, (h, c)


class JointNetwork(nn.Module):
    def __init__(self, vocab_size: int, encoder_size: int, predictor_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.w_l = nn.Linear(encoder_size, hidden_size)
        self.w_p = nn.Linear(predictor_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_h = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_enc: torch.Tensor, x_prd: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder hidden sequence tensor (batch_size, frame_length, 1, encoder_size).
            x_prd (torch.Tensor): Predicton hidden sequence tensor (batch_size, 1, sequence_length, predictor_size).

        Returns:
            torch.Tensor: Logit tesnor (batch_size, frame_length, sequence_length, vocab_size).
        """
        x = self.w_l(x_enc) + self.w_p(x_prd)  # (batch_size, frame_length, sequence_length, hidden_size)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_h(x)  # (batch_size, frame_length, sequence_length, vocab_size)
        return x
