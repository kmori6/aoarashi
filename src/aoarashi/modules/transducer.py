import torch
import torch.nn as nn


class Predictor(nn.Module):
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
            token (torch.Tensor): Input tensor (batch, seq).

        Returns:
            torch.Tensor: Output tensor (batch, seq, hidden_size).
        """
        x = self.embed(token)  # (batch, seq, hidden_size)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, state)
        return x, (h, c)


class Joiner(nn.Module):
    def __init__(self, vocab_size: int, encoder_size: int, predictor_size: int, joiner_size: int, dropout_rate: float):
        super().__init__()
        self.linear_enc = nn.Linear(encoder_size, joiner_size)
        self.linear_pred = nn.Linear(predictor_size, joiner_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_out = nn.Linear(joiner_size, vocab_size)

    def forward(self, x_enc: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder output sequence (batch, frame, 1, encoder_size).
            x_pred (torch.Tensor): Predictor output sequence (batch, 1, seq, predictor_size).

        Returns:
            torch.Tensor: Output sequence (batch, frame, seq, vocab_size).
        """
        x = self.linear_enc(x_enc) + self.linear_pred(x_pred)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x
