"""RNN Transducer module.

Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks," in ICASSP, 2013, pp. 6645-6649.

"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Sequence:
    """Sequence dataclass for output sequence beam search.

    Args:
        token (list[int]): Token list corresponding to y vector in the paper.
        hidden_state (torch.Tensor): Hidden state tensor for the prediction network
            (num_layers, batch_size, hidden_size).
        cell_state (torch.Tensor): Cell state tensor for the prediction network
            (num_layers, batch_size, hidden_size).
        total_score (float): Total probability in log scale for efficient computation
            corresponding to Pr(y) in the paper.
        score_history (list[float]): Score history for prefix offset calculation.
    """

    token: list[int]
    hidden_state: torch.Tensor
    cell_state: torch.Tensor
    total_score: float
    score_history: list[float]


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
        self, token: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            token (torch.Tensor): Token tensor (batch_size, sequence_length).
            hidden_state (torch.Tensor): Hidden state tensor (num_layers, batch_size, hidden_size).
            cell_state (torch.Tensor): Cell state tensor (num_layers, batch_size, hidden_size).

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                torch.Tensor: Hidden state tensor (batch_size, sequence_length, hidden_size).
                torch.Tensor: Hidden state tensor (num_layers, batch_size, hidden_size).
                torch.Tensor: Cell state tensor (num_layers, batch_size, hidden_size).
        """
        x = self.embed(token)  # (batch_size, sequence_length, hidden_size)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, (hidden_state, cell_state))
        return x, h, c


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


@torch.no_grad()
def beam_search(
    x: torch.Tensor,
    prediction_network: PredictionNetwork,
    joint_network: JointNetwork,
    beam_width: int,
    blank_token_id: int,
    initial_set: list[Sequence] = [],
    return_highest: bool = False,
) -> list[Sequence]:
    """Beam search algorithm for RNN-T model.

    Proposed in A. Graves, "Sequence transduction with recurrent neural networks,"
    arXiv preprint arXiv:1211.3711, 2012.

    Args:
        x (torch.Tensor): Encoder embedding tensor (batch_size, frame_length, encoder_size).
        beam_width (int): Beam width.
        blank_token_id (int): Blank token ID.
        return_highest (bool): Return the highest log probability sequence.

    Returns:
        Sequence: Decoded sequence with highest log probability.
    """
    if len(initial_set) == 0:
        # Initalize: B = {\varnothing}; Pr(\varnothing) = 1
        # NOTE: use log probability instead of probability for easier computation
        h, c = prediction_network.init_state(1, x.device)
        B = [
            Sequence(
                token=[blank_token_id],
                hidden_state=h,
                cell_state=c,
                total_score=0.0,
                score_history=[0.0],
            )
        ]
    else:
        B = initial_set
    for t in range(x.shape[1]):
        A = B
        B = []
        for y in A:
            # Pr(y) += \Sigma_{\hat{y} \in pref(y) \cap A} Pr(\hat{y}) Pr(y|\hat{y},t)
            ys_hat = [_y for _y in A if _y.token in [y.token[:i] for i in range(len(y.token))]]
            for y_hat in ys_hat:
                y.total_score = torch.logsumexp(
                    torch.tensor([y.total_score] + y.score_history[len(y_hat.token) :]), dim=0
                ).item()
        while len([_y for _y in B if _y.total_score > max(A, key=lambda x: x.total_score).total_score]) <= beam_width:
            # y^∗ = most probable in A
            y_star = max(A, key=lambda x: x.total_score)
            # Remove y^∗ from A
            A = [_y for _y in A if _y.total_score != y_star.total_score]
            # Pr(y^∗) = Pr(y^∗) Pr(\varnothing|y, t)
            # WARNING: use y^* instead of y because y is not defined
            z, h, c = prediction_network(
                token=torch.tensor([y_star.token[-1:]], dtype=torch.long, device=x.device),
                hidden_state=y_star.hidden_state,
                cell_state=y_star.cell_state,
            )
            scores = torch.log_softmax(joint_network(x[:, t : t + 1, None, :], z[:, None, :, :]), dim=-1).squeeze()
            y_star.total_score += scores[-1].item()
            # Add y^∗ to B
            B.append(y_star)
            # NOTE: limit the number of k \in Y to the beam width
            for score, k in zip(*torch.topk(scores[:-1], beam_width)):
                # Pr(y^∗ + k) = Pr(y^∗) Pr(k|y^∗, t)
                A.append(
                    Sequence(
                        token=y_star.token + [k.item()],
                        hidden_state=h,
                        cell_state=c,
                        total_score=y_star.total_score + score.item(),
                        score_history=y_star.score_history + [score.item()],
                    )
                )
        # Remove all but the W most probable from B
        B = sorted(B, key=lambda x: x.total_score, reverse=True)[:beam_width]
    if return_highest:
        # Return: y with highest log Pr(y)/|y| in B
        return [max(B, key=lambda x: x.total_score / len(x.token))]
    return B
