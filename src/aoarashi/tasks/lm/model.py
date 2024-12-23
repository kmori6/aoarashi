import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        dropout_rate: float,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        label_smoothing: float,
        ignore_token_id: int = -100,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.ignore_token_id = ignore_token_id
        self.input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.layers = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # share the same weight matrix
        self.linear.weight = self.input_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, token: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x = self.input_embedding(token)  # (batch, time, d_model)
        x, _ = self.layers(x)
        x = self.linear(x)  # (batch, time, vocab_size)
        # loss
        loss = self.loss_fn(x.flatten(0, 1), target.flatten())
        mask_valid = target != self.ignore_token_id
        acc = (x.argmax(-1)[mask_valid] == target[mask_valid]).sum() / mask_valid.sum()
        ppl = torch.exp(loss)
        return loss, {"loss": loss.item(), "acc": acc.item(), "ppl": ppl.item()}
