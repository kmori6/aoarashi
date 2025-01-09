from dataclasses import dataclass

import torch
import torch.nn as nn
from torchaudio.transforms import RNNTLoss

from aoarashi.modules.conformer import Conformer
from aoarashi.modules.log_mel_spectrogram import LogMelSpectrogram
from aoarashi.modules.spec_augment import SpecAugment
from aoarashi.modules.transducer import Joiner, Predictor


@dataclass
class Sequence:
    token: list[int]
    state: tuple[torch.Tensor, torch.Tensor]
    total_score: float
    score_history: list[float]


class Model(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        vocab_size: int,
        n_mels: int,
        num_time_masks: int,
        num_freq_masks: int,
        time_mask_ratio: float,
        max_freq_mask_size: int,
        input_size: int,
        d_model: int,
        num_heads: int,
        kernel_size: int,
        num_blocks: int,
        hidden_size: int,
        num_layers: int,
        encoder_size: int,
        predictor_size: int,
        joiner_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.blank_token_id = vocab_size - 1
        self.frontend = LogMelSpectrogram(
            fft_size=n_fft, hop_size=hop_length, window_size=win_length, mel_size=n_mels, sample_rate=16000
        )
        self.specaug = SpecAugment(
            num_time_masks=num_time_masks,
            num_freq_masks=num_freq_masks,
            time_mask_ratio=time_mask_ratio,
            max_freq_mask_size=max_freq_mask_size,
        )
        self.encoder = Conformer(
            input_size=input_size,
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )
        self.predictor = Predictor(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            blank_token_id=self.blank_token_id,
        )
        self.joiner = Joiner(
            vocab_size=vocab_size,
            encoder_size=encoder_size,
            predictor_size=predictor_size,
            joiner_size=joiner_size,
            dropout_rate=dropout_rate,
        )
        self.loss_fn = RNNTLoss(blank=self.blank_token_id, reduction="mean", fused_log_softmax=False)

    def forward(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        token: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            audio (torch.Tensor): Sqeech (batch, sample).
            audio_length (torch.Tensor): Speech length (batch,).
            token (torch.Tensor): Predictor input token (batch, length + 1).
            target (torch.Tensor): Target token (batch, length).
            target_length (torch.Tensor): Target token length (batch,).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Loss.
                dict[str, torch.Tensor]: Statistics.
        """
        b = audio.shape[0]
        x_enc, mask = self.frontend(audio, audio_length)
        if self.training:
            x_enc = self.specaug(x_enc)
        x_enc, mask = self.encoder(x_enc, mask)  # (batch, frame, encoder_size)
        x_dec, _ = self.predictor(token, self.predictor.init_state(b, x_enc.device))  # (batch, time, predictor_size)
        x_rnnt = self.joiner(x_enc[:, :, None, :], x_dec[:, None, :, :])  # (batch, frame, time, vocab_size)
        #  loss
        frame_length = mask.sum(-1)
        loss = self.loss_fn(torch.log_softmax(x_rnnt, dim=-1), target.int(), frame_length.int(), target_length.int())
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def recognize(self, audio: torch.Tensor, beam_size: int) -> Sequence:
        """

        Args:
            audio (torch.Tensor): Speech waveform (sample,).
            beam_size (int): Beam size.

        Returns:
            Sequence: Decoded sequence with highest log probability.
        """
        audio_length = torch.tensor([audio.shape[0]], dtype=torch.long, device=audio.device)
        x, mask = self.frontend(audio[None, :], audio_length)
        x, _ = self.encoder(x, mask)
        hyp = self._beam_search(x, beam_size)
        return hyp

    @torch.no_grad()
    def _beam_search(self, x: torch.Tensor, beam_width: int) -> Sequence:
        """Beam search algorithm for RNN-T model.

        Proposed in A. Graves, "Sequence Transduction with Recurrent Neural Networks,"
        arXiv preprint arXiv:1211.3711, 2012.

        Args:
            x (torch.Tensor): Encoder output sequence of shape (batch, frame, encoder_size).

        Returns:
            Sequence: Decoded sequence with highest log probability.
        """
        # Initalize: B = {\varnothing}; Pr(\varnothing) = 1
        # NOTE: use log probability instead of probability for easier computation
        B = [
            Sequence(
                token=[self.blank_token_id],
                total_score=0.0,
                score_history=[0.0],
                state=self.predictor.init_state(1, x.device),
            )
        ]
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
            while (
                len([_y for _y in B if _y.total_score > max(A, key=lambda x: x.total_score).total_score]) <= beam_width
            ):
                # y^∗ = most probable in A
                y_star = max(A, key=lambda x: x.total_score)
                # Remove y^∗ from A
                A = [_y for _y in A if _y.total_score != y_star.total_score]
                # Pr(y^∗) = Pr(y^∗) Pr(\varnothing|y, t)
                # WARNING: use y^* instead of y because y is not defined
                z, state = self.predictor(
                    torch.tensor([y_star.token[-1:]], dtype=torch.long, device=x.device), y_star.state
                )
                scores = torch.log_softmax(self.joiner(x[:, t : t + 1, None, :], z[:, None, :, :]), dim=-1).squeeze()
                y_star.total_score += scores[-1].item()
                # Add y^∗ to B
                B.append(y_star)
                # NOTE: limit the number of k \in Y to the beam width
                for score, k in zip(*torch.topk(scores[:-1], beam_width)):
                    # Pr(y^∗ + k) = Pr(y^∗) Pr(k|y^∗, t)
                    A.append(
                        Sequence(
                            token=y_star.token + [k.item()],
                            total_score=y_star.total_score + score.item(),
                            score_history=y_star.score_history + [score.item()],
                            state=state,
                        )
                    )
            # Remove all but the W most probable from B
            B = sorted(B, key=lambda x: x.total_score, reverse=True)[:beam_width]
        # Return: y with highest log Pr(y)/|y| in B
        return max(B, key=lambda x: x.total_score / len(x.token))
