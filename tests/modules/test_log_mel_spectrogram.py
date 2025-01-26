import torch

from konpeki.modules.log_mel_spectrogram import LogMelSpectrogram


def test_log_mel_spectrogram():
    length = torch.tensor([16000, 8000])
    speech = torch.randn(2, 16000)
    module = LogMelSpectrogram(fft_size=512, hop_size=128, window_size=512, mel_size=80, sample_rate=16000)
    mel, mask = module(speech, length)
    max_frame_length = 16000 // module.hop_size + 1
    # shape
    assert mel.shape == (2, max_frame_length, 80)
    assert mask.shape == (2, max_frame_length)
    # mask
    assert torch.all(mask[0, :])
    assert torch.all(mask[1, : 8000 // module.hop_size + 1]) and torch.all(~mask[1, 8000 // module.hop_size + 1 :])
    assert torch.all(mel.masked_select(~mask[:, :, None]) == 0.0)
