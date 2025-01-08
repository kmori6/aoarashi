import torch

from aoarashi.modules.energy import Energy


def test_energy():
    length = torch.tensor([16000, 8000])
    speech = torch.randn(2, 16000)
    module = Energy(fft_size=512, hop_size=128, window_size=512)
    energy, mask = module(speech, length)
    max_frame_length = 16000 // module.hop_size + 1
    # shape
    assert energy.shape == (2, max_frame_length)
    assert mask.shape == (2, max_frame_length)
    # mask
    assert torch.all(mask[0, :])
    assert torch.all(mask[1, : 8000 // module.hop_size + 1]) and torch.all(~mask[1, 8000 // module.hop_size + 1 :])
    assert torch.all(energy.masked_select(~mask) == 0.0)
