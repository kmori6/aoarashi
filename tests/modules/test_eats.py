import torch

from konpeki.modules.eats import GaussianResampling


def test_gaussian_resampling():
    x = torch.randn(2, 3, 2)
    d = torch.tensor([[1, 2, 3], [2, 3, 4]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    resampling = GaussianResampling(variance=1.0)
    y = resampling(x, d, mask)
    assert y.shape == (2, 9, 2)
