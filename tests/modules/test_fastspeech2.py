import torch

from konpeki.modules.fastspeech2 import extract_segment


def test_extract_segment():
    batch_size = 4
    hidden_size = 8
    frame_size = 16
    segment_size = 4
    x = torch.randn(batch_size, hidden_size, frame_size)
    start_frame = torch.tensor([0, 1, 2, 3])
    segment = extract_segment(x, start_frame, segment_size)
    assert segment.shape == (batch_size, hidden_size, segment_size)
    assert torch.equal(segment[0, :, :], x[0, :, 0:segment_size])
    assert torch.equal(segment[1, :, :], x[1, :, 1 : segment_size + 1])
    assert torch.equal(segment[2, :, :], x[2, :, 2 : segment_size + 2])
    assert torch.equal(segment[3, :, :], x[3, :, 3 : segment_size + 3])
