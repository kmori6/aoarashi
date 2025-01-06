import numpy as np

from aoarashi.modules.glow_tts import monotonic_alignment_search


def test_monotonic_alignment_search():
    value = np.array([[-0.1, -0.2, -0.3, -0.4], [-0.2, -0.3, -0.4, -0.5], [-0.3, -0.4, -0.5, -0.6]])
    expected_output = np.array([0, 0, 1, 2])
    output = monotonic_alignment_search(value, token_length=3, feat_length=4)
    np.testing.assert_array_equal(output, expected_output)
