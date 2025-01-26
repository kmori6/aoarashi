from logging import getLogger

import hydra
import numpy as np
import torch
from fastdtw import fastdtw
from omegaconf import DictConfig
from pysptk import mcep
from pysptk.sptk import hamming
from pysptk.util import mcepalpha
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from konpeki.tasks.tts.dataset import Dataset
from konpeki.tasks.tts.model import Model
from konpeki.utils.tokenizer import PhonemeTokenizer, normalize_text

logger = getLogger(__name__)


def calculate_mel_cepstral_distortion(
    x: np.ndarray,
    y: np.ndarray,
    fft_size: int,
    hop_size: int,
    sample_rate: int = 22050,
    mfc_size: int = 34,
    eps: float = 1e-12,
) -> float:
    """

    Based on T. Hayashi et al., "Non-Autoregressive sequence-to-sequence voice conversion,"
    in ICASSP, 2021, pp. 7068-7072.

    Args:
        x (np.ndarray): Source waveform of shape (x_sample_length,)
        y (np.ndarray): Target waveform of shape (y_sample_length,)
        fft_size (int): FFT size
        hop_size (int): Hop size
        sample_rate (int): Sampling rate
        mfc_size (int): Mel-cepstral coefficient size
        eps (float): Epsilon

    Returns:
        float: Mel-cepstral distortion
    """
    x_frame_length = 1 + (len(x) - fft_size) // hop_size
    y_frame_length = 1 + (len(y) - fft_size) // hop_size
    x_mceps = [
        mcep(
            x[i * hop_size : i * hop_size + fft_size] * hamming(fft_size),
            order=mfc_size,
            alpha=mcepalpha(sample_rate),
            etype=1,
            eps=eps,
        )  # (order + 1,)
        for i in range(x_frame_length)
    ]
    y_mceps = [
        mcep(
            y[i * hop_size : i * hop_size + fft_size] * hamming(fft_size),
            order=mfc_size,
            alpha=mcepalpha(sample_rate),
            etype=1,
            eps=eps,
        )  # (order + 1,)
        for i in range(y_frame_length)
    ]
    x = np.stack(x_mceps, axis=0)  # (x_frame_length, order + 1)
    y = np.stack(y_mceps, axis=0)  # (y_frame_length, order + 1)
    _, path = fastdtw(x, y, dist=euclidean)
    x_t, y_t = np.array(path).T  # (max(x_frame_length, y_frame_length),)
    mcd = 10.0 * np.sqrt(2) / np.log(10.0) * np.sqrt(np.sum(2 * (x[x_t] - y[y_t]) ** 2, axis=-1)).mean()
    return mcd


@hydra.main(version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(config.dataset.test_json_path)
    tokenizer = PhonemeTokenizer(config.tokenizer.model_path)
    state_dict = torch.load(config.evaluate.model_path, map_location=device)["model_state_dict"]
    model = Model(**config.model).to(device).eval()
    model.load_state_dict(state_dict)
    mcds = []
    for i in tqdm(range(len(test_dataset))):
        sample = test_dataset[i]
        source_wav = sample["audio"].cpu().numpy()
        text = normalize_text(sample["text"])
        token = tokenizer.encode(text)
        tgt_wav = model.synthesize(torch.tensor(token, dtype=torch.long, device=device)).squeeze().cpu().numpy()
        mcd = calculate_mel_cepstral_distortion(
            source_wav, tgt_wav, config.model.fft_size, config.model.hop_size, config.model.sample_rate
        )
        mcds.append(mcd)
    metric = np.mean(mcds)
    logger.info(f"mcd: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"mcd: {metric:.5f}")


if __name__ == "__main__":
    main()
