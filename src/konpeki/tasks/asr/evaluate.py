from logging import getLogger

import hydra
import torch
from jiwer import process_words
from omegaconf import DictConfig
from tqdm import tqdm

from konpeki.tasks.asr.dataset import Dataset
from konpeki.tasks.asr.model import Model
from konpeki.utils.tokenizer import Tokenizer

logger = getLogger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(config.dataset.test_json_path)
    tokenizer = Tokenizer(config.tokenizer.model_path)
    state_dict = torch.load(config.evaluate.model_path, map_location=device)
    model = Model(**config.model).to(device).eval()
    model.load_state_dict(state_dict)
    hyp_list, ref_list = [], []
    with open(f"{config.evaluate.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref, open(
        f"{config.evaluate.out_dir}/hyp.txt", "w", encoding="utf-8"
    ) as f_hyp:
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            ref_list.append(sample["text"])
            f_ref.write(sample["text"] + "\n")
            hyp = model.recognize(sample["audio"].to(device), config.evaluate.beam_size)
            text = tokenizer.decode(hyp.token[1:])  # remove the first blank token
            hyp_list.append(text)
            f_hyp.write(text + "\n")
    error = total = 0
    for ref_text, hyp_text in zip(ref_list, hyp_list):
        output = process_words(ref_text, hyp_text)
        error += output.substitutions + output.deletions + output.insertions
        total += output.substitutions + output.deletions + output.hits
    metric = error / total
    logger.info(f"wer: {metric:.5f}")
    with open(f"{config.decode.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
