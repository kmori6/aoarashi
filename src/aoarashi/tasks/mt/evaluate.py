from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from aoarashi.tasks.mt.dataset import Dataset
from aoarashi.tasks.mt.model import Model
from aoarashi.utils.tokenizer import Tokenizer

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
            ref_list.append(sample["tgt_text"])
            f_ref.write(sample["tgt_text"] + "\n")
            enc_token = torch.tensor(
                [tokenizer.encode(sample["src_text"])], dtype=torch.long, device=device
            )  # (1, time1)
            hyp = model.translate(
                enc_token, config.evaluate.beam_size, config.evaluate.length_buffer, config.evaluate.length_penalty
            )
            hyp_text = tokenizer.decode(hyp.token[1:-1])
            hyp_list.append(hyp_text)
            f_hyp.write(hyp_text + "\n")
    bleu = BLEU()
    metric = bleu.corpus_score(hyp_list, [ref_list]).score
    logger.info(f"bleu: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
