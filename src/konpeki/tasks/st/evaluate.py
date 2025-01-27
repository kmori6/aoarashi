from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from konpeki.modules.rnn_transducer import Sequence
from konpeki.tasks.st.dataset import Dataset
from konpeki.tasks.st.model import Model
from konpeki.utils.tokenizer import Tokenizer

logger = getLogger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
            # NOTE: streaming fashion
            audio = sample["audio"].to(device)  # (sample_length,)
            audio_chunks = audio.split(config.evaluate.chunk_sample)
            initial_set: list[Sequence] = []
            audio = torch.zeros(config.evaluate.history_sample, device=device)
            for j, audio_chunk in tqdm(enumerate(audio_chunks), total=len(audio_chunks), leave=False):
                if len(audio_chunk) < config.evaluate.chunk_sample:
                    audio_chunk = torch.cat([audio_chunk, torch.zeros(config.evaluate.chunk_sample - len(audio_chunk))])
                audio = torch.cat([audio, audio_chunk])
                if len(audio) > config.evaluate.chunk_sample + config.evaluate.history_sample:
                    audio = audio[config.evaluate.chunk_sample :]
                assert len(audio) == config.evaluate.chunk_sample + config.evaluate.history_sample
                initial_set = model.translate(
                    audio,
                    beam_size=config.evaluate.beam_size,
                    chunk_size=config.evaluate.chunk_size,
                    history_size=config.evaluate.history_size,
                    initial_set=initial_set,
                    end_chunk=j == len(audio_chunks) - 1,
                )
            text = tokenizer.decode(initial_set[0].token[1:])
            hyp_list.append(text)
            f_hyp.write(text + "\n")
    bleu = BLEU()
    metric = bleu.corpus_score(hyp_list, [ref_list]).score
    logger.info(f"bleu: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
