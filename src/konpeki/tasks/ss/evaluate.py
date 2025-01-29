from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from konpeki.tasks.ss.collate_fn import CollateFn
from konpeki.tasks.ss.dataset import Dataset
from konpeki.tasks.ss.model import Model

logger = getLogger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(config.dataset.test_json_path)
    state_dict = torch.load(config.evaluate.model_path, map_location=device)
    model = Model(**config.model).to(device).eval()
    model.load_state_dict(state_dict)
    collate_fn = CollateFn(config.model.autoencoder_stride)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1, shuffle=False, drop_last=False)
    total_loss = 0
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        _, stats = model(**{k: v.to(device) for k, v in batch.items()})
        total_loss += -stats["loss"]
    metric = total_loss / len(test_dataloader)
    logger.info(f"si-snr: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"si-snr: {metric:.5f}")


if __name__ == "__main__":
    main()
