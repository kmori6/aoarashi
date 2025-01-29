import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from konpeki.tasks.ss.collate_fn import CollateFn
from konpeki.tasks.ss.dataset import Dataset
from konpeki.tasks.ss.model import Model
from konpeki.utils.trainer import Trainer


@hydra.main(version_base=None)
def main(config: DictConfig):
    os.makedirs(config.train.out_dir, exist_ok=True)
    train_dataset = Dataset(config.dataset.train_json_path)
    valid_dataset = Dataset(config.dataset.valid_json_path)
    model = Model(**config.model)
    collate_fn = CollateFn(config.model.autoencoder_stride)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **config.dataloader.train)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, **config.dataloader.valid)
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.train)
    trainer.train()


if __name__ == "__main__":
    main()
