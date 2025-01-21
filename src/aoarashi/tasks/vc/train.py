import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from aoarashi.tasks.vc.collate_fn import CollateFn
from aoarashi.tasks.vc.dataset import Dataset
from aoarashi.tasks.vc.model import Model
from aoarashi.utils.trainer import AdversarialTrainer


@hydra.main(version_base=None)
def main(config: DictConfig):
    os.makedirs(config.train.out_dir, exist_ok=True)
    train_dataset = Dataset(config.dataset.train_json_path)
    valid_dataset = Dataset(config.dataset.valid_json_path)
    model = Model(**config.model)
    collate_fn = CollateFn()
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **config.dataloader.train)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, **config.dataloader.valid)
    trainer = AdversarialTrainer(model, train_dataloader, valid_dataloader, config.train)
    trainer.train()


if __name__ == "__main__":
    main()
