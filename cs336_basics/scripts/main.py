import time
import logging
import sys
import random


import hydra
import torch
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from cs336_basics.train import hydra_train, lr_range_test, train_overfit_one_batch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Full Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        set_seed(cfg.seed)
    
    if cfg.get("wandb"):
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name, 
            tags=cfg.wandb.get("tags", []),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    log.info(f"Instantiating model <{cfg.model._target_}>")

    model = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")

    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.parameters(),
    )

    log.info(f"Start training")

    hydra_train(
        model=model,
        optimizer=optimizer,

        data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        batch_size=cfg.data.batch_size,
        context_length=cfg.data.context_length,

        steps=cfg.training.steps,
        warmup_ratio=cfg.training.warmup_ratio,
        val_steps=cfg.training.val_steps,
        eval_interval=cfg.training.eval_interval,
        save_path=cfg.training.save_path,
        max_norm=cfg.training.max_norm,

        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    wandb.finish()


if __name__ == "__main__":
    main()