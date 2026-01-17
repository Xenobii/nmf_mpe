import logging
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from hydra.utils import instantiate

from sklearn.model_selection import train_test_split
import numpy as np


log = logging.getLogger(__name__)


def configure_torch(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32


class Trainer:
    def __init__(
        self,
        device: str,
        epochs: int,
        
        model: nn.Module,
        frontend: nn.Module,
        
        data_train: DataLoader,
        data_valid: DataLoader,
        
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.modules.loss._Loss,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        log.info(f"Training settings:")
        log.info(f" Epochs: {self.epochs}")
        log.info(f" Device: {self.device}")

        self.model      = model.to(self.device)
        self.frontend   = frontend.to(self.device)

        self.data_valid = data_valid
        self.data_train = data_train
        
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.criterion = criterion

    def train(self):
        log.info("-- Training --")
        
        for epoch in range(1, self.epochs+1):
            self.model.train(True)
            
            epoch_train_loss = 0.0
            epoch_valid_loss = 0.0

            for batch in tqdm(self.data_train, total=len(self.data_train), desc="Training..."
            ):
                train_loss = self.step_train(batch)
                epoch_train_loss += train_loss
            
            epoch_train_loss /= len(self.data_train)

            for batch in tqdm(self.data_valid, total=len(self.data_valid), desc="Validating..."
            ):
                valid_loss = self.step_valid(batch)
                epoch_valid_loss += valid_loss
            
            epoch_valid_loss /= len(self.data_valid)

            log.info(
                f"Epoch [{epoch}/{self.epochs}]"
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Valid Loss: {epoch_valid_loss:.4f}"
            )
                

    def step_train(self, batch):
        wave = batch["wave"].to(self.device, non_blocking=True)
        roll = batch["roll"].to(self.device, non_blocking=True)

        # Forward
        x = self.frontend(wave)
        y = self.model(roll)

        # Loss
        loss = self.criterion(y, x)

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def step_valid(self, batch):
        with torch.no_grad():
            wave = batch["wave"].to(self.device, non_blocking=True)
            roll = batch["roll"].to(self.device, non_blocking=True)
    
            # Forward
            x = self.frontend(wave)
            y = self.model(roll)
    
            # Loss
            loss = self.criterion(y, x)

        return loss.item()



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def train(cfg: DictConfig):
    log.info("-- Initialization --")
    # --- torch ---
    configure_torch(cfg.torch)

    # --- dataset + split + dataloader ---
    dataset = instantiate(cfg.dataset)
    
    files = np.arange(len(dataset))
    train_indices, valid_indices = train_test_split(
        files,
        test_size=cfg.train.valid_size,
        random_state=cfg.torch.seed,
        shuffle=cfg.train.shuffle
    )
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    dataloader_train = instantiate(cfg.dataloader, dataset=train_dataset)
    dataloader_valid = instantiate(cfg.dataloader, dataset=valid_dataset)
    
    # --- audio frontend ---
    frontend = instantiate(cfg.audio_frontend)

    # --- model ---
    W = instantiate(cfg.w_init)
    model = instantiate(cfg.model, W=W)
    
    # --- optimizer ---
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    criterion = instantiate(cfg.loss)

    # --- trainer ---
    trainer = instantiate(
        cfg.trainer,
        model=model, frontend=frontend,
        data_train=dataloader_train, data_valid=dataloader_valid,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
    )
    trainer.train()



if __name__ == "__main__":
    train()