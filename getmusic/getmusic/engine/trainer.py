import os
import time
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from ..modeling.base import BaseDiffusionModel
from ..modeling.config import ModelConfig, TrainingConfig
from .ema import EMA

class TrainingManager:
    """Handles model training, validation, and checkpointing"""
    
    def __init__(self, 
                 model: BaseDiffusionModel,
                 config: ModelConfig,
                 save_dir: str,
                 distributed: bool = False,
                 local_rank: int = 0,
                 world_size: int = 1):
        
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Setup directories
        self.ckpt_dir = self.save_dir / 'checkpoint'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.last_epoch = -1
        self.last_iter = -1
        self.start_train_time = None
        
        # Initialize mixed precision if available
        self.amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.amp else None
        
        # Setup EMA if enabled
        if config.diffusion.use_ema and local_rank == 0:
            self.ema = EMA(
                model=model,
                decay=config.diffusion.ema_decay,
                update_interval=config.diffusion.ema_update_interval,
                device=model.device
            )
        else:
            self.ema = None
            
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_optimizer_and_scheduler(self) -> None:
        """Initialize optimizer and learning rate scheduler"""
        # Calculate learning rate based on world size
        base_lr = self.config.training.base_lr
        if self.config.training.adjust_lr == 'sqrt':
            lr = base_lr * (self.world_size * self.config.training.batch_size) ** 0.5
        elif self.config.training.adjust_lr == 'linear':
            lr = base_lr * self.world_size * self.config.training.batch_size
        else:
            lr = base_lr
            
        self.logger.info(f'Using learning rate {lr} (base={base_lr})')
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=lr * 0.01
        )
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Train for one epoch"""
        self.model.train()
        self.last_epoch += 1
        
        if self.distributed:
            train_loader.sampler.set_epoch(self.last_epoch)
            
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.amp):
                loss_dict = self.model(**batch)
                loss = loss_dict['loss']
                
            # Backward pass
            self.optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                if self.config.training.clip_grad_norm_max > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), 
                                  self.config.training.clip_grad_norm_max)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.clip_grad_norm_max > 0:
                    clip_grad_norm_(self.model.parameters(),
                                  self.config.training.clip_grad_norm_max)
                self.optimizer.step()
                
            # Update EMA model
            if self.ema is not None:
                self.ema.update(self.last_iter)
                
            self.last_iter += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch: {self.last_epoch}/{self.config.training.max_epochs} '
                    f'[{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )
                
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                with autocast(enabled=self.amp):
                    loss_dict = self.model(**batch)
                    val_loss += loss_dict['loss'].item()
                    
        val_loss /= len(val_loader)
        return {'val_loss': val_loss}
        
    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save training checkpoint"""
        if self.local_rank != 0:
            return
            
        state = {
            'last_epoch': self.last_epoch,
            'last_iter': self.last_iter,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics or {},
            'config': self.config.to_dict()
        }
        
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
            
        # Save checkpoint
        ckpt_path = self.ckpt_dir / f'checkpoint_{self.last_epoch}.pt'
        torch.save(state, ckpt_path)
        
        # Save latest checkpoint
        latest_path = self.ckpt_dir / 'latest.pt'
        torch.save(state, latest_path)
        
        self.logger.info(f'Saved checkpoint: {ckpt_path}')
        
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
            
        state = torch.load(path, map_location=self.model.device)
        
        # Load model weights
        self.model.load_state_dict(state['model'])
        
        # Load EMA if available
        if self.ema is not None and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        
        # Restore training state
        self.last_epoch = state['last_epoch']
        self.last_iter = state['last_iter']
        
        self.logger.info(f'Loaded checkpoint from {path}')
        
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None) -> None:
        """Main training loop"""
        self.start_train_time = time.time()
        
        for epoch in range(self.last_epoch + 1, self.config.training.max_epochs):
            # Train for one epoch
            self.train_epoch(train_loader)
            
            # Run validation if available
            metrics = None
            if val_loader is not None and (epoch + 1) % self.config.training.validation_epochs == 0:
                metrics = self.validate(val_loader)
                
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_epochs == 0:
                self.save_checkpoint(metrics)
                
            # Update learning rate
            self.scheduler.step()
            
        # Save final checkpoint
        self.save_checkpoint(metrics)