"""Training utilities and metrics for MusicBERT."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import metrics
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm

from .config import MusicBERTConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    auc_micro: float = 0.0
    auc_macro: float = 0.0
    
    def update(self, metrics_dict: Dict[str, float]) -> None:
        """Update metrics from dictionary."""
        for k, v in metrics_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

class MetricsCalculator:
    """Handles metric calculation for different tasks."""
    
    @staticmethod
    def calculate_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
        task: str = "classification"
    ) -> Dict[str, float]:
        """Calculate metrics based on predictions and labels.
        
        Args:
            logits: Model predictions (before sigmoid/softmax)
            labels: Ground truth labels
            loss: Loss tensor
            task: Task type (classification/multilabel)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics_dict = {"loss": loss.item()}
        
        if task == "classification":
            preds = torch.argmax(logits, dim=1)
            metrics_dict["accuracy"] = (
                (preds == labels).float().mean().item()
            )
            
        elif task == "multilabel":
            # Convert logits to predictions
            preds = torch.sigmoid(logits) > 0.5
            
            # Move tensors to CPU and convert to numpy
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            
            # Calculate metrics
            metrics_dict.update({
                "f1_micro": f1_score(
                    labels_np, preds_np, average="micro", zero_division=0
                ),
                "f1_macro": f1_score(
                    labels_np, preds_np, average="macro", zero_division=0
                )
            })
            
            try:
                metrics_dict.update({
                    "auc_micro": roc_auc_score(
                        labels_np, logits_np, average="micro"
                    ),
                    "auc_macro": roc_auc_score(
                        labels_np, logits_np, average="macro"
                    )
                })
            except ValueError:
                # ROC AUC fails if there's only one class
                metrics_dict.update({
                    "auc_micro": 0.0,
                    "auc_macro": 0.0
                })
                
        return metrics_dict

class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: MusicBERTConfig,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Setup metrics
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics = TrainingMetrics()
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            leave=False
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if self.config.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_norm
                )
                
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            # Calculate metrics
            metrics_dict = MetricsCalculator.calculate_metrics(
                outputs.logits,
                batch["labels"],
                loss,
                "multilabel" if self.config.num_classes > 2 else "classification"
            )
            
            # Update progress bar
            progress_bar.set_postfix(loss=metrics_dict["loss"])
            
            # Update metrics
            self.train_metrics.update(metrics_dict)
            
        return self.train_metrics
        
    @torch.no_grad()
    def evaluate(self) -> TrainingMetrics:
        """Evaluate the model."""
        if not self.val_dataloader:
            return TrainingMetrics()
            
        self.model.eval()
        self.val_metrics = TrainingMetrics()
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Calculate metrics
            metrics_dict = MetricsCalculator.calculate_metrics(
                outputs.logits,
                batch["labels"],
                outputs.loss,
                "multilabel" if self.config.num_classes > 2 else "classification"
            )
            
            # Update metrics
            self.val_metrics.update(metrics_dict)
            
        return self.val_metrics