#!/usr/bin/env python3
"""
Unified training script for MusicBERT.
Supports masked language modeling, genre classification, and next sentence prediction.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import Dictionary

from musicbert.config import (
    MusicBERTConfig,
    MASKED_LM_CONFIG,
    GENRE_CLASSIFICATION_CONFIG,
    NEXT_SENTENCE_CONFIG
)
from musicbert.data_utils import load_and_cache_examples
from musicbert.training_utils import Trainer

logger = logging.getLogger(__name__)

def setup_training(args):
    """Setup training configuration and environment."""
    utils.import_user_module(args)
    
    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 8192
        
    return tasks.setup_task(args)

def load_pretrained_model(model_path: str, task):
    """Load a pretrained model checkpoint."""
    model = None
    if os.path.exists(model_path):
        model = checkpoint_utils.load_checkpoint_to_cpu(model_path)
        logger.info(f"Loaded pretrained model from {model_path}")
    else:
        logger.warning(f"No pretrained model found at {model_path}")
    return model

def get_training_config(task_name: str) -> MusicBERTConfig:
    """Get task-specific training configuration."""
    if task_name == "mlm":
        return MASKED_LM_CONFIG
    elif task_name == "genre":
        return GENRE_CLASSIFICATION_CONFIG
    elif task_name == "nsp":
        return NEXT_SENTENCE_CONFIG
    else:
        raise ValueError(f"Unknown task: {task_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mlm", "genre", "nsp"],
        help="Training task"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory containing preprocessed files"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base", "large", "medium", "small", "mini", "tiny"],
        help="Model architecture size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # Set random seed
    utils.set_torch_seed(args.seed)
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = get_training_config(args.task)
    if args.model_size:
        config.model_size = args.model_size
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.peak_learning_rate = args.learning_rate
        
    # Setup task and load dictionary
    task = setup_training(args)
    dictionary = Dictionary.load(os.path.join(args.data_dir, "dict.txt"))
    
    # Load datasets
    train_dataset = load_and_cache_examples(
        args.data_dir,
        dictionary,
        config,
        evaluate=False
    )
    val_dataset = load_and_cache_examples(
        args.data_dir,
        dictionary,
        config,
        evaluate=True
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Build model
    model = task.build_model(args)
    
    # Load pretrained weights if specified
    if args.model_path:
        state_dict = load_pretrained_model(args.model_path, task)
        if state_dict is not None:
            model.load_state_dict(state_dict["model"], strict=False)
            
    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.peak_learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Task = {args.task}")
    logger.info(f"  Model size = {config.model_size}")
    logger.info(f"  Batch size = {config.batch_size}")
    logger.info(f"  Learning rate = {config.peak_learning_rate}")
    logger.info(f"  Max epochs = {args.max_epochs}")
    
    best_val_metric = float("inf")
    
    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch()
        logger.info(f"Training loss: {train_metrics.loss:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate()
        logger.info(f"Validation loss: {val_metrics.loss:.4f}")
        
        # Save checkpoint if validation metric improved
        if val_metrics.loss < best_val_metric:
            best_val_metric = val_metrics.loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "val_metrics": val_metrics.__dict__
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "best_model.pt")
            )
            logger.info("Saved new best model checkpoint")
            
    logger.info("***** Training finished *****")

if __name__ == "__main__":
    main()