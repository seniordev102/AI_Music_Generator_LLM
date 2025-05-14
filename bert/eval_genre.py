#!/usr/bin/env python3
"""Evaluate MusicBERT on genre classification."""

import os
import argparse
import logging
from pathlib import Path

import torch
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

from musicbert import (
    MusicBERTConfig,
    MusicBERTForGenreClassification,
    load_and_cache_examples,
    GENRE_CLASSIFICATION_CONFIG
)

logger = logging.getLogger(__name__)

def evaluate(args):
    """Evaluate model on genre classification task."""
    # Load model configuration
    config = GENRE_CLASSIFICATION_CONFIG
    if args.model_size:
        config.model_size = args.model_size
        
    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = MusicBERTForGenreClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load evaluation data
    eval_dataset = load_and_cache_examples(
        args.data_dir,
        config,
        evaluate=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    logger.info("***** Running evaluation *****")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions
            outputs = model(**batch)
            logits = outputs.logits
            
            # Convert logits to predictions
            if config.num_labels == 1:
                preds = torch.sigmoid(logits) > 0.5
            else:
                preds = torch.argmax(logits, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
    # Generate classification report
    target_names = eval_dataset.get_genre_labels()
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    
    # Save results
    results = {
        "predictions": all_preds,
        "labels": all_labels,
        "report": report
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions and labels
    pd.DataFrame({
        "prediction": all_preds,
        "label": all_labels
    }).to_csv(output_dir / "predictions.csv", index=False)
    
    # Save classification report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
        
    logger.info("***** Evaluation Results *****")
    logger.info("\n" + report)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory containing evaluation files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["base", "large", "medium", "small", "mini", "tiny"],
        help="Model architecture size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    evaluate(args)

if __name__ == "__main__":
    main()
