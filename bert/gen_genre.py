#!/usr/bin/env python3
"""Generate music with genre conditioning using MusicBERT."""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from tqdm import tqdm

from musicbert import (
    MusicBERTConfig,
    MusicBERTForMaskedLM,
    MusicTokenizer,
    MASKED_LM_CONFIG
)
from musicbert.data_utils import MIDIEvent

logger = logging.getLogger(__name__)

def generate_sequence(
    model: MusicBERTForMaskedLM,
    tokenizer: MusicTokenizer,
    genre_tokens: List[int],
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: str = "cuda"
) -> List[List[MIDIEvent]]:
    """Generate a musical sequence conditioned on genre.
    
    Args:
        model: The MusicBERT model
        tokenizer: Music tokenizer
        genre_tokens: Genre conditioning tokens
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of sequences to generate
        device: Device to run generation on
        
    Returns:
        List of generated sequences as MIDI events
    """
    model.eval()
    
    # Start with genre tokens
    input_ids = torch.tensor(genre_tokens).unsqueeze(0)
    input_ids = input_ids.repeat(num_return_sequences, 1)
    input_ids = input_ids.to(device)
    
    generated_sequences = []
    
    with torch.no_grad():
        for _ in tqdm(range(max_length - len(genre_tokens)), desc="Generating"):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("Inf")
                
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Stop if we hit an EOS token
            if (next_tokens == tokenizer.tokens.EOS).any():
                break
                
    # Convert generated tokens to MIDI events
    for seq in input_ids:
        events = []
        tokens = seq.tolist()
        
        # Group tokens into events
        i = 0
        while i < len(tokens):
            event_tokens = []
            while i < len(tokens) and tokens[i] not in [tokenizer.tokens.EOS, tokenizer.tokens.PAD]:
                event_tokens.append(tokens[i])
                i += 1
                
            if event_tokens:
                event = tokenizer.tokens_to_event(event_tokens)
                if event:
                    events.append(event)
            i += 1
            
        generated_sequences.append(events)
        
    return generated_sequences

def save_midi(events: List[MIDIEvent], output_path: str):
    """Save MIDI events to a MIDI file."""
    from midiutil import MIDIFile
    
    midi = MIDIFile(1)  # One track
    midi.addTempo(0, 0, 120)  # Default tempo
    
    time = 0
    for event in events:
        if hasattr(event, 'tempo'):
            midi.addTempo(0, time, event.tempo)
            
        if hasattr(event, 'pitch'):
            # Convert duration from ticks to beats
            duration = event.duration / 480  # Assuming standard PPQN
            midi.addNote(0, 0, event.pitch, time, duration, event.velocity)
            
        # Advance time based on duration
        time += event.duration / 480
        
    with open(output_path, "wb") as f:
        midi.writeFile(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated MIDI files"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["base", "large", "medium", "small", "mini", "tiny"],
        help="Model architecture size"
    )
    parser.add_argument(
        "--genre",
        type=str,
        required=True,
        help="Genre to condition generation on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # Load model configuration
    config = MASKED_LM_CONFIG
    if args.model_size:
        config.model_size = args.model_size
        
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = MusicBERTForMaskedLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = MusicTokenizer.from_pretrained(args.model_path)
    
    # Get genre tokens
    genre_tokens = tokenizer.encode_genre(args.genre)
    
    # Generate sequences
    logger.info(f"Generating {args.num_samples} samples for genre: {args.genre}")
    sequences = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        genre_tokens=genre_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_samples,
        device=device
    )
    
    # Save generated sequences
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sequence in enumerate(sequences):
        output_path = output_dir / f"generated_{args.genre}_{i+1}.mid"
        save_midi(sequence, str(output_path))
        logger.info(f"Saved generated MIDI to {output_path}")

if __name__ == "__main__":
    main()
