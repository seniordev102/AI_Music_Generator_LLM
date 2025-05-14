"""Data processing utilities for MusicBERT."""

import os
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils
from fairseq.data.encoders.utils import get_whole_word_mask

from .config import MusicBERTConfig

logger = logging.getLogger(__name__)

@dataclass
class MusicTokens:
    """Container for special MIDI tokens."""
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3
    MASK: int = 4

@dataclass 
class MIDIEvent:
    """Representation of a MIDI event."""
    bar: int
    position: int  # Position within bar
    program: int  # MIDI program number
    pitch: int
    duration: int
    velocity: int
    time_sig: Optional[Tuple[int, int]] = None  # Numerator, denominator
    tempo: Optional[float] = None  # In BPM

class MusicTokenizer:
    """Handles tokenization of MIDI events."""
    
    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary
        self.tokens = MusicTokens()
        
    def event_to_tokens(self, event: MIDIEvent) -> List[int]:
        """Convert MIDI event to token sequence."""
        tokens = []
        
        # Bar marker
        if event.bar is not None:
            tokens.append(self.dictionary.index(f"BAR_{event.bar}"))
            
        # Position
        if event.position is not None:
            tokens.append(self.dictionary.index(f"POS_{event.position}"))
            
        # Program 
        if event.program is not None:
            tokens.append(self.dictionary.index(f"PROG_{event.program}"))
            
        # Note info
        tokens.extend([
            self.dictionary.index(f"PITCH_{event.pitch}"),
            self.dictionary.index(f"DUR_{event.duration}"),
            self.dictionary.index(f"VEL_{event.velocity}")
        ])
        
        # Time signature
        if event.time_sig:
            num, denom = event.time_sig
            tokens.append(self.dictionary.index(f"TIME_{num}_{denom}"))
            
        # Tempo
        if event.tempo:
            tokens.append(self.dictionary.index(f"TEMPO_{int(event.tempo)}"))
            
        return tokens
        
    def tokens_to_event(self, tokens: List[int]) -> Optional[MIDIEvent]:
        """Convert token sequence back to MIDI event."""
        try:
            # Parse tokens back to event attributes
            event_dict = {}
            for token in tokens:
                token_str = self.dictionary[token]
                
                if token_str.startswith("BAR_"):
                    event_dict["bar"] = int(token_str[4:])
                elif token_str.startswith("POS_"):
                    event_dict["position"] = int(token_str[4:])
                elif token_str.startswith("PROG_"):
                    event_dict["program"] = int(token_str[5:])
                elif token_str.startswith("PITCH_"):
                    event_dict["pitch"] = int(token_str[6:])
                elif token_str.startswith("DUR_"):
                    event_dict["duration"] = int(token_str[4:])
                elif token_str.startswith("VEL_"):
                    event_dict["velocity"] = int(token_str[4:])
                elif token_str.startswith("TIME_"):
                    num, denom = map(int, token_str[5:].split("_"))
                    event_dict["time_sig"] = (num, denom)
                elif token_str.startswith("TEMPO_"):
                    event_dict["tempo"] = float(token_str[6:])
                    
            return MIDIEvent(**event_dict)
            
        except Exception as e:
            logger.warning(f"Failed to convert tokens to event: {str(e)}")
            return None

class MusicDataset(torch.utils.data.Dataset):
    """Dataset for music sequence training."""
    
    def __init__(
        self,
        path: str,
        dictionary: Dictionary,
        config: MusicBERTConfig,
        is_train: bool = True
    ):
        """Initialize dataset.
        
        Args:
            path: Path to data directory
            dictionary: Token dictionary
            config: Model configuration
            is_train: Whether this is training data
        """
        self.path = Path(path)
        self.dictionary = dictionary
        self.config = config
        self.is_train = is_train
        
        self.tokenizer = MusicTokenizer(dictionary)
        
        # Load data
        self.examples = self._load_examples()
        self.sizes = np.array([len(x) for x in self.examples])
        
        # Setup masking
        self.mask_idx = dictionary.index("<mask>")
        self.mask_whole_words = get_whole_word_mask(config, dictionary)
        
    def _load_examples(self) -> List[torch.Tensor]:
        """Load examples from disk."""
        examples = []
        
        input_path = self.path / ("train.txt" if self.is_train else "valid.txt")
        if not input_path.exists():
            raise FileNotFoundError(f"Data file not found: {input_path}")
            
        logger.info(f"Loading data from {input_path}")
        with open(input_path) as f:
            for line in f:
                tokens = self.dictionary.encode_line(
                    line.strip(),
                    append_eos=True,
                    add_if_not_exist=False
                )
                examples.append(tokens)
                
        return examples
        
    def __getitem__(self, index: int) -> torch.Tensor:
        """Get an example from the dataset."""
        item = self.examples[index]
        
        if self.is_train:
            item = self._mask_tokens(item)
            
        return item
        
    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.examples)
        
    def _mask_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mask tokens for masked language model training."""
        assert self.mask_idx >= 0, "Mask index must be non-negative"
        
        # Generate masking probability tensor
        mask = torch.full(tokens.shape, self.config.mask_prob)
        mask[tokens == self.dictionary.pad()] = 0
        mask[tokens == self.dictionary.eos()] = 0
        mask[tokens == self.dictionary.bos()] = 0
        
        # Get indices to mask
        masked_indices = torch.bernoulli(mask).bool()
        
        # Replace with mask/random/unchanged
        result = tokens.clone()
        
        # Mask tokens
        mask_indices = masked_indices & \
            (torch.rand(tokens.shape) < self.config.mask_prob)
        result[mask_indices] = self.mask_idx
        
        # Random tokens
        random_indices = masked_indices & \
            (torch.rand(tokens.shape) < self.config.random_token_prob)
        random_tokens = torch.randint(
            len(self.dictionary),
            tokens.shape,
            dtype=tokens.dtype
        )
        result[random_indices] = random_tokens[random_indices]
        
        # Leave unchanged
        pass_indices = masked_indices & \
            (torch.rand(tokens.shape) < self.config.leave_unmasked_prob)
        result[pass_indices] = tokens[pass_indices]
        
        return result

def load_and_cache_examples(
    data_dir: Union[str, Path],
    dictionary: Dictionary,
    config: MusicBERTConfig,
    evaluate: bool = False
) -> MusicDataset:
    """Load data and cache features.
    
    Args:
        data_dir: Directory containing the data files
        dictionary: Token dictionary
        config: Model configuration
        evaluate: Whether to load evaluation data
        
    Returns:
        MusicDataset instance
    """
    return MusicDataset(
        path=data_dir,
        dictionary=dictionary,
        config=config,
        is_train=not evaluate
    )