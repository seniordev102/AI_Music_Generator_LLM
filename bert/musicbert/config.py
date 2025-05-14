"""Configuration for MusicBERT model and training."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class MusicBERTConfig:
    """Configuration for MusicBERT model architecture and training."""
    
    # Model architecture
    model_size: str = "base"  # base, large, medium, small, mini, tiny
    max_sequence_length: int = 8192
    vocab_size: int = 30000
    hidden_size: int = 768  # Embedding dimension
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    activation_dropout_prob: float = 0.0
    pooler_dropout_prob: float = 0.0
    layerdrop: float = 0.0
    
    # Training configuration
    batch_size: int = 64
    max_sentences: int = 4
    tokens_per_sample: int = 8192
    total_num_updates: int = 250000
    warmup_updates: int = 50000
    peak_learning_rate: float = 5e-5
    weight_decay: float = 0.1
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    clip_norm: float = 0.0
    
    # Data processing
    mask_prob: float = 0.15
    leave_unmasked_prob: float = 0.1
    random_token_prob: float = 0.1
    mask_whole_words: bool = False
    mask_multiple_length: int = 1
    max_source_positions: int = 8192
    
    # Task specific
    num_classes: int = 2  # For classification tasks
    classification_head_name: Optional[str] = None
    
    def __post_init__(self):
        """Set size-specific parameters after initialization."""
        if self.model_size == "large":
            self.hidden_size = 1024
            self.num_hidden_layers = 24
            self.num_attention_heads = 16
            self.intermediate_size = 4096
            
        elif self.model_size == "medium":
            self.hidden_size = 512
            self.num_hidden_layers = 8
            self.num_attention_heads = 8
            self.intermediate_size = 2048
            
        elif self.model_size == "small":
            self.hidden_size = 512
            self.num_hidden_layers = 4 
            self.num_attention_heads = 8
            self.intermediate_size = 2048
            
        elif self.model_size == "mini":
            self.hidden_size = 256
            self.num_hidden_layers = 4
            self.num_attention_heads = 4
            self.intermediate_size = 1024
            
        elif self.model_size == "tiny":
            self.hidden_size = 128
            self.num_hidden_layers = 2
            self.num_attention_heads = 2
            self.intermediate_size = 512

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicBERTConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

# Default configurations for different training tasks
MASKED_LM_CONFIG = MusicBERTConfig(
    model_size="base",
    total_num_updates=125000,
    warmup_updates=25000,
    peak_learning_rate=5e-4,
    batch_size=256,
    weight_decay=0.01
)

GENRE_CLASSIFICATION_CONFIG = MusicBERTConfig(
    model_size="base", 
    total_num_updates=20000,
    warmup_updates=4000,
    peak_learning_rate=5e-5,
    batch_size=64,
    weight_decay=0.01,
    classification_head_name="genre_head"
)

NEXT_SENTENCE_CONFIG = MusicBERTConfig(
    model_size="base",
    total_num_updates=250000, 
    warmup_updates=50000,
    peak_learning_rate=5e-5,
    batch_size=64,
    num_classes=2,
    classification_head_name="nsp_head"
)