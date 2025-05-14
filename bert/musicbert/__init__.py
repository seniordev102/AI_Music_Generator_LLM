"""MusicBERT model and utilities."""

from .config import (
    MusicBERTConfig,
    MASKED_LM_CONFIG,
    GENRE_CLASSIFICATION_CONFIG,
    NEXT_SENTENCE_CONFIG
)
from .model import (
    MusicBERTOutput,
    MusicBERTModel,
    MusicBERTPreTrainedModel
)
from .task_models import (
    MusicBERTForMaskedLM,
    MusicBERTForGenreClassification,
    MusicBERTForNextSequencePrediction
)
from .data_utils import (
    MusicTokenizer,
    MusicDataset,
    load_and_cache_examples
)
from .training_utils import (
    TrainingMetrics,
    MetricsCalculator,
    Trainer
)

__all__ = [
    # Config
    'MusicBERTConfig',
    'MASKED_LM_CONFIG',
    'GENRE_CLASSIFICATION_CONFIG',
    'NEXT_SENTENCE_CONFIG',
    
    # Core model
    'MusicBERTOutput',
    'MusicBERTModel',
    'MusicBERTPreTrainedModel',
    
    # Task-specific models
    'MusicBERTForMaskedLM',
    'MusicBERTForGenreClassification',
    'MusicBERTForNextSequencePrediction',
    
    # Data utilities
    'MusicTokenizer',
    'MusicDataset',
    'load_and_cache_examples',
    
    # Training utilities
    'TrainingMetrics',
    'MetricsCalculator',
    'Trainer'
]
