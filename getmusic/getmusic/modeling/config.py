from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch

@dataclass
class DiffusionConfig:
    diffusion_step: int = 100
    alpha_init_type: str = 'alpha1'
    auxiliary_loss_weight: float = 0.001
    adaptive_auxiliary_loss: bool = True
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_update_interval: int = 1

@dataclass
class RoformerConfig:
    vocab_size: int = 11880
    cond_weight: float = 0.5
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    pre_layernorm: bool = True

@dataclass
class TrainingConfig:
    base_lr: float = 3.0e-6
    adjust_lr: str = 'none'  # 'none', 'sqrt' or 'linear'
    max_epochs: int = 50 
    save_epochs: int = 10
    validation_epochs: int = 1
    sample_iterations: str = 'epoch'
    validate_iterations: int = 1000
    vocab_path: str = 'getmusic/utils/dict.txt'
    print_specific_things: bool = True
    batch_size: int = 3
    num_workers: int = 28
    clip_grad_norm_max: float = 0.5

@dataclass 
class ModelConfig:
    """Configuration for model architecture and training"""
    diffusion: DiffusionConfig
    roformer: RoformerConfig
    training: TrainingConfig
    device: torch.device = torch.device('cpu')

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        diffusion_config = DiffusionConfig(**config.get('diffusion', {}))
        roformer_config = RoformerConfig(**config.get('roformer', {}))
        training_config = TrainingConfig(**config.get('training', {}))
        
        return cls(
            diffusion=diffusion_config,
            roformer=roformer_config, 
            training=training_config,
            device=torch.device(config.get('device', 'cpu'))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'diffusion': self.diffusion.__dict__,
            'roformer': self.roformer.__dict__,
            'training': self.training.__dict__,
            'device': str(self.device)
        }