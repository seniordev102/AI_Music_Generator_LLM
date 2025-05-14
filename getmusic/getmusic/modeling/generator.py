import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from .config import ModelConfig
from .diffusion_roformer import DiffusionRoformer

class MusicGenerator:
    """Interface for music generation using trained models"""
    
    def __init__(self,
                 model: DiffusionRoformer,
                 config: ModelConfig,
                 device: Optional[torch.device] = None):
        """Initialize generator
        
        Args:
            model: Trained diffusion model
            config: Model configuration
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.device = device or model.device
        
        self.model.eval()
        if device:
            self.model = self.model.to(device)
            
    @classmethod
    def from_pretrained(cls, 
                       checkpoint_path: Union[str, Path],
                       config: Optional[ModelConfig] = None,
                       device: Optional[torch.device] = None) -> 'MusicGenerator':
        """Load pretrained model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config
        if config is None:
            if 'config' in state:
                config = ModelConfig.from_dict(state['config'])
            else:
                raise ValueError("No config found in checkpoint and none provided")
                
        # Create model
        model = DiffusionRoformer(config)
        
        # Load weights
        if 'ema' in state:
            model.load_state_dict(state['ema'])
        else:
            model.load_state_dict(state['model'])
            
        return cls(model, config, device)
        
    @torch.no_grad()
    def generate(self,
                num_samples: int = 1,
                condition: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                use_sampling: bool = True,
                max_length: Optional[int] = None) -> torch.Tensor:
        """Generate music samples
        
        Args:
            num_samples: Number of samples to generate
            condition: Optional conditioning input
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling threshold
            use_sampling: Whether to sample or use argmax
            max_length: Maximum sequence length
        
        Returns:
            Generated token sequences
        """
        self.model.eval()
        device = self.device
        
        # Setup generation parameters
        max_length = max_length or self.config.roformer.max_position_embeddings
        batch_size = num_samples
        
        # Initialize sequence
        x = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        
        # Setup conditioning if provided
        if condition is not None:
            if isinstance(condition, (list, tuple)):
                condition = torch.tensor(condition, device=device)
            condition = condition.to(device)
            
            # Create conditioning mask
            condition_pos = torch.ones_like(x, dtype=torch.bool, device=device)
            if len(condition.shape) == 1:
                condition = condition.unsqueeze(0).expand(batch_size, -1)
            x[:, :condition.size(1)] = condition
            condition_pos[:, condition.size(1):] = False
        else:
            condition_pos = None
            
        # Generate tokens step by step
        for t in range(self.config.diffusion.diffusion_step):
            # Forward pass
            logits = self.model.p_pred(x, torch.tensor([t], device=device), 
                                     condition_pos=condition_pos)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply sampling methods
            if use_sampling:
                if top_k is not None:
                    logits = self._top_k_logits(logits, top_k)
                if top_p is not None:
                    logits = self._top_p_logits(logits, top_p)
                    
                # Sample from logits
                probs = torch.softmax(logits, dim=-1)
                x_t = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                x_t = torch.argmax(logits, dim=-1)
                
            # Update sequence
            if condition_pos is not None:
                x = torch.where(condition_pos, x, x_t)
            else:
                x = x_t
                
        return x
        
    def _top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k sampling to logits"""
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out
        
    def _top_p_logits(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('Inf')
        return logits