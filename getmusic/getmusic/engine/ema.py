import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Dict

class EMA:
    """Exponential Moving Average for model parameters
    
    Maintains a shadow copy of model parameters and updates them using
    exponential moving average during training.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 decay: float = 0.999,
                 update_interval: int = 1,
                 device: Optional[torch.device] = None):
        """Initialize EMA instance
        
        Args:
            model: Model whose parameters to track
            decay: EMA decay rate (default: 0.999)
            update_interval: Update shadow params every N steps (default: 1)
            device: Device to store parameters on (default: same as model)
        """
        self.model = model
        self.decay = decay
        self.update_interval = update_interval
        self.device = device or next(model.parameters()).device
        
        # Create shadow parameters
        self.shadow_params = [
            p.clone().detach().to(device)
            for p in model.parameters()
        ]
        
        self.collected_params = []
        self.num_updates = 0
        
    def update(self, iteration: Optional[int] = None) -> None:
        """Update shadow parameters using EMA
        
        Args:
            iteration: Current training iteration
        """
        if iteration is not None and iteration % self.update_interval != 0:
            return
            
        decay = self.decay
        
        with torch.no_grad():
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            
            for s_param, param in zip(self.shadow_params, parameters):
                # Update shadow parameter
                s_param.sub_((1 - decay) * (s_param - param))
                
        self.num_updates += 1
        
    def store(self) -> None:
        """Store current model parameters"""
        self.collected_params = [
            param.clone().detach()
            for param in self.model.parameters()
        ]
        
    def restore(self) -> None:
        """Restore stored parameters"""
        if not self.collected_params:
            raise RuntimeError("No parameters to restore")
            
        for param, collected in zip(self.model.parameters(), self.collected_params):
            param.data.copy_(collected.data)
            
    def apply_shadow(self) -> None:
        """Apply shadow parameters to model"""
        self.store()
        for param, shadow in zip(self.model.parameters(), self.shadow_params):
            param.data.copy_(shadow.data)
            
    def restore_backup(self) -> None:
        """Restore original parameters from backup"""
        self.restore()
        
    def state_dict(self) -> Dict:
        """Return state dict for checkpointing"""
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params,
            'num_updates': self.num_updates
        }
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']
        self.num_updates = state_dict['num_updates']


