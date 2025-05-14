from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .config import ModelConfig

class BaseDiffusionModel(nn.Module, ABC):
    """Base class for diffusion models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_timesteps = config.diffusion.diffusion_step
        self.num_classes = config.roformer.vocab_size + 1  # Add mask token
        self.register_diffusion_parameters()
        
    def register_diffusion_parameters(self) -> None:
        """Initialize and register diffusion schedule parameters"""
        # Calculate alpha schedule
        if self.config.diffusion.alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = self._alpha_schedule(
                self.num_timesteps,
                N=self.num_classes-1
            )
        else:
            raise ValueError(f"Unknown alpha_init_type: {self.config.diffusion.alpha_init_type}")
            
        # Convert to torch tensors and register buffers
        self._register_schedule_parameters(at, bt, ct, att, btt, ctt)
        
    def _alpha_schedule(self, time_step: int, N: int = 100, 
                       att_1: float = 0.99999, att_T: float = 0.000009,
                       ctt_1: float = 0.000009, ctt_T: float = 0.99999) -> Tuple[np.ndarray, ...]:
        """Calculate alpha schedule parameters"""
        # Calculate schedule values
        att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
        att = np.concatenate(([1], att))
        at = att[1:]/att[:-1]
        
        ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
        ctt = np.concatenate(([0], ctt))
        one_minus_ctt = 1 - ctt
        one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
        ct = 1-one_minus_ct
        
        bt = (1-at-ct)/N
        att = np.concatenate((att[1:], [1]))
        ctt = np.concatenate((ctt[1:], [0]))
        btt = (1-att-ctt)/N
        
        return at, bt, ct, att, btt, ctt
        
    def _register_schedule_parameters(self, at: np.ndarray, bt: np.ndarray,
                                    ct: np.ndarray, att: np.ndarray,
                                    btt: np.ndarray, ctt: np.ndarray) -> None:
        """Register diffusion schedule parameters as model buffers"""
        # Convert to torch tensors
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        
        # Calculate log values
        log_at = torch.log(at) 
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        
        log_1_min_ct = self._log_1_min_a(log_ct)
        log_1_min_cumprod_ct = self._log_1_min_a(log_cumprod_ct)
        
        # Register all parameters as buffers
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        
        # Loss tracking buffers
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract appropriate timestep values from a schedule tensor"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        
    def _log_1_min_a(self, a: torch.Tensor) -> torch.Tensor:
        """Calculate log(1-exp(a))"""
        return torch.log(1 - a.exp() + 1e-40)
        
    def _log_add_exp(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Stable implementation of log(exp(a) + exp(b))"""
        maximum = torch.max(a, b)
        return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))
        
    @abstractmethod
    def q_pred(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Abstract method for q(x_t|x_0) prediction"""
        pass
        
    @abstractmethod
    def p_pred(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Abstract method for p(x_{t-1}|x_t) prediction"""
        pass
        
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Abstract forward method"""
        pass