import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
from .base import BaseDiffusionModel
from .config import ModelConfig
from .roformer_utils import DiffusionRoformerModel

class DiffusionRoformer(BaseDiffusionModel):
    """RoFormer-based diffusion model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.roformer = DiffusionRoformerModel(config.roformer)
        self.prior_ps = 1024  # max number to sample per step
        self.figure_size = config.roformer.max_position_embeddings
        self.tracks = 14  # Number of music tracks/voices
        self.pad_index = config.roformer.pad_token_id
        
    def log_sample_categorical(self, logits: torch.Tensor, inference: bool = False, 
                             figure_size: Optional[int] = None) -> torch.Tensor:
        """Sample from categorical distribution using Gumbel softmax"""
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = gumbel_noise + logits

        if inference:
            # Handle track-specific decoding during inference
            for i in range(self.tracks - 2):  # Skip chord tracks
                track = sample[:, :, i * self.figure_size:(i+1) * self.figure_size]
                if i % 2 == 1:  # Duration track
                    track[:, self.pad_index+1:-1, :] = -70
                else:  # Pitch track
                    track[:, :self.pad_index, :] = -70
                    track[:, self.pad_index+1:self.tracks_start[i//2], :] = -70 
                    track[:, self.tracks_end[i//2]+1:-1, :] = -70
                sample[:, :, i * self.figure_size:(i+1) * self.figure_size] = track

        sample = sample.argmax(dim=1)
        return self._index_to_log_onehot(sample)

    def q_pred(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute q(x_t|x_0)"""
        # Handle t wrapping for training
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        
        log_cumprod_at = self.extract(self.log_cumprod_at, t, log_x_start.shape)
        log_cumprod_bt = self.extract(self.log_cumprod_bt, t, log_x_start.shape)
        log_cumprod_ct = self.extract(self.log_cumprod_ct, t, log_x_start.shape)
        log_1_min_cumprod_ct = self.extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)

        log_probs = torch.cat([
            self._log_add_exp(
                log_x_start[:, :-1, :] + log_cumprod_at,
                log_cumprod_bt
            ),
            self._log_add_exp(
                log_x_start[:, -1:, :] + log_1_min_cumprod_ct,
                log_cumprod_ct
            )
        ], dim=1)

        return log_probs

    def p_pred(self, log_x: torch.Tensor, t: torch.Tensor, 
               condition_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute p(x_{t-1}|x_t)"""
        if not t.numel() == 1:
            raise ValueError("t should be a single timestep tensor")
            
        batch_size = log_x.shape[0]
        log_x_recon = self.predict_start(log_x, t, condition_pos)  # x_0
        
        log_model_pred = self.q_pred(log_x_recon, t)
        log_true_prob = self.q_pred(log_x, t)
        
        # Mask tokens that should not be predicted
        if condition_pos is not None:
            log_model_pred = torch.where(
                condition_pos.unsqueeze(1).expand(-1, self.num_classes, -1),
                log_model_pred,
                log_true_prob
            )
            
        return log_model_pred

    def predict_start(self, log_x_t: torch.Tensor, t: torch.Tensor,
                     condition_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict x_0 given x_t"""
        x_t = self._log_onehot_to_index(log_x_t)
        out = self.roformer(x_t, t, condition_pos)
        
        log_pred = F.log_softmax(out.double(), dim=2).float()
        batch_size = log_x_t.size()[0]
        
        # Add padding predictions
        zero_vector = torch.zeros(batch_size, out.size()[1], 2).type_as(log_x_t) - 70
        log_pred = torch.cat((log_pred, zero_vector), dim=2)
        log_pred = torch.clamp(log_pred, -70, 0)
        
        return log_pred.transpose(2, 1)

    def _index_to_log_onehot(self, x: torch.Tensor) -> torch.Tensor:
        """Convert index tensor to log one-hot encoding"""
        x_onehot = F.one_hot(x, self.num_classes)
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
        log_x = torch.log(x_onehot.float().clamp(min=1e-30))
        return log_x

    def _log_onehot_to_index(self, log_x: torch.Tensor) -> torch.Tensor:
        """Convert log one-hot tensor to index tensor"""
        return log_x.argmax(1)

    def forward(self, batch: torch.Tensor, tempo: torch.Tensor,
                condition_pos: Optional[torch.Tensor] = None,
                not_empty_pos: Optional[torch.Tensor] = None,
                return_loss: bool = True,
                step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass implementation"""
        b, device = batch.size(0), batch.device
        
        # Sample timestep
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Convert input to log-onehot
        log_x_start = self._index_to_log_onehot(batch)
        
        # Get noise prediction
        noise = torch.rand_like(log_x_start)
        log_x_t = self.q_sample(log_x_start, t)
        
        # Model prediction
        log_model_prob = self.p_pred(log_x_t, t, condition_pos)
        
        # Calculate loss
        loss_dict = {}
        if return_loss:
            loss = -self.log_categorical(log_x_start, log_model_prob)
            
            # Apply conditioning mask if provided
            if condition_pos is not None:
                loss = loss * (1 - condition_pos)
            
            # Track metrics
            loss_dict["loss"] = loss.mean()
            if not_empty_pos is not None:
                loss_dict["loss_not_empty"] = (loss * not_empty_pos).sum() / not_empty_pos.sum()
        
        return loss_dict
        
    def log_categorical(self, log_x_start: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute categorical log probability"""
        return (log_x_start.exp() * log_prob).sum(dim=1)

    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t|x_0)"""
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        return self.log_sample_categorical(log_EV_qxt_x0)