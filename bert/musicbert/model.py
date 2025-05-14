"""MusicBERT model architecture."""

import math
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .config import MusicBERTConfig

@dataclass
class MusicBERTOutput:
    """Container for model outputs."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MusicBERTEmbeddings(nn.Module):
    """Input embeddings for MusicBERT."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length,
            config.hidden_size
        )
        self.LayerNorm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs (from 0 to max_sequence_length-1)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_sequence_length).expand((1, -1))
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            position_ids: Optional position IDs
            
        Returns:
            Embedded representation [batch_size, seq_length, hidden_size]
        """
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MusicBERTSelfAttention(nn.Module):
    """Multi-headed self-attention."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for attention computation."""
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class MusicBERTLayer(nn.Module):
    """Transformer layer with attention and feed-forward networks."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()
        self.attention = MusicBERTSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.LayerNorm1 = LayerNorm(config.hidden_size)
        self.LayerNorm2 = LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.activation = F.gelu
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass."""
        # Self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions
        )
        attention_output = attention_outputs[0]
        
        # First residual connection
        hidden_states = self.LayerNorm1(hidden_states + self.dropout1(attention_output))
        
        # Feed-forward network
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        layer_output = self.output(intermediate_output)
        
        # Second residual connection
        layer_output = self.LayerNorm2(hidden_states + self.dropout2(layer_output))
        
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class MusicBERTEncoder(nn.Module):
    """Stack of Transformer layers."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            MusicBERTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through encoder stack."""
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs

class MusicBERTPooler(nn.Module):
    """Pool the output of the encoder for classification tasks."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.pooler_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool first token ([CLS]) representation."""
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

class MusicBERTPreTrainedModel(nn.Module):
    """Base class for all MusicBERT models."""
    
    config_class = MusicBERTConfig
    base_model_prefix = "musicbert"
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__()        
        self.config = config
        
    def init_weights(self):
        """Initialize weights."""
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class MusicBERTModel(MusicBERTPreTrainedModel):
    """MusicBERT model with pre-training heads."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__(config)
        
        self.embeddings = MusicBERTEmbeddings(config)
        self.encoder = MusicBERTEncoder(config)
        self.pooler = MusicBERTPooler(config)
        
        # Initialize weights
        self.init_weights()
        
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings
        
    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MusicBERTOutput]:
        """Forward pass through model."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
            
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
            
        return MusicBERTOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )