"""Task-specific MusicBERT model variants."""

from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MusicBERTConfig
from .model import MusicBERTModel, MusicBERTOutput

class MusicBERTForMaskedLM(MusicBERTModel):
    """MusicBERT model with masked language modeling head."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MusicBERTOutput]:
        """Forward pass."""
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            
        return MusicBERTOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class MusicBERTForGenreClassification(MusicBERTModel):
    """MusicBERT model for genre classification."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__(config)
        
        self.num_labels = config.num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MusicBERTOutput]:
        """Forward pass."""
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if labels.dtype in [torch.long, torch.int]:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
                    
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return MusicBERTOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class MusicBERTForNextSequencePrediction(MusicBERTModel):
    """MusicBERT model for next sequence prediction."""
    
    def __init__(self, config: MusicBERTConfig):
        super().__init__(config)
        
        self.nsp_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )
        
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MusicBERTOutput]:
        """Forward pass."""
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        seq_relationship_score = self.nsp_head(pooled_output)
        
        next_sequence_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sequence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                labels.view(-1)
            )
            
        if not return_dict:
            output = (seq_relationship_score,) + outputs[2:]
            return ((next_sequence_loss,) + output) if next_sequence_loss is not None else output
            
        return MusicBERTOutput(
            loss=next_sequence_loss,
            logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )