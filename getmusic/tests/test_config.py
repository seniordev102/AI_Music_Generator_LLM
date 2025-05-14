import pytest
import torch
from getmusic.modeling.config import (
    ModelConfig, DiffusionConfig, RoformerConfig, TrainingConfig
)

def test_model_config_creation():
    """Test basic model config creation"""
    config = ModelConfig(
        diffusion=DiffusionConfig(),
        roformer=RoformerConfig(),
        training=TrainingConfig(),
        device=torch.device('cpu')
    )
    
    assert config.diffusion.diffusion_step == 100
    assert config.roformer.vocab_size == 11880
    assert config.training.batch_size == 3
    
def test_model_config_from_dict():
    """Test creating config from dictionary"""
    config_dict = {
        'diffusion': {
            'diffusion_step': 50,
            'alpha_init_type': 'alpha2'
        },
        'roformer': {
            'vocab_size': 5000,
            'hidden_size': 512
        },
        'training': {
            'batch_size': 8,
            'max_epochs': 100
        },
        'device': 'cuda'
    }
    
    config = ModelConfig.from_dict(config_dict)
    
    assert config.diffusion.diffusion_step == 50
    assert config.diffusion.alpha_init_type == 'alpha2'
    assert config.roformer.vocab_size == 5000
    assert config.roformer.hidden_size == 512
    assert config.training.batch_size == 8
    assert config.training.max_epochs == 100
    assert str(config.device) == 'cuda'
    
def test_config_to_dict():
    """Test converting config to dictionary"""
    original_config = ModelConfig(
        diffusion=DiffusionConfig(diffusion_step=75),
        roformer=RoformerConfig(vocab_size=8000),
        training=TrainingConfig(batch_size=4),
        device=torch.device('cpu')
    )
    
    config_dict = original_config.to_dict()
    
    assert config_dict['diffusion']['diffusion_step'] == 75
    assert config_dict['roformer']['vocab_size'] == 8000
    assert config_dict['training']['batch_size'] == 4
    assert config_dict['device'] == 'cpu'
    
def test_invalid_config_values():
    """Test validation of config values"""
    with pytest.raises(ValueError):
        DiffusionConfig(diffusion_step=-1)
        
    with pytest.raises(ValueError):
        RoformerConfig(vocab_size=0)
        
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=-2)