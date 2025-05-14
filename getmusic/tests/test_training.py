import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
from getmusic.engine.ema import EMA
from getmusic.modeling.config import ModelConfig, DiffusionConfig, RoformerConfig, TrainingConfig
from getmusic.engine.trainer import TrainingManager

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, x):
        return self.bn(self.linear(x))

@pytest.fixture
def model():
    """Create a simple model for testing"""
    return SimpleModel()

@pytest.fixture
def config():
    """Create a test configuration"""
    return ModelConfig(
        diffusion=DiffusionConfig(diffusion_step=10),
        roformer=RoformerConfig(vocab_size=100),
        training=TrainingConfig(batch_size=2),
        device=torch.device('cpu')
    )

def test_ema_initialization(model):
    """Test EMA initialization"""
    ema = EMA(model, decay=0.99, update_interval=1)
    
    # Check shadow parameters exist
    assert len(ema.shadow_params) == sum(1 for _ in model.parameters())
    
    # Check parameters are copied correctly
    for param, shadow in zip(model.parameters(), ema.shadow_params):
        assert torch.equal(param, shadow)

def test_ema_update(model):
    """Test EMA parameter updates"""
    ema = EMA(model, decay=0.9, update_interval=1)
    
    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Modify model parameters
    for p in model.parameters():
        p.data += torch.randn_like(p)
        
    # Update EMA
    ema.update()
    
    # Check shadow parameters are between initial and current values
    for init_p, curr_p, shadow_p in zip(initial_params, 
                                       model.parameters(),
                                       ema.shadow_params):
        assert torch.all((shadow_p >= torch.minimum(init_p, curr_p)) & 
                        (shadow_p <= torch.maximum(init_p, curr_p)))

def test_ema_store_restore(model):
    """Test storing and restoring parameters"""
    ema = EMA(model, decay=0.99)
    
    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    ema.store()
    
    # Modify parameters
    for p in model.parameters():
        p.data += 1.0
        
    # Restore parameters
    ema.restore()
    
    # Check parameters are restored correctly
    for init_p, curr_p in zip(initial_params, model.parameters()):
        assert torch.equal(init_p, curr_p)

def test_trainer_initialization(model, config, tmp_path):
    """Test trainer initialization"""
    trainer = TrainingManager(
        model=model,
        config=config,
        save_dir=str(tmp_path),
        distributed=False
    )
    
    assert trainer.model == model
    assert trainer.config == config
    assert trainer.last_epoch == -1
    assert trainer.ema is not None

def test_trainer_save_load(model, config, tmp_path):
    """Test saving and loading checkpoints"""
    trainer = TrainingManager(
        model=model,
        config=config,
        save_dir=str(tmp_path)
    )
    
    # Save checkpoint
    trainer.save_checkpoint()
    
    # Modify model parameters
    for p in model.parameters():
        p.data += 1.0
        
    # Load checkpoint
    ckpt_path = tmp_path / 'checkpoint' / 'latest.pt'
    trainer.load_checkpoint(ckpt_path)
    
    # Check parameters are restored
    loaded_state = torch.load(ckpt_path)
    for saved_p, curr_p in zip(loaded_state['model'].values(),
                              model.parameters()):
        assert torch.equal(saved_p, curr_p)

@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_learning_rate_scaling(model, config, tmp_path, world_size):
    """Test learning rate scaling with different world sizes"""
    trainer = TrainingManager(
        model=model,
        config=config,
        save_dir=str(tmp_path),
        distributed=(world_size > 1),
        world_size=world_size
    )
    
    base_lr = config.training.base_lr
    if config.training.adjust_lr == 'sqrt':
        expected_lr = base_lr * (world_size * config.training.batch_size) ** 0.5
    elif config.training.adjust_lr == 'linear':
        expected_lr = base_lr * world_size * config.training.batch_size
    else:
        expected_lr = base_lr
        
    assert abs(trainer.optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6