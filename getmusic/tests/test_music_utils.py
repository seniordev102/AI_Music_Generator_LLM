import pytest
import torch
import tempfile
from pathlib import Path
import numpy as np
from getmusic.utils.music_utils import MusicConverter, quantize_duration

@pytest.fixture
def sample_vocab_file():
    """Create a temporary vocabulary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        vocab = [
            'pitch_60',  # Middle C
            'pitch_62',  # D
            'pitch_64',  # E
            'duration_1.0',
            'duration_0.5',
            'duration_0.25'
        ]
        f.write('\n'.join(vocab))
        return Path(f.name)

@pytest.fixture
def converter(sample_vocab_file):
    """Create MusicConverter instance for testing"""
    return MusicConverter(sample_vocab_file)

def test_load_vocab(converter):
    """Test vocabulary loading"""
    assert len(converter.token2id) == 6
    assert len(converter.id2token) == 6
    assert converter.token2id['pitch_60'] == 0
    assert converter.id2token[0] == 'pitch_60'

def test_tokens_to_midi(converter, tmp_path):
    """Test converting tokens to MIDI"""
    tokens = [0, 3, 1, 4]  # C(1.0), D(0.5)
    output_path = tmp_path / 'test.mid'
    
    converter.tokens_to_midi(tokens, output_path, tempo=120)
    assert output_path.exists()

def test_midi_to_tokens(converter, tmp_path):
    """Test converting MIDI to tokens"""
    # First create a MIDI file
    tokens = [0, 3, 1, 4]  # C(1.0), D(0.5)
    midi_path = tmp_path / 'test.mid'
    converter.tokens_to_midi(tokens, midi_path)
    
    # Now convert back to tokens
    result = converter.midi_to_tokens(midi_path)
    assert len(result) > 0
    assert all(t in converter.id2token for t in result)

def test_save_load_tokens(converter, tmp_path):
    """Test saving and loading tokens"""
    tokens = [0, 1, 2, 3, 4, 5]
    save_path = tmp_path / 'tokens.npy'
    
    converter.save_tokens(tokens, save_path)
    loaded_tokens = converter.load_tokens(save_path)
    
    assert np.array_equal(tokens, loaded_tokens)

def test_quantize_duration():
    """Test duration quantization"""
    # Test exact values
    assert quantize_duration(1.0) == 1.0
    assert quantize_duration(0.5) == 0.5
    assert quantize_duration(0.25) == 0.25
    
    # Test rounding
    assert quantize_duration(0.35) == 0.375  # Should round to nearest 1/8th note
    assert quantize_duration(0.1) == 0.125   # Should round up to minimum duration

def test_invalid_vocab_file():
    """Test handling of invalid vocabulary file"""
    with pytest.raises(FileNotFoundError):
        MusicConverter('nonexistent.txt')

def test_token_sequence_length(converter, tmp_path):
    """Test token sequence length handling"""
    tokens = list(range(600))  # Long sequence
    midi_path = tmp_path / 'long.mid'
    converter.tokens_to_midi(tokens, midi_path)
    
    # Test with max_length
    result = converter.midi_to_tokens(midi_path, max_length=100)
    assert len(result) <= 100