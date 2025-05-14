import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import midiutil
from music21 import converter, stream, note, chord, instrument

class MusicConverter:
    """Utilities for converting between different music formats"""
    
    def __init__(self, vocab_path: Union[str, Path]):
        """Initialize converter with vocabulary
        
        Args:
            vocab_path: Path to vocabulary file
        """
        self.vocab_path = Path(vocab_path)
        self.load_vocab()
        
    def load_vocab(self) -> None:
        """Load vocabulary from file"""
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
            
        # Load token to id mapping
        with open(self.vocab_path) as f:
            vocab = [line.strip() for line in f]
        self.token2id = {token: i for i, token in enumerate(vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}
        
    def tokens_to_midi(self, 
                      tokens: List[int],
                      output_path: Union[str, Path],
                      tempo: int = 120) -> None:
        """Convert token sequence to MIDI file
        
        Args:
            tokens: List of token IDs
            output_path: Output MIDI file path
            tempo: Music tempo in BPM
        """
        # Convert IDs to tokens
        token_seq = [self.id2token[t] for t in tokens]
        
        # Create MIDI file
        midi = midiutil.MIDIFile(1)  # One track
        midi.addTempo(0, 0, tempo)
        
        # Parse tokens and add notes
        time = 0
        for token in token_seq:
            if token.startswith('pitch_'):
                pitch = int(token.split('_')[1])
                # Look ahead for duration
                dur_idx = token_seq.index(next(t for t in token_seq if t.startswith('duration_')))
                duration = float(token_seq[dur_idx].split('_')[1])
                
                midi.addNote(0, 0, pitch, time, duration, 100)
                time += duration
                
        # Write MIDI file
        with open(output_path, 'wb') as f:
            midi.writeFile(f)
            
    def midi_to_tokens(self, 
                      midi_path: Union[str, Path],
                      max_length: Optional[int] = None) -> List[int]:
        """Convert MIDI file to token sequence
        
        Args:
            midi_path: Input MIDI file path
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        # Load MIDI file
        midi = converter.parse(str(midi_path))
        
        tokens = []
        for element in midi.flatten():
            if isinstance(element, note.Note):
                # Add pitch token
                pitch_token = f'pitch_{element.pitch.midi}'
                if pitch_token in self.token2id:
                    tokens.append(self.token2id[pitch_token])
                
                # Add duration token
                dur_token = f'duration_{element.duration.quarterLength}'
                if dur_token in self.token2id:
                    tokens.append(self.token2id[dur_token])
                    
            elif isinstance(element, chord.Chord):
                # Handle chords if needed
                for pitch in element.pitches:
                    pitch_token = f'pitch_{pitch.midi}'
                    if pitch_token in self.token2id:
                        tokens.append(self.token2id[pitch_token])
                
                dur_token = f'duration_{element.duration.quarterLength}'
                if dur_token in self.token2id:
                    tokens.append(self.token2id[dur_token])
                    
        # Truncate if needed
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
        
    def save_tokens(self,
                   tokens: List[int],
                   output_path: Union[str, Path]) -> None:
        """Save token sequence to file
        
        Args:
            tokens: List of token IDs
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy array
        np.save(output_path, np.array(tokens))
        
    def load_tokens(self, input_path: Union[str, Path]) -> List[int]:
        """Load token sequence from file
        
        Args:
            input_path: Input file path
            
        Returns:
            List of token IDs
        """
        return np.load(input_path).tolist()
        
def quantize_duration(duration: float, 
                     ticks_per_beat: int = 480,
                     min_duration: float = 0.125) -> float:
    """Quantize note duration to nearest allowed value
    
    Args:  
        duration: Duration in quarter notes
        ticks_per_beat: MIDI ticks per quarter note
        min_duration: Minimum allowed duration
        
    Returns:
        Quantized duration
    """
    # Convert to ticks
    dur_ticks = int(duration * ticks_per_beat)
    
    # Quantize to minimum duration
    min_ticks = int(min_duration * ticks_per_beat)
    dur_ticks = max(min_ticks, dur_ticks)
    
    # Round to nearest allowed duration
    dur_ticks = int(round(dur_ticks / min_ticks) * min_ticks)
    
    return dur_ticks / ticks_per_beat