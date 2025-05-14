from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data processing"""
    input_path: Path
    output_path: Path
    with_beat: bool = False
    beat_mode: int = 0
    min_length: int = 4
    num_pieces: int = 10
    stride: int = 512
    enable_final: bool = True
    enable_sentence: bool = True
    enable_pos: bool = True
    enable_beat: bool = False
    segment: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate config"""
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
            
        self.validate()
        
    def validate(self):
        """Validate configuration parameters"""
        if not self.input_path.exists():
            raise ValueError(f"Input path {self.input_path} does not exist")
        
        if not 0 <= self.beat_mode <= 2:
            raise ValueError(f"Invalid beat_mode {self.beat_mode}. Must be 0, 1, or 2")
            
        if self.min_length < 1:
            raise ValueError(f"min_length must be positive, got {self.min_length}")
            
        if self.stride < 1:
            raise ValueError(f"stride must be positive, got {self.stride}")