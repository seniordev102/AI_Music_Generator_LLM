import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

class MusicDataset(Dataset):
    """Dataset class for loading music training data"""
    
    def __init__(self,
                 data_path: str,
                 prefix: str = 'train',
                 vocab_size: int = 11880,
                 max_length: int = 512):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.prefix = prefix
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load and preprocess data from disk"""
        data_path = self.data_path / self.prefix
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
        # Load content, finals, sentences
        content_path = data_path / 'content.npy'
        finals_path = data_path / 'finals.npy'
        sentences_path = data_path / 'sentences.npy'
        
        try:
            content = np.load(content_path)
            finals = np.load(finals_path)
            sentences = np.load(sentences_path)
            
            # Validate shapes
            if not (content.shape[0] == finals.shape[0] == sentences.shape[0]):
                raise ValueError("Mismatched number of samples across files")
                
            # Convert to torch tensors
            data = []
            for i in range(len(content)):
                item = {
                    'content': torch.from_numpy(content[i]).long(),
                    'finals': torch.from_numpy(finals[i]).long(),
                    'sentences': torch.from_numpy(sentences[i]).long(),
                }
                
                # Add condition and empty position masks
                item['condition_pos'] = self._get_condition_mask(item['content'])
                item['not_empty_pos'] = self._get_nonempty_mask(item['content'])
                
                data.append(item)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _get_condition_mask(self, content: torch.Tensor) -> torch.Tensor:
        """Generate conditioning mask for content"""
        # Mask is 1 where we want to condition (keep original values)
        # and 0 where we want to generate new values
        mask = torch.zeros_like(content, dtype=torch.bool)
        
        # Add conditioning logic here based on music structure
        # For example, mask chord tokens, beat markers etc.
        
        return mask
        
    def _get_nonempty_mask(self, content: torch.Tensor) -> torch.Tensor:
        """Generate mask for non-empty positions"""
        # Mask is 1 for actual content and 0 for padding
        return content != self.vocab_size - 1  # Assuming last token is padding
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Apply any runtime augmentations here
        
        return item

def create_dataloader(
    dataset: MusicDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False
) -> torch.utils.data.DataLoader:
    """Create dataloader with proper configuration"""
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
        
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not distributed),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )