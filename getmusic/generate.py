import argparse
import torch
import logging
from pathlib import Path
from typing import Optional

from getmusic.modeling.config import ModelConfig
from getmusic.modeling.generator import MusicGenerator
from getmusic.utils.music_utils import MusicConverter

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                      help='Path to config file (optional if included in checkpoint)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Output directory for generated files')
    parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of samples to generate')
    parser.add_argument('--condition_midi', type=str,
                      help='Optional MIDI file to condition generation on')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Sampling temperature')
    parser.add_argument('--top_k', type=int,
                      help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float,
                      help='Nucleus sampling parameter')
    parser.add_argument('--tempo', type=int, default=120,
                      help='Music tempo in BPM')
    parser.add_argument('--seed', type=int,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / 'generate.log'),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f'Using random seed: {args.seed}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load config if provided
    config = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config = ModelConfig.from_dict(yaml.safe_load(f))
    
    # Initialize generator
    logger.info(f'Loading model from checkpoint: {args.checkpoint}')
    generator = MusicGenerator.from_pretrained(
        args.checkpoint,
        config=config,
        device=device
    )
    
    # Initialize music converter
    converter = MusicConverter(generator.config.training.vocab_path)
    
    # Load conditioning if provided
    condition = None
    if args.condition_midi:
        logger.info(f'Loading conditioning from MIDI: {args.condition_midi}')
        condition = torch.tensor(
            converter.midi_to_tokens(args.condition_midi),
            device=device
        )
    
    # Generate samples
    logger.info(f'Generating {args.num_samples} samples...')
    samples = generator.generate(
        num_samples=args.num_samples,
        condition=condition,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length
    )
    
    # Save generated samples
    for i, sample in enumerate(samples):
        # Convert to MIDI
        output_path = output_dir / f'sample_{i:03d}.mid'
        logger.info(f'Saving sample to {output_path}')
        converter.tokens_to_midi(
            sample.cpu().tolist(),
            output_path,
            tempo=args.tempo
        )

if __name__ == '__main__':
    main()