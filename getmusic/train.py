import os
import argparse
import logging
import torch
import torch.distributed as dist
import yaml
from pathlib import Path
from typing import Dict, Any

from getmusic.modeling.config import ModelConfig
from getmusic.modeling.diffusion_roformer import DiffusionRoformer
from getmusic.data.dataset import MusicDataset, create_dataloader
from getmusic.engine.trainer import TrainingManager

def parse_args():
    parser = argparse.ArgumentParser(description='Music Generation Training')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    
    # Distributed training params
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate config file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def setup_distributed(args):
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                          world_size=args.world_size, rank=args.rank)
    dist.barrier()
    
def setup_logging(output_dir: str, rank: int):
    """Setup logging configuration"""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    logging.basicConfig(
        format=log_format,
        level=logging.INFO if rank == 0 else logging.WARNING,
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    model_config = ModelConfig.from_dict(config)
    
    # Setup distributed training
    if args.world_size > 1:
        setup_distributed(args)
    
    # Setup logging
    setup_logging(args.output_dir, args.rank)
    logger = logging.getLogger()
    
    # Create model
    model = DiffusionRoformer(model_config)
    device = torch.device(f'cuda:{args.local_rank}')
    model = model.to(device)
    
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    
    # Create datasets and dataloaders
    train_dataset = MusicDataset(
        data_path=config['data']['train_path'],
        prefix='train',
        vocab_size=model_config.roformer.vocab_size
    )
    
    val_dataset = MusicDataset(
        data_path=config['data']['val_path'],
        prefix='val',
        vocab_size=model_config.roformer.vocab_size
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=model_config.training.batch_size,
        num_workers=model_config.training.num_workers,
        distributed=(args.world_size > 1)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=model_config.training.batch_size,
        shuffle=False,
        num_workers=model_config.training.num_workers,
        distributed=(args.world_size > 1)
    )
    
    # Create trainer
    trainer = TrainingManager(
        model=model,
        config=model_config,
        save_dir=args.output_dir,
        distributed=(args.world_size > 1),
        local_rank=args.local_rank,
        world_size=args.world_size
    )
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Train
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
    except Exception as e:
        logger.exception(f'Training failed: {str(e)}')
        raise
    finally:
        # Cleanup
        if args.world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
