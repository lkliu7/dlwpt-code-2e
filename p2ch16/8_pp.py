#!/usr/bin/env python3
"""
Torch Distributed Pipeline Parallelism Example with Gloo Backend

The example splits a simple MLP model across multiple processes and uses
pipeline parallelism schedules like GPipe and 1F1B to train the model.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import (
    pipeline,
    ScheduleGPipe,
    Schedule1F1B,
    SplitPoint,
)


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for demonstration."""
    def __init__(self, hidden_size=512, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)
        ])
        # Define split_spec to split the model based on layer FQN
        # Split at the BEGINNING of each layer (except layer 0)
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def setup_distributed(backend='gloo'):
    """Initialize distributed process group."""
    # Check if running in a distributed environment
    if 'RANK' not in os.environ:
        print("Error: This script requires a distributed environment.")
        print("Run with: torchrun --nproc_per_node=4 pp_example.py")
        exit(1)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group with gloo backend
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    return rank, world_size


def run_pipeline_example(rank, world_size, schedule_type='gpipe'):
    """Run the pipeline parallelism example."""
    # Model configuration
    hidden_size = 512
    n_layers = world_size  # One layer per rank
    batch_size = 64
    num_microbatches = 4
    
    # Create the model
    device = torch.device('cpu')
    model = SimpleMLP(hidden_size, n_layers).to(device)
    
    # Create sample input data
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, device=device)
    target = torch.randn(batch_size, hidden_size, device=device)
    
    # Split model using the pipeline API with split_spec
    x_mb = x.chunk(num_microbatches)[0]
    pipe = pipeline(model, mb_args=(x_mb,), split_spec=model.split_spec)
    
    # Build the pipeline stage for this rank
    stage = pipe.build_stage(rank, device)
    
    # Create loss function
    loss_fn = nn.MSELoss(reduction='mean')
    
    # Create the schedule
    if schedule_type == 'gpipe':
        schedule = ScheduleGPipe(stage, num_microbatches, loss_fn=loss_fn)
    elif schedule_type == '1f1b':
        schedule = Schedule1F1B(stage, num_microbatches, loss_fn=loss_fn)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    # Training loop
    print(f"[Rank {rank}] Starting training with {schedule_type} schedule...")
    
    for iteration in range(5):
        losses = []
        
        # Execute pipeline stage
        if rank == 0:
            # First rank receives input
            schedule.step(x)
        elif rank == world_size - 1:
            # Last rank computes loss
            output = schedule.step(target=target, losses=losses)
        else:
            # Middle ranks just forward/backward
            schedule.step()
        
        # Print progress from last rank
        if rank == world_size - 1 and losses:
            avg_loss = sum(losses) / len(losses)
            print(f"[Rank {rank}] Iteration {iteration}: Loss = {avg_loss:.4f}")
    
    # Synchronize all ranks
    dist.barrier()
    print(f"[Rank {rank}] Training completed!")


def run_manual_stage_example(rank, world_size):
    """Run pipeline example with manually created stages (no tracer)."""
    from torch.distributed.pipelining import PipelineStage
    
    hidden_size = 512
    n_layers = world_size
    batch_size = 64
    num_microbatches = 4
    
    device = torch.device('cpu')
    full_model = SimpleMLP(hidden_size, n_layers).to(device)
    
    # Manually extract the submodule for this rank
    submod_name = f"layers.{rank}"
    stage_module = full_model.get_submodule(submod_name)
    
    # Create pipeline stage manually
    stage = PipelineStage(stage_module, rank, world_size, device)
    
    # Create sample data
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, device=device)
    target = torch.randn(batch_size, hidden_size, device=device)
    
    loss_fn = nn.MSELoss(reduction='sum')
    schedule = ScheduleGPipe(stage, num_microbatches, loss_fn=loss_fn)
    
    print(f"[Rank {rank}] Running manual stage example...")
    
    for iteration in range(3):
        losses = []
        
        if rank == 0:
            schedule.step(x)
        elif rank == world_size - 1:
            output = schedule.step(target=target, losses=losses)
        else:
            schedule.step()
        
        if rank == world_size - 1 and losses:
            avg_loss = sum(losses) / len(losses)
            print(f"[Rank {rank}] Iteration {iteration}: Loss = {avg_loss:.4f}")
    
    dist.barrier()
    print(f"[Rank {rank}] Manual stage example completed!")


def main():
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Example with Gloo')
    parser.add_argument('--backend', type=str, default='gloo', 
                       choices=['gloo', 'nccl'],
                       help='Distributed backend to use')
    parser.add_argument('--schedule', type=str, default='gpipe',
                       choices=['gpipe', '1f1b'],
                       help='Pipeline schedule to use')
    parser.add_argument('--manual', action='store_true',
                       help='Use manual stage creation instead of tracer')
    
    args = parser.parse_args()
    
    # Setup distributed environment
    rank, world_size = setup_distributed(args.backend)
    
    print(f"[Rank {rank}/{world_size}] Initialized with {args.backend} backend")
    
    try:
        if args.manual:
            run_manual_stage_example(rank, world_size)
        else:
            run_pipeline_example(rank, world_size, args.schedule)
    finally:
        # Cleanup
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
