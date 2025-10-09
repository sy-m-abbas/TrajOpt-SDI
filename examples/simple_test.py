"""
Simple test: Push a square to a goal position
"""

import torch
from trajectory_opt_with_contact import TrajectoryOptimizer

# Set random seed
torch.manual_seed(0)

print("Creating optimizer...")
optimizer = TrajectoryOptimizer(
    mass=1.0,
    side_length=0.2,
    mu=0.6,
    horizon=30,  
    dt=0.05,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {optimizer.device}")
print("\nOptimizing trajectory...")

result = optimizer.optimize(
    q0=[0.0, 0.0, 0.0],
    v0=[0.0, 0.0, 0.0],
    pusher0=[0.15, 0.0],
    goal=[0.2, 0.1, 0.0],
    max_iters=20,
    lr=0.01,
    verbose=True
)

print(f"\n{'='*60}")
print("Optimization Results:")
print(f"{'='*60}")
print(f"Final loss: {result['loss']:.4f}")
print(f"Final position: {result['q_final']}")
print(f"Goal position: [0.2, 0.1, 0.0]")
print(f"Position error: {result['q_final'] - [0.2, 0.1, 0.0]}")
print(f"{'='*60}")