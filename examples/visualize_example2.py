"""
Example: Trajectory optimization with full visualization
Matches the original main.py visualization style
"""

import torch
from trajectory_opt_with_contact import TrajectoryOptimizer, visualize_result

# Set random seed
torch.manual_seed(0)

print("="*60)
print("Trajectory Optimization with Visualization")
print("="*60)

# Create optimizer (matching original parameters)
print("\nCreating optimizer...")
optimizer = TrajectoryOptimizer(
    mass=1.0,
    side_length=0.2,
    mu=0.6,
    horizon=60,
    dt=0.05,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {optimizer.device}")

# Define problem (matching original main.py)
q0 = [0.0, 0.0, 0.0]
v0 = [0.0, 0.0, 0.0]
pusher0 = [-0.3, 0.0]
goal = [0.3, 0.1, 0.3]
obstacle = [-0.1, 0.0]  # Obstacle position from original code
u_init = [[0.2, 0.0]] * optimizer.horizon  # Initial guess from original code

print(f"\nInitial pose: {q0}")
print(f"Goal pose: {goal}")
print(f"Obstacle at: {obstacle}")

# Optimize
print("\nOptimizing trajectory...")
result = optimizer.optimize(
    q0=q0,
    v0=v0,
    pusher0=pusher0,
    goal=goal,
    u_init=u_init,
    max_iters=100,
    lr=0.01,
    lr_decay_step=10,
    lr_decay_gamma=0.5,
    obstacle_pos=obstacle,
    verbose=True
)

print(f"\n{'='*60}")
print("Optimization Complete!")
print(f"{'='*60}")
print(f"Final loss: {result['loss']:.4f}")
print(f"Final pose [x,y,theta]: {result['q_final']}")
print(f"Position error: {result['q_final'] - goal}")
print(f"{'='*60}")

# Visualize (with original style parameters)
visualize_result(
    result, 
    goal, 
    half_size=optimizer.half,
    save_trajectory='trajectory.png',
    save_animation='animation.mp4',
    save_analysis='analysis.png',
    obstacle_pos=obstacle,
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0)
)

print("\n✓ Done! Check the generated files:")
print("  - trajectory2.png")
print("  - analysis2.png")
print("  - animation2.mp4")
print("\nContact forces summary:")
print(f"  Mean normal impulse: {result['contact_forces'][:, 0].mean():.4f}")
print(f"  Mean tangential impulse: {result['contact_forces'][:, 1].mean():.4f}")