# Examples Guide

This document describes the example scripts included in the package.

## Available Examples

### 1. simple_test.py - Quick Functionality Test

**Purpose:** Fast test to verify the package works correctly.

**Features:**
- Short horizon (30 steps)
- Few iterations (20)
- Simple goal configuration
- Runs in ~30 seconds

**Usage:**
```bash
python examples/simple_test.py
```

**Output:**
- Console output showing optimization progress
- Final loss and position error

---

### 2. visualize_example.py - Full Visualization (Scenario 1)

**Purpose:** Complete trajectory optimization with full visualization capabilities.

**Configuration:**
- Initial pose: `[0.0, 0.0, 0.0]`
- Initial pusher: `[0.3, -0.3]`
- Goal pose: `[-0.2, 0.5, -0.3]`
- Obstacle at: `[0.2, -0.2]`
- Horizon: 60 steps
- Iterations: 100

**Usage:**
```bash
python examples/visualize_example.py
```

**Output Files:**
- `trajectory.png` - 4-subplot trajectory analysis
  - XY trajectory with start/goal markers
  - Contact forces over time
  - Configuration (x, y, θ) over time
  - Control inputs over time
- `analysis.png` - 6-subplot detailed analysis
  - Position error
  - Force magnitude
  - Signed distance (contact detection)
  - Control magnitude
  - Friction cone constraint check
  - Object velocity
- `animation.mp4` - 60fps animation with:
  - Moving square and pusher
  - Trajectory trails
  - Goal indicator
  - Obstacle visualization

**Running Time:** ~3-5 minutes (depending on hardware)

---

### 3. visualize_example2.py - Full Visualization (Scenario 2)

**Purpose:** Alternative configuration to demonstrate different pushing strategies.

**Configuration:**
- Initial pose: `[0.0, 0.0, 0.0]`
- Initial pusher: `[-0.3, 0.0]` (different side)
- Goal pose: `[0.3, 0.1, 0.3]` (different orientation)
- Obstacle at: `[-0.1, 0.0]` (different location)
- Horizon: 60 steps
- Iterations: 100

**Usage:**
```bash
python examples/visualize_example2.py
```

**Output Files:**
Same as `visualize_example.py` but with different trajectory results.

**Running Time:** ~3-5 minutes (depending on hardware)

---

## Customizing Examples

### Change Goal Configuration

```python
# Edit the example file
goal = [0.5, 0.3, 0.0]  # Your desired [x, y, theta]
```

### Adjust Optimization Parameters

```python
result = optimizer.optimize(
    ...
    max_iters=50,      # Fewer iterations = faster
    lr=0.02,           # Higher learning rate
    ...
)
```

### Modify Visualization Settings

```python
visualize_result(
    result, 
    goal,
    xlim=(-0.5, 0.5),  # Zoom in
    ylim=(-0.5, 0.5),
    obstacle_pos=None   # Remove obstacle
)
```

### Change Horizon

```python
optimizer = TrajectoryOptimizer(
    horizon=30,  # Shorter = faster but less planning
    ...
)
```

---

## Creating Your Own Example

```python
import torch
from trajectory_opt_with_contact import TrajectoryOptimizer, visualize_result

# Setup
torch.manual_seed(42)  # For reproducibility
optimizer = TrajectoryOptimizer(
    mass=1.0,
    side_length=0.2,
    mu=0.6,
    horizon=60,
    dt=0.05,
    device='cuda'
)

# Define your scenario
q0 = [0.0, 0.0, 0.0]          # Start position
v0 = [0.0, 0.0, 0.0]          # Start velocity
pusher0 = [0.2, 0.0]          # Pusher start
goal = [0.4, 0.2, 0.5]        # Goal configuration
obstacle = [0.2, 0.1]         # Optional obstacle

# Optimize
result = optimizer.optimize(
    q0=q0, v0=v0, pusher0=pusher0, goal=goal,
    max_iters=100, lr=0.01,
    obstacle_pos=obstacle
)

# Visualize
visualize_result(result, goal, half_size=0.1, obstacle_pos=obstacle)

# Analyze results
print(f"Final position: {result['q_final']}")
print(f"Position error: {result['q_final'] - goal}")
print(f"Final loss: {result['loss']:.4f}")
```

---

## Tips for Good Results

1. **Start Position:** Place pusher close enough to make contact (within 0.3m)
2. **Initial Guess:** Provide `u_init` pointing roughly toward the goal
3. **Learning Rate:** Start with 0.01, increase if convergence is slow
4. **Iterations:** 50-100 usually sufficient for good results
5. **Horizon:** 40-60 steps works well for most tasks
6. **Obstacle:** Place obstacles at least 0.05m away from trajectory

---

## Troubleshooting

### Poor Convergence
- Increase iterations: `max_iters=200`
- Decrease learning rate: `lr=0.005`
- Provide better initial guess: `u_init=...`

### Animation Not Saved
- Install ffmpeg: `conda install ffmpeg`
- Or use just the plots without animation

### Out of Memory
- Reduce horizon: `horizon=30`
- Use CPU: `device='cpu'`
- Close other GPU applications

### Collision with Obstacle
- Adjust obstacle position
- Change initial pusher position
- Modify cost function weights in `optimizer.py`

---

## Batch Processing Multiple Scenarios

```python
scenarios = [
    {'pusher0': [0.3, -0.3], 'goal': [-0.2, 0.5, -0.3]},
    {'pusher0': [-0.3, 0.0], 'goal': [0.3, 0.1, 0.3]},
    {'pusher0': [0.0, 0.3], 'goal': [0.0, -0.3, 0.0]},
]

for i, scenario in enumerate(scenarios):
    result = optimizer.optimize(
        q0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        **scenario,
        max_iters=100
    )
    visualize_result(
        result, 
        scenario['goal'],
        save_animation=f'animation_{i}.mp4'
    )
```
