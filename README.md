# Trajectory Optimization with Contact

Differentiable contact-implicit trajectory optimization for planar manipulation tasks using PyTorch and CVXPy.

## Features

- **Differentiable Contact Physics**: Automatic differentiation through contact constraints using CVXPyLayers
- **Coulomb Friction Model**: Realistic friction cone constraints with customizable coefficient
- **Trajectory Optimization**: Gradient-based optimization of pusher trajectories
- **GPU Acceleration**: Full CUDA support for fast computation

## Installation

### From Source

```bash
git clone https://github.com/JPark-0624/trajectory_opt_with_contact.git
cd trajectory_opt_with_contact
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CVXPY >= 1.2.0
- CVXPyLayers >= 0.1.5
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

## Quick Start

### Basic Usage

```python
from trajectory_opt_with_contact import TrajectoryOptimizer

# Create optimizer
optimizer = TrajectoryOptimizer(
    mass=1.0,           # Mass of the object
    side_length=0.2,    # Side length of square object
    mu=0.6,             # Friction coefficient
    horizon=60,         # Planning horizon (time steps)
    dt=0.05,            # Time step size
    device='cuda'       # 'cuda' or 'cpu'
)

# Optimize trajectory
result = optimizer.optimize(
    q0=[0.0, 0.0, 0.0],        # Initial pose [x, y, theta]
    v0=[0.0, 0.0, 0.0],        # Initial velocity [vx, vy, omega]
    pusher0=[0.3, -0.3],       # Initial pusher position [px, py]
    goal=[0.3, 0.0, 0.0],      # Goal pose [x, y, theta]
    max_iters=100,             # Optimization iterations
    lr=0.01                    # Learning rate
)

print("Final position:", result['q_final'])
print("Final loss:", result['loss'])
```

### With Visualization

```python
from trajectory_opt_with_contact import TrajectoryOptimizer, visualize_result

# Create and run optimizer
optimizer = TrajectoryOptimizer(horizon=60, device='cuda')
result = optimizer.optimize(
    q0=[0.0, 0.0, 0.0],
    v0=[0.0, 0.0, 0.0],
    pusher0=[0.3, -0.3],
    goal=[-0.2, 0.5, -0.3],
    max_iters=100
)

# Generate visualizations
visualize_result(
    result, 
    goal=[-0.2, 0.5, -0.3],
    half_size=optimizer.half,
    obstacle_pos=[0.2, -0.2],  # Optional obstacle
    xlim=(-1, 1),
    ylim=(-1, 1)
)
# Creates: trajectory.png, analysis.png, animation.mp4
```

### Running Examples

```bash
# Example 1: Basic pushing task
python examples/visualize_example.py

# Example 2: Different configuration
python examples/visualize_example2.py

# Quick functionality test
python examples/simple_test.py
```

## Testing

Verify the installation with provided tests:

```bash
# Test imports
python -c "from trajectory_opt_with_contact import TrajectoryOptimizer; print('✓ Import successful')"

# Run functionality tests
python test_functionality.py

# Run example
python examples/simple_test.py
```

## API Reference

### TrajectoryOptimizer

Main class for trajectory optimization.

#### Constructor

```python
TrajectoryOptimizer(
    mass=1.0,           # Mass of object (kg)
    side_length=0.2,    # Side length of square (m)
    mu=0.6,             # Friction coefficient
    horizon=60,         # Number of time steps
    dt=0.05,            # Time step size (s)
    device='cuda'       # 'cuda' or 'cpu'
)
```

#### optimize() Method

```python
result = optimizer.optimize(
    q0,                 # Initial configuration [x, y, theta]
    v0,                 # Initial velocity [vx, vy, omega]
    pusher0,            # Initial pusher position [px, py]
    goal,               # Goal configuration [x, y, theta]
    u_init=None,        # Initial control guess (optional)
    max_iters=100,      # Maximum optimization iterations
    lr=0.01,            # Learning rate
    lr_decay_step=10,   # Learning rate decay step
    lr_decay_gamma=0.5, # Learning rate decay factor
    obstacle_pos=None,  # Obstacle position [x, y] (optional)
    verbose=True        # Print progress
)
```

#### Return Value

Returns a dictionary with:
- `'loss'`: Final loss value
- `'q_final'`: Final configuration (3,)
- `'u_seq'`: Optimized control sequence (T, 2)
- `'trajectory'`: Full state trajectory (T+1, 3)
- `'pusher_trajectory'`: Pusher trajectory (T+1, 2)
- `'contact_forces'`: Contact force history (T, 2)
- `'signed_distances'`: Signed distance history (T,)

### TrajectoryVisualizer

Visualization utilities for results.

```python
from trajectory_opt_with_contact import TrajectoryVisualizer

viz = TrajectoryVisualizer(half_size=0.1)

# Plot trajectory analysis
viz.plot_trajectory(result, goal, save_path='trajectory.png',
                   xlim=(-1, 1), ylim=(-1, 1))

# Create animation
viz.animate_trajectory(result, goal, save_path='animation.mp4',
                      fps=60, obstacle_pos=[0.2, -0.2])

# Detailed analysis plots
viz.plot_analysis(result, save_path='analysis.png')
```

### Convenience Function

```python
from trajectory_opt_with_contact import visualize_result

# Generate all visualizations at once
visualize_result(
    result, 
    goal, 
    half_size=0.1,
    save_trajectory='trajectory.png',
    save_animation='animation.mp4',
    save_analysis='analysis.png',
    obstacle_pos=[0.2, -0.2],
    xlim=(-1, 1),
    ylim=(-1, 1)
)
```

### Low-Level API

For custom implementations:

```python
from trajectory_opt_with_contact import (
    ContactQPSolver,
    step_square,
    rollout,
    contact_frame_and_J
)

# Create QP solver
qp_solver = ContactQPSolver(mu=0.6, n_contacts=1)

# Single time step
q_next, v_next, pusher_next, forces, phi = step_square(
    q, v, pusher_pos, u_push,
    h=0.05, m=1.0, Izz=0.0067, half=0.1, mu=0.6,
    qp_solver=qp_solver
)

# Full trajectory rollout
loss, q_final, lambdas, phis, qs, pusher_hist = rollout(
    u_seq, q0, v0, pr0, horizon, h, m, Izz, half, mu, goal,
    qp_solver=qp_solver
)
```

## Mathematical Background

The optimizer uses contact-implicit trajectory optimization:

1. **Contact QP** - At each time step, contact forces are computed by solving:
   ```
   minimize    0.5 * λ^T G λ + b^T λ
   subject to  λ_n >= 0
               |λ_t| <= μ * λ_n  (friction cone)
   ```

2. **Time Integration** - Semi-implicit Euler with contact impulses:
   ```
   v_{k+1} = v_k + M^{-1} J^T λ_k
   q_{k+1} = q_k + h * v_{k+1}
   ```

3. **Trajectory Optimization** - Gradient descent on control sequence:
   ```
   minimize  Σ ||q_T - q_goal||² + α||u||² + ...
   ```

Gradients flow through the QP solution via CVXPyLayers for end-to-end differentiability.

## Project Structure

```
trajectory_opt_with_contact/
├── trajectory_opt_with_contact/    # Main package
│   ├── __init__.py                 # Package interface
│   ├── qp_solver.py                # Differentiable QP solver
│   ├── geometry.py                 # Geometry utilities
│   ├── dynamics.py                 # Physics simulation
│   ├── optimizer.py                # High-level optimizer
│   └── visualizer.py               # Visualization tools
├── examples/                       # Usage examples
│   ├── simple_test.py              # Quick test
│   ├── visualize_example.py        # Full visualization example 1
│   └── visualize_example2.py       # Full visualization example 2
├── tests/                          # Test files
│   └── test_functionality.py       # Functionality tests
├── setup.py                        # Installation script
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── LICENSE.md                      # MIT License
```

## Troubleshooting

### CUDA Out of Memory
```python
# Use CPU instead
optimizer = TrajectoryOptimizer(..., device='cpu')

# Or reduce horizon
optimizer = TrajectoryOptimizer(..., horizon=30)
```

### CVXPy/CVXPyLayers Issues
```bash
pip install --upgrade cvxpy cvxpylayers
# Or specific versions
pip install cvxpy==1.3.0 cvxpylayers==0.1.6
```

### Import Errors
```bash
# Reinstall in editable mode
pip uninstall trajectory_opt_with_contact
cd trajectory_opt_with_contact
pip install -e .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{trajectory_opt_with_contact,
  title = {Trajectory Optimization with Contact-Implicit Dynamics},
  author = {June Il Park},
  year = {2025},
  url = {https://github.com/JPark-0624/trajectory_opt_with_contact.git}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: parkjuneil94@gmail.com / jpark470@uic.edu

## Acknowledgments

This package uses:
- [PyTorch](https://pytorch.org/) for automatic differentiation
- [CVXPY](https://www.cvxpy.org/) for convex optimization
- [CVXPyLayers](https://github.com/cvxgrp/cvxpylayers) for differentiable optimization layers