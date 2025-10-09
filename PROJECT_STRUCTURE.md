# Contact Optimizer - Project Structure

## Directory Layout

```
trajectory_opt_with_contact/
├── README.md                      # Main documentation
├── LICENSE                        # License file (MIT recommended)
├── setup.py                       # Package installation script
├── requirements.txt               # Dependencies
├── PROJECT_STRUCTURE.md          # This file
│
├── trajectory_opt_with_contact/  # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── qp_solver.py             # QP solver for contact forces
│   ├── geometry.py              # Geometry utilities (rotation, closest point, etc.)
│   ├── dynamics.py              # Time-stepping dynamics
│   ├── optimizer.py             # High-level trajectory optimizer API
│   └── visualizer.py            # Visualization tools (NEW)
│
├── examples/                     # Example scripts
│   ├── simple_test.py           # Quick functionality test
│   ├── visualize_example.py     # Full visualization example 1 (NEW)
│   └── visualize_example2.py    # Full visualization example 2 (NEW)
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_functionality.py
│
└── docs/                         # Additional documentation
    ├── api_reference.md
    ├── mathematical_background.md
    └── examples.md
```

## Module Descriptions

### `contact_optimizer/__init__.py`
- Package initialization
- Exports main classes and functions
- Version information

### `contact_optimizer/qp_solver.py`
**Class: `ContactQPSolver`**
- Differentiable QP solver using CVXPyLayers
- Solves contact force optimization with friction cone constraints
- Key methods:
  - `__init__(mu, n_contacts)`: Initialize solver
  - `solve(Q, b)`: Solve QP and return optimal forces

### `contact_optimizer/geometry.py`
**Functions:**
- `rot(theta)`: 2D rotation matrix
- `closest_point_on_square_and_normal(p_local, half)`: Signed distance and normal
- `contact_frame_and_J(q_body, v_body, pusher_pos, half, mu)`: Contact Jacobian

### `contact_optimizer/dynamics.py`
**Functions:**
- `step_square(...)`: Single time-step integration with contact
- `rollout(...)`: Full trajectory simulation with cost computation

### `trajectory_opt_with_contact/visualizer.py`
**Class: `TrajectoryVisualizer`**
- Visualization utilities for optimization results
- Key methods:
  - `__init__(half_size)`: Initialize visualizer
  - `plot_trajectory(result, goal, xlim, ylim, ...)`: Plot trajectory analysis
  - `animate_trajectory(result, goal, obstacle_pos, ...)`: Create animation
  - `plot_analysis(result, ...)`: Detailed analysis plots

**Function: `visualize_result(...)`**
- Convenience function to generate all visualizations at once
- Creates trajectory plot, animation, and analysis plots

### `examples/`
**Example scripts demonstrating package usage:**
- `simple_test.py`: Quick functionality test (short horizon, fast)
- `visualize_example.py`: Complete example with visualization (Goal 1 with obstacle)
- `visualize_example2.py`: Alternative configuration (Goal 2 with different obstacle)

## Installation Instructions

### For Users

```bash
# Clone repository
git clone https://github.com/JPark-0624/trajectory_opt_with_contact.git
cd contact_optimizer

# Install package
pip install -e .

# Or install from GitHub directly
pip install git+https://github.com/JPark-0624/trajectory_opt_with_contact.git
```

### For Developers

```bash
# Clone repository
git clone https://github.com/JPark-0624/trajectory_opt_with_contact.git
cd contact_optimizer

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black contact_optimizer/ examples/ tests/

# Lint code
flake8 contact_optimizer/ examples/ tests/
```

## Usage Workflow

### 1. Basic Usage (High-level API)

```python
from contact_optimizer import TrajectoryOptimizer

optimizer = TrajectoryOptimizer(mass=1.0, side_length=0.2, mu=0.6)
result = optimizer.optimize(q0, v0, pusher0, goal)
```

### 2. Advanced Usage (Low-level API)

```python
from contact_optimizer import ContactQPSolver, step_square, rollout

# Create QP solver
qp_solver = ContactQPSolver(mu=0.6)

# Run single step
q_next, v_next, pusher_next, forces, phi = step_square(
    q, v, pusher_pos, u_push, h, m, Izz, half, mu, qp_solver=qp_solver
)

# Run full rollout
loss, q_final, lambdas, phis, qs, pusher_hist = rollout(
    u_seq, q0, v0, pr0, horizon, h, m, Izz, half, mu, goal, qp_solver=qp_solver
)
```

### 3. Custom Extensions

Users can extend the package by:
- Subclassing `TrajectoryOptimizer` for custom cost functions
- Using low-level functions to build custom dynamics
- Implementing new contact models by modifying QP constraints
- Customizing visualization with `TrajectoryVisualizer` class

## Sharing with Other Labs

### Option 1: GitHub Repository

1. Create a GitHub repository
2. Push code with proper documentation
3. Share repository URL
4. Users can install via: `pip install git+<repo_url>`

### Option 2: PyPI Package (Recommended for wide distribution)

```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
pip install twine
twine upload dist/*

# Users install via
pip install contact_optimizer
```

### Option 3: Direct Sharing (For internal use)

1. Zip the entire directory
2. Share via email/Dropbox/Google Drive
3. Recipients extract and run: `pip install -e contact_optimizer/`

## Dependencies

### Required
- Python >= 3.8
- torch >= 1.10.0
- numpy >= 1.20.0
- cvxpy >= 1.2.0
- cvxpylayers >= 0.1.5
- matplotlib >= 3.3.0

### Optional (for development)
- pytest >= 6.0 (testing)
- black >= 21.0 (code formatting)
- flake8 >= 3.8 (linting)

## API Stability

**Current version: 0.1.0 (Alpha)**

- Core API (TrajectoryOptimizer, ContactQPSolver) is stable
- Low-level functions may change in minor versions
- Breaking changes will bump major version

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run tests: `pytest tests/`
5. Format code: `black .`
6. Submit pull request

## License

MIT License - See LICENSE file for details.

## Contact

For questions or collaborations:
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/contact_optimizer/issues