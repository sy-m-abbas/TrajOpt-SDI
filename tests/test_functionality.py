"""
Functional tests for trajectory_opt_with_contact package.
Tests basic functionality without running full optimization.
"""

import torch
import numpy as np
from trajectory_opt_with_contact import (
    ContactQPSolver, 
    rot, 
    closest_point_on_square_and_normal,
    contact_frame_and_J,
    step_square,
    TrajectoryOptimizer
)

print("=" * 60)
print("Testing Package Functionality")
print("=" * 60)

# Test 1: QP Solver
print("\n[Test 1] QP Solver Initialization")
try:
    solver = ContactQPSolver(mu=0.6, n_contacts=1)
    print("   ✓ QP solver created successfully")
    print(f"   - Friction coefficient: {solver.mu}")
    print(f"   - Number of contacts: {solver.n_contacts}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Rotation matrix
print("\n[Test 2] Rotation Matrix")
try:
    theta = torch.tensor(np.pi/4, dtype=torch.double)
    R = rot(theta)
    print(f"   ✓ Rotation matrix created: shape {R.shape}")
    print(f"   - Determinant (should be 1.0): {torch.det(R).item():.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 3: Closest point on square
print("\n[Test 3] Closest Point on Square")
try:
    half = 0.1
    p_local = torch.tensor([0.15, 0.05], dtype=torch.double)
    q_local, n_local, phi = closest_point_on_square_and_normal(p_local, half)
    print(f"   ✓ Closest point computed")
    print(f"   - Signed distance: {phi.item():.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Contact Jacobian
print("\n[Test 4] Contact Jacobian")
try:
    q_body = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
    v_body = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
    pusher_pos = torch.tensor([0.15, 0.0], dtype=torch.double)
    half = 0.1
    
    J, n, t, c_world, phi = contact_frame_and_J(q_body, v_body, pusher_pos, half, mu=0.6)
    print(f"   ✓ Contact Jacobian computed")
    print(f"   - Jacobian shape: {J.shape}")
    print(f"   - Signed distance: {phi.item():.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Single time step
print("\n[Test 5] Single Time Step")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    q = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
    v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double, device=device)
    pusher_pos = torch.tensor([0.15, 0.0], dtype=torch.double, device=device)
    u_push = torch.tensor([-0.1, 0.0], dtype=torch.double, device=device)
    
    q_next, v_next, pusher_next, lam, phi = step_square(
        q, v, pusher_pos, u_push,
        h=0.05, m=1.0, Izz=0.0067, half=0.1, mu=0.6,
        qp_solver=solver, device=device
    )
    
    print(f"   ✓ Time step computed")
    print(f"   - Next position: {q_next.detach().cpu().numpy()}")
    print(f"   - Signed distance: {phi.item():.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: TrajectoryOptimizer initialization
print("\n[Test 6] TrajectoryOptimizer Initialization")
try:
    optimizer = TrajectoryOptimizer(
        mass=1.0,
        side_length=0.2,
        mu=0.6,
        horizon=10,
        dt=0.05,
        device='cpu'
    )
    print(f"   ✓ TrajectoryOptimizer created")
    print(f"   - Horizon: {optimizer.horizon}")
    print(f"   - Device: {optimizer.device}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Quick optimization (2 iterations)
print("\n[Test 7] Quick Optimization (2 iterations)")
try:
    result = optimizer.optimize(
        q0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        pusher0=[0.15, 0.0],
        goal=[0.1, 0.0, 0.0],
        max_iters=2,
        lr=0.01,
        verbose=False
    )
    print(f"   ✓ Optimization completed")
    print(f"   - Final loss: {result['loss']:.4f}")
    print(f"   - Control shape: {result['u_seq'].shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ All functionality tests passed!")
print("=" * 60)