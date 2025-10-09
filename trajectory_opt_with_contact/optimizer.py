"""
High-level trajectory optimizer interface.

Provides a convenient class-based API for setting up and solving
trajectory optimization problems.
"""

import torch
from .dynamics import rollout
from .qp_solver import ContactQPSolver


class TrajectoryOptimizer:
    """
    Trajectory optimizer for contact-implicit manipulation.
    
    Example:
        >>> opt = TrajectoryOptimizer(
        ...     mass=1.0,
        ...     side_length=0.2,
        ...     mu=0.6,
        ...     horizon=60,
        ...     dt=0.05
        ... )
        >>> result = opt.optimize(
        ...     q0=[0, 0, 0],
        ...     v0=[0, 0, 0],
        ...     pusher0=[0.3, -0.3],
        ...     goal=[0.3, 0.0, 0.0],
        ...     max_iters=100,
        ...     lr=0.01
        ... )
    """
    
    def __init__(self, mass=1.0, side_length=0.2, mu=0.6, 
                 horizon=60, dt=0.05, device='cuda'):
        """
        Initialize trajectory optimizer.
        
        Args:
            mass: Mass of the object
            side_length: Side length of the square
            mu: Coefficient of friction
            horizon: Number of time steps
            dt: Time step size
            device: 'cuda' or 'cpu'
        """
        self.m = mass
        self.side = side_length
        self.half = side_length / 2
        self.mu = mu
        self.horizon = horizon
        self.dt = dt
        
        # Moment of inertia for square plate
        self.Izz = (1.0/6.0) * self.m * (self.side**2 + self.side**2)
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)
        
        # Create QP solver
        self.qp_solver = ContactQPSolver(mu=self.mu, n_contacts=1)
    
    def optimize(self, q0, v0, pusher0, goal, 
                 u_init=None, max_iters=100, lr=0.01,
                 lr_decay_step=10, lr_decay_gamma=0.5,
                 obstacle_pos=None, verbose=True):
        """
        Optimize a pushing trajectory.
        
        Args:
            q0: Initial configuration [x, y, theta] (list or tensor)
            v0: Initial velocity [vx, vy, omega] (list or tensor)
            pusher0: Initial pusher position [px, py] (list or tensor)
            goal: Goal configuration [x, y, theta] (list or tensor)
            u_init: Initial control guess (T, 2) or None
            max_iters: Maximum optimization iterations
            lr: Learning rate
            lr_decay_step: Step size for learning rate decay
            lr_decay_gamma: Decay factor for learning rate
            obstacle_pos: Optional obstacle position [x, y]
            verbose: Print progress
        
        Returns:
            dict with keys:
                - 'loss': Final loss value
                - 'q_final': Final configuration
                - 'u_seq': Optimized control sequence
                - 'trajectory': Full state trajectory
                - 'pusher_trajectory': Full pusher trajectory
                - 'contact_forces': Contact force history
                - 'signed_distances': Signed distance history
        """
        # Convert inputs to tensors
        q0 = self._to_tensor(q0, requires_grad=False)
        v0 = self._to_tensor(v0, requires_grad=False)
        pusher0 = self._to_tensor(pusher0, requires_grad=False)
        goal = self._to_tensor(goal, requires_grad=False)
        
        if obstacle_pos is not None:
            obstacle_pos = self._to_tensor(obstacle_pos, requires_grad=False)
        
        # Initialize control sequence
        if u_init is None:
            u_seq = torch.zeros(self.horizon, 2, dtype=torch.double, 
                              device=self.device, requires_grad=True)
        else:
            u_seq = self._to_tensor(u_init, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([u_seq], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )
        
        # Optimization loop
        for it in range(max_iters):
            optimizer.zero_grad()
            
            loss, q_final, lambdas, phis, qs, pusher_traj = rollout(
                u_seq, q0, v0, pusher0, self.horizon, self.dt,
                self.m, self.Izz, self.half, self.mu, goal,
                qp_solver=self.qp_solver, obstacle_pos=obstacle_pos,
                device=self.device
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and (it + 1) % 10 == 0:
                print(f"Iter {it+1:3d} | Loss={loss.item():.4f} | "
                      f"Final x={q_final[0].item():.3f}")
        
        # Return results
        return {
            'loss': loss.item(),
            'q_final': q_final.detach().cpu().numpy(),
            'u_seq': u_seq.detach().cpu().numpy(),
            'trajectory': qs.detach().cpu().numpy(),
            'pusher_trajectory': pusher_traj.detach().cpu().numpy(),
            'contact_forces': lambdas.detach().cpu().numpy(),
            'signed_distances': phis.detach().cpu().numpy()
        }
    
    def _to_tensor(self, x, requires_grad=False):
        """Convert input to tensor on correct device."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double, device=self.device)
        else:
            x = x.to(dtype=torch.double, device=self.device)
        x.requires_grad = requires_grad
        return x