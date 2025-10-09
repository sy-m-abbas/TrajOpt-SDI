"""
QP Solver for Contact Forces

Implements a differentiable quadratic programming layer for computing
contact forces with Coulomb friction constraints.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class ContactQPSolver:
    """
    Differentiable QP solver for contact forces with friction cone constraints.
    
    Solves: min 0.5 * λ^T G λ + b^T λ
            s.t. λ_n >= 0
                 |λ_t| <= μ * λ_n  (friction cone)
    
    Args:
        mu: Coefficient of friction
        n_contacts: Number of contact points (default: 1)
    """
    
    def __init__(self, mu=0.6, n_contacts=1):
        self.mu = mu
        self.n_contacts = n_contacts
        self.dim = 2 * n_contacts  # [λ_n, λ_t] per contact
        
        # Build the QP problem
        self._build_qp()
    
    def _build_qp(self):
        """Construct the CVXPY problem and create CvxpyLayer."""
        # Decision variable: contact forces [λ_n1, λ_t1, λ_n2, λ_t2, ...]
        lam = cp.Variable(self.dim)
        
        # Parameters (passed from PyTorch)
        Q_param = cp.Parameter((self.dim, self.dim), PSD=True)  # Cholesky factor
        b_param = cp.Parameter(self.dim)                         # Linear term
        
        # Objective: 0.5 * ||Q @ λ||^2 + b^T λ
        # (reformulated to avoid quad_form for better numerical stability)
        objective = cp.Minimize(0.5 * cp.sum_squares(Q_param @ lam) + b_param @ lam)
        
        # Constraints
        constraints = []
        
        # Non-penetration: λ_n >= 0 for all contacts
        for i in range(self.n_contacts):
            idx_n = 2 * i
            constraints.append(lam[idx_n] >= 0)
        
        # Friction cone: |λ_t| <= μ * λ_n for all contacts
        for i in range(self.n_contacts):
            idx_n = 2 * i
            idx_t = 2 * i + 1
            constraints.append(lam[idx_t] <= self.mu * lam[idx_n])
            constraints.append(lam[idx_t] >= -self.mu * lam[idx_n])
        
        # Create the problem and layer
        problem = cp.Problem(objective, constraints)
        self.qp_layer = CvxpyLayer(problem, parameters=[Q_param, b_param], variables=[lam])
    
    def solve(self, Q, b):
        """
        Solve the contact QP.
        
        Args:
            Q: Cholesky factor of the Delassus matrix (dim x dim)
            b: Linear term (dim,)
        
        Returns:
            lam_star: Optimal contact forces (dim,)
        """
        lam_star, = self.qp_layer(Q, b)
        return lam_star