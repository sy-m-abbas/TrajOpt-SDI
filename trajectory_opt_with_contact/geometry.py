"""
Geometry utilities for contact detection and Jacobian computation.

Provides functions for:
- 2D rotation matrices
- Signed distance and closest points for squares
- Contact Jacobians for rigid bodies
"""

import torch


def rot(theta):
    """
    Create a 2D rotation matrix.
    
    Args:
        theta: Rotation angle (radians)
    
    Returns:
        R: 2x2 rotation matrix
    """
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    return R


def closest_point_on_square_and_normal(p_local, half):
    """
    Find the closest point on a square boundary and its outward normal.
    
    Args:
        p_local: Query point in body frame (2,)
        half: Half side length of the square
    
    Returns:
        q_local: Closest point on boundary in body frame (2,)
        n_local: Outward unit normal at boundary point (2,)
        phi: Signed distance (positive outside, negative inside)
    """
    # Signed distance for rectangle (square)
    ax = torch.abs(p_local[0]) - half
    ay = torch.abs(p_local[1]) - half
    d = torch.stack([ax, ay])
    outside = torch.clamp(d, min=0.0)
    outside_len = torch.linalg.norm(outside)
    inside = torch.clamp(torch.max(d), max=0.0)
    phi = outside_len + inside
    
    # Closest point - use functional operations instead of in-place
    q_local_x = torch.clamp(p_local[0], -half, half)
    q_local_y = torch.clamp(p_local[1], -half, half)
    q_local = torch.stack([q_local_x, q_local_y])
    
    # Determine outward normal
    eps = 1e-9
    if outside_len > eps:
        n_local = (p_local - q_local) / (outside_len + 1e-12)
    else:
        dx = half - torch.abs(p_local[0])
        dy = half - torch.abs(p_local[1])
        if dx < dy:
            n_local = torch.tensor([torch.sign(p_local[0]), 0.0], 
                                  dtype=p_local.dtype, device=p_local.device)
        else:
            n_local = torch.tensor([0.0, torch.sign(p_local[1])], 
                                  dtype=p_local.dtype, device=p_local.device)
    
    # Re-project q_local to boundary when inside
    if (phi < 0.0):
        if torch.abs(n_local[0]) > 0.5:
            q_local = torch.stack([half * torch.sign(p_local[0]), q_local[1]])
        else:
            q_local = torch.stack([q_local[0], half * torch.sign(p_local[1])])
    
    # Normalize n_local
    n_local = n_local / (torch.linalg.norm(n_local) + 1e-12)
    return q_local, n_local, phi


def contact_frame_and_J(q_body, v_body, pusher_pos, half, mu):
    """
    Compute contact frame and Jacobian for a point pusher and square.
    
    Args:
        q_body: Body configuration (x, y, theta)
        v_body: Body velocity (vx, vy, omega)
        pusher_pos: Pusher position in world frame (px, py)
        half: Half side length of the square
        mu: Coefficient of friction (unused, kept for API compatibility)
    
    Returns:
        J: Contact Jacobian (2x3) mapping body velocity to contact velocity
        n: Contact normal in world frame (2,)
        t: Contact tangent in world frame (2,)
        c_world: Contact point in world frame (2,)
        phi: Signed distance (positive = separated, negative = penetrating)
    """
    x, y, th = q_body
    R = rot(th)
    center = torch.stack([x, y])
    
    # Transform pusher position to body frame
    p_local = R.T @ (pusher_pos - center)
    
    # Find closest point and normal on square
    q_local, n_local, phi = closest_point_on_square_and_normal(p_local, half)
    
    # Transform back to world frame
    c_world = center + R @ q_local
    n = R @ n_local
    n = -n  # Flip normal to point into the body
    n = n / (torch.linalg.norm(n) + 1e-12)
    t = torch.stack([-n[1], n[0]])  # Tangent (perpendicular to normal)
    
    # Compute Jacobian: maps body velocity [vx, vy, omega] to contact velocity [v_n, v_t]
    r = c_world - center  # Vector from COM to contact point
    
    # Cross product terms for angular velocity contribution
    s_n_omega = -n[0] * r[1] + n[1] * r[0]
    s_t_omega = -t[0] * r[1] + t[1] * r[0]
    
    J = torch.stack([
        torch.stack([n[0], n[1], s_n_omega]),  # Normal direction
        torch.stack([t[0], t[1], s_t_omega]),  # Tangent direction
    ])
    
    return J, n, t, c_world, phi