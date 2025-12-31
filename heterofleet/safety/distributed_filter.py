"""
Distributed QP-based Safety Filter for heterogeneous agents.

Implements Control Barrier Function (CBF) based safety filtering
that modifies unsafe control inputs to ensure safety constraints.

Key features:
- Quadratic Programming for minimal intervention
- Distributed computation per agent
- Platform-pair specific safety margins
- JAX-accelerated QP solving

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3, CollisionEnvelope


@dataclass
class SafetyConstraint:
    """A safety constraint for the QP."""
    
    # Constraint: A @ u <= b
    A: np.ndarray  # Constraint matrix row
    b: float       # Constraint bound
    
    # Metadata
    name: str = ""
    type: str = "barrier"  # barrier, velocity, acceleration
    agent_pair: Optional[Tuple[str, str]] = None
    
    @property
    def is_active(self) -> bool:
        """Check if constraint is active (b close to 0)."""
        return abs(self.b) < 0.1


@dataclass
class SafetyFilterResult:
    """Result of safety filter application."""
    
    # Original and filtered inputs
    original_input: np.ndarray
    filtered_input: np.ndarray
    
    # Modification info
    was_modified: bool = False
    modification_magnitude: float = 0.0
    
    # Constraint info
    num_active_constraints: int = 0
    active_constraint_names: List[str] = field(default_factory=list)
    
    # Solver info
    solver_status: str = "optimal"
    solve_time_ms: float = 0.0
    
    # Safety certificate
    is_safe: bool = True
    safety_margin: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "was_modified": self.was_modified,
            "modification_magnitude": self.modification_magnitude,
            "num_active_constraints": self.num_active_constraints,
            "active_constraints": self.active_constraint_names,
            "solver_status": self.solver_status,
            "solve_time_ms": self.solve_time_ms,
            "is_safe": self.is_safe,
            "safety_margin": self.safety_margin,
        }


class QPSolver:
    """
    Quadratic Programming solver for safety filtering.
    
    Solves: min_u (u - u_nom)^T @ H @ (u - u_nom)
            s.t. A @ u <= b
    """
    
    def __init__(self, use_jax: bool = False):
        """
        Initialize QP solver.
        
        Args:
            use_jax: Whether to use JAX acceleration
        """
        self.use_jax = use_jax
        self._jax_available = False
        
        if use_jax:
            try:
                import jax
                import jax.numpy as jnp
                self._jax_available = True
            except ImportError:
                logger.warning("JAX not available, using numpy fallback")
    
    def solve(
        self,
        u_nom: np.ndarray,
        H: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, str]:
        """
        Solve the QP problem.
        
        Args:
            u_nom: Nominal control input
            H: Hessian matrix (positive definite)
            A: Constraint matrix
            b: Constraint bounds
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (optimal_u, status)
        """
        if self._jax_available and self.use_jax:
            return self._solve_jax(u_nom, H, A, b, max_iterations)
        else:
            return self._solve_numpy(u_nom, H, A, b, max_iterations)
    
    def _solve_numpy(
        self,
        u_nom: np.ndarray,
        H: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        max_iterations: int
    ) -> Tuple[np.ndarray, str]:
        """Solve using numpy with projected gradient descent."""
        n = len(u_nom)
        u = u_nom.copy()
        
        # Check if nominal is already feasible
        if len(A) == 0 or np.all(A @ u <= b + 1e-6):
            return u, "optimal"
        
        # Projected gradient descent
        step_size = 0.1
        
        for iteration in range(max_iterations):
            # Gradient of quadratic objective
            grad = H @ (u - u_nom)
            
            # Gradient step
            u_new = u - step_size * grad
            
            # Project onto constraint set
            u_new = self._project_constraints(u_new, A, b)
            
            # Check convergence
            if np.linalg.norm(u_new - u) < 1e-6:
                return u_new, "optimal"
            
            u = u_new
        
        return u, "max_iterations"
    
    def _project_constraints(
        self,
        u: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Project u onto constraint set A @ u <= b."""
        if len(A) == 0:
            return u
        
        # Iterative projection (Dykstra's algorithm simplified)
        u_proj = u.copy()
        
        for _ in range(10):  # Inner iterations
            for i in range(len(A)):
                violation = A[i] @ u_proj - b[i]
                if violation > 0:
                    # Project onto this constraint
                    a_i = A[i]
                    a_norm_sq = np.dot(a_i, a_i)
                    if a_norm_sq > 1e-10:
                        u_proj = u_proj - (violation / a_norm_sq) * a_i
        
        return u_proj
    
    def _solve_jax(
        self,
        u_nom: np.ndarray,
        H: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        max_iterations: int
    ) -> Tuple[np.ndarray, str]:
        """Solve using JAX (if available)."""
        # JAX implementation would go here
        # For now, fall back to numpy
        return self._solve_numpy(u_nom, H, A, b, max_iterations)


class ControlBarrierFunction:
    """
    Control Barrier Function for collision avoidance.
    
    Implements CBF h(x) such that h(x) >= 0 implies safety.
    """
    
    def __init__(
        self,
        safety_radius: float = 0.5,
        gamma: float = 1.0
    ):
        """
        Initialize CBF.
        
        Args:
            safety_radius: Minimum safe distance
            gamma: CBF gain parameter
        """
        self.safety_radius = safety_radius
        self.gamma = gamma
    
    def compute_h(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        combined_radius: float
    ) -> float:
        """
        Compute barrier function value.
        
        h(x) = ||p_i - p_j||^2 - (r_i + r_j + r_safety)^2
        
        h(x) >= 0 means safe distance maintained.
        """
        diff = pos_i - pos_j
        dist_sq = np.dot(diff, diff)
        min_dist_sq = (combined_radius + self.safety_radius) ** 2
        
        return dist_sq - min_dist_sq
    
    def compute_h_dot(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        vel_i: np.ndarray,
        vel_j: np.ndarray
    ) -> float:
        """
        Compute time derivative of barrier function.
        
        h_dot = 2 * (p_i - p_j) @ (v_i - v_j)
        """
        diff = pos_i - pos_j
        vel_diff = vel_i - vel_j
        
        return 2 * np.dot(diff, vel_diff)
    
    def compute_constraint(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        vel_i: np.ndarray,
        vel_j: np.ndarray,
        combined_radius: float
    ) -> Tuple[np.ndarray, float]:
        """
        Compute CBF constraint for QP.
        
        Constraint: A @ u_i <= b
        where A = -2 * (p_i - p_j)
              b = gamma * h + 2 * (p_i - p_j) @ v_j
              
        This ensures h_dot + gamma * h >= 0 (CBF condition)
        """
        diff = pos_i - pos_j
        h = self.compute_h(pos_i, pos_j, combined_radius)
        
        # A @ u_i <= b
        A = -2 * diff
        b = self.gamma * h + 2 * np.dot(diff, vel_j)
        
        return A, b
    
    def compute_relative_constraint(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        vel_i: np.ndarray,
        vel_j: np.ndarray,
        combined_radius: float
    ) -> Tuple[np.ndarray, float]:
        """
        Compute CBF constraint considering both agents.
        
        For distributed implementation, each agent accounts for
        the other's expected behavior.
        """
        diff = pos_i - pos_j
        dist = np.linalg.norm(diff)
        
        if dist < 1e-6:
            # Very close - emergency constraint
            return np.zeros(3), -1.0
        
        unit_diff = diff / dist
        
        h = self.compute_h(pos_i, pos_j, combined_radius)
        
        # Constraint on velocity component along collision axis
        A = -unit_diff
        
        # Allow approach up to gamma * h rate
        b = self.gamma * h / (2 * dist) + np.dot(unit_diff, vel_j)
        
        return A, b


class DistributedSafetyFilter:
    """
    Distributed safety filter for heterogeneous agents.
    
    Each agent runs a local filter that ensures its control
    inputs maintain safety with respect to neighbors.
    """
    
    def __init__(
        self,
        agent_id: str,
        platform_type: PlatformType,
        collision_envelope: CollisionEnvelope,
        gamma: float = 1.0,
        safety_margin: float = 0.1,
        use_jax: bool = False
    ):
        """
        Initialize distributed safety filter.
        
        Args:
            agent_id: Agent identifier
            platform_type: Platform type
            collision_envelope: Agent's collision envelope
            gamma: CBF gain
            safety_margin: Additional safety margin
            use_jax: Whether to use JAX acceleration
        """
        self.agent_id = agent_id
        self.platform_type = platform_type
        self.collision_envelope = collision_envelope
        
        self.cbf = ControlBarrierFunction(
            safety_radius=safety_margin,
            gamma=gamma
        )
        
        self.qp_solver = QPSolver(use_jax=use_jax)
        
        # Hessian for QP (identity = minimize modification)
        self._H = np.eye(3)
        
        # Statistics
        self._stats = {
            "total_calls": 0,
            "modifications": 0,
            "avg_modification": 0.0,
            "max_modification": 0.0,
        }
    
    def filter_velocity(
        self,
        nominal_velocity: Vector3,
        agent_position: Vector3,
        agent_velocity: Vector3,
        neighbors: List[Tuple[str, Vector3, Vector3, PlatformType, CollisionEnvelope]]
    ) -> SafetyFilterResult:
        """
        Filter a nominal velocity command to ensure safety.
        
        Args:
            nominal_velocity: Desired velocity command
            agent_position: Current agent position
            agent_velocity: Current agent velocity
            neighbors: List of (id, position, velocity, type, envelope) for neighbors
            
        Returns:
            SafetyFilterResult with filtered velocity
        """
        import time
        start_time = time.time()
        
        self._stats["total_calls"] += 1
        
        # Convert to numpy
        u_nom = np.array([nominal_velocity.x, nominal_velocity.y, nominal_velocity.z])
        pos_i = np.array([agent_position.x, agent_position.y, agent_position.z])
        vel_i = np.array([agent_velocity.x, agent_velocity.y, agent_velocity.z])
        
        # Build constraints
        constraints = []
        
        for neighbor_id, neighbor_pos, neighbor_vel, neighbor_type, neighbor_envelope in neighbors:
            pos_j = np.array([neighbor_pos.x, neighbor_pos.y, neighbor_pos.z])
            vel_j = np.array([neighbor_vel.x, neighbor_vel.y, neighbor_vel.z])
            
            # Compute combined radius
            combined_radius = self._compute_combined_radius(neighbor_envelope)
            
            # Compute CBF constraint
            A, b = self.cbf.compute_relative_constraint(
                pos_i, pos_j, vel_i, vel_j, combined_radius
            )
            
            constraints.append(SafetyConstraint(
                A=A,
                b=b,
                name=f"cbf_{neighbor_id}",
                type="barrier",
                agent_pair=(self.agent_id, neighbor_id)
            ))
        
        # Check if nominal is already safe
        if not constraints or all(c.A @ u_nom <= c.b + 1e-6 for c in constraints):
            result = SafetyFilterResult(
                original_input=u_nom,
                filtered_input=u_nom,
                was_modified=False,
                is_safe=True,
                safety_margin=self._compute_min_margin(constraints, u_nom),
                solve_time_ms=(time.time() - start_time) * 1000,
            )
            return result
        
        # Build QP
        A_matrix = np.array([c.A for c in constraints])
        b_vector = np.array([c.b for c in constraints])
        
        # Solve QP
        u_filtered, status = self.qp_solver.solve(u_nom, self._H, A_matrix, b_vector)
        
        # Compute result
        modification = np.linalg.norm(u_filtered - u_nom)
        was_modified = modification > 1e-6
        
        if was_modified:
            self._stats["modifications"] += 1
            self._update_modification_stats(modification)
        
        # Find active constraints
        active = [c.name for c in constraints 
                 if abs(c.A @ u_filtered - c.b) < 1e-4]
        
        result = SafetyFilterResult(
            original_input=u_nom,
            filtered_input=u_filtered,
            was_modified=was_modified,
            modification_magnitude=modification,
            num_active_constraints=len(active),
            active_constraint_names=active,
            solver_status=status,
            solve_time_ms=(time.time() - start_time) * 1000,
            is_safe=status == "optimal",
            safety_margin=self._compute_min_margin(constraints, u_filtered),
        )
        
        return result
    
    def _compute_combined_radius(self, neighbor_envelope: CollisionEnvelope) -> float:
        """Compute combined collision radius."""
        r_self = max(self.collision_envelope.semi_axes)
        r_neighbor = max(neighbor_envelope.semi_axes)
        return r_self + r_neighbor
    
    def _compute_min_margin(
        self,
        constraints: List[SafetyConstraint],
        u: np.ndarray
    ) -> float:
        """Compute minimum constraint margin."""
        if not constraints:
            return float('inf')
        
        margins = [c.b - c.A @ u for c in constraints]
        return min(margins)
    
    def _update_modification_stats(self, modification: float) -> None:
        """Update modification statistics."""
        n = self._stats["modifications"]
        avg = self._stats["avg_modification"]
        
        self._stats["avg_modification"] = (avg * (n - 1) + modification) / n
        self._stats["max_modification"] = max(self._stats["max_modification"], modification)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_calls": 0,
            "modifications": 0,
            "avg_modification": 0.0,
            "max_modification": 0.0,
        }
    
    def get_filtered_velocity_vector(
        self,
        result: SafetyFilterResult
    ) -> Vector3:
        """Convert result to Vector3."""
        return Vector3(
            result.filtered_input[0],
            result.filtered_input[1],
            result.filtered_input[2]
        )


class CentralizedSafetyVerifier:
    """
    Centralized safety verifier for validation.
    
    Verifies that the entire fleet maintains safety constraints.
    Used for validation, not real-time control.
    """
    
    def __init__(self, safety_margin: float = 0.1):
        """Initialize centralized verifier."""
        self.safety_margin = safety_margin
    
    def verify_fleet_safety(
        self,
        positions: Dict[str, Vector3],
        velocities: Dict[str, Vector3],
        envelopes: Dict[str, CollisionEnvelope],
        dt: float = 0.1,
        horizon: float = 1.0
    ) -> Tuple[bool, List[Tuple[str, str, float]]]:
        """
        Verify that fleet is safe for prediction horizon.
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            envelopes: Agent collision envelopes
            dt: Time step for prediction
            horizon: Prediction horizon
            
        Returns:
            Tuple of (is_safe, list of (agent_i, agent_j, min_distance))
        """
        violations = []
        
        agent_ids = list(positions.keys())
        
        # Check all pairs
        for i, id_i in enumerate(agent_ids):
            for j, id_j in enumerate(agent_ids):
                if i >= j:
                    continue
                
                pos_i = positions[id_i]
                pos_j = positions[id_j]
                vel_i = velocities[id_i]
                vel_j = velocities[id_j]
                
                # Combined radius
                r_i = max(envelopes[id_i].semi_axes)
                r_j = max(envelopes[id_j].semi_axes)
                min_dist = r_i + r_j + self.safety_margin
                
                # Check over horizon
                for t in np.arange(0, horizon, dt):
                    pred_i = Vector3(
                        pos_i.x + vel_i.x * t,
                        pos_i.y + vel_i.y * t,
                        pos_i.z + vel_i.z * t
                    )
                    pred_j = Vector3(
                        pos_j.x + vel_j.x * t,
                        pos_j.y + vel_j.y * t,
                        pos_j.z + vel_j.z * t
                    )
                    
                    dist = (pred_i - pred_j).norm()
                    
                    if dist < min_dist:
                        violations.append((id_i, id_j, dist))
                        break
        
        return len(violations) == 0, violations
