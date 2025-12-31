"""
Cross-platform friction calculations for heterogeneous agents.

Implements velocity alignment/friction forces that help agents
coordinate their motion when in proximity.

Key features:
- Platform-specific friction distances
- Adaptive friction based on relative velocity
- Conservative deceleration limits

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3, CollisionEnvelope


@dataclass
class FrictionParameters:
    """
    Parameters for friction/velocity alignment calculations.
    
    Friction helps nearby agents align their velocities to prevent
    collisions and enable smooth coordination.
    """
    
    # Friction distance multiplier (relative to avoidance radius)
    # R_friction = friction_distance_factor * R_avoid
    friction_distance_factor: float = 1.5
    
    # Maximum friction force magnitude
    max_friction_magnitude: float = 2.0
    
    # Friction gain
    friction_gain: float = 1.0
    
    # Minimum velocity difference to apply friction (m/s)
    min_velocity_diff: float = 0.05
    
    # Conservative deceleration limit (m/s²)
    # Agents will not decelerate faster than this due to friction
    max_deceleration: float = 3.0
    
    # Platform-specific friction multipliers
    platform_friction_multipliers: Dict[PlatformType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default platform friction multipliers."""
        if not self.platform_friction_multipliers:
            self.platform_friction_multipliers = {
                # Smaller/more agile platforms have lower friction (can adjust faster)
                PlatformType.MICRO_UAV: 0.8,
                PlatformType.SMALL_UAV: 0.9,
                PlatformType.MEDIUM_UAV: 1.0,
                PlatformType.LARGE_UAV: 1.2,
                # Ground vehicles have higher friction (less agile)
                PlatformType.SMALL_UGV: 1.1,
                PlatformType.MEDIUM_UGV: 1.3,
                PlatformType.LARGE_UGV: 1.5,
                # Surface vehicles
                PlatformType.SMALL_USV: 1.2,
                PlatformType.MEDIUM_USV: 1.4,
            }
    
    def get_friction_multiplier(self, platform_type: PlatformType) -> float:
        """Get friction multiplier for a platform type."""
        return self.platform_friction_multipliers.get(platform_type, 1.0)


class FrictionCalculator:
    """
    Calculator for cross-platform friction forces.
    
    Friction encourages velocity alignment between nearby agents
    to prevent collisions and enable coordinated motion.
    
    The friction force is:
    v_friction_i = Σ_j F[τ_i][τ_j] · f(d_ij, R_friction) · (v_j - v_i)
    
    where:
    - F[τ_i][τ_j] is the platform-pair friction factor
    - f(d, R) is a distance-dependent weighting function
    - (v_j - v_i) is the velocity difference
    """
    
    def __init__(self, params: FrictionParameters = None):
        """
        Initialize friction calculator.
        
        Args:
            params: Friction parameters
        """
        self.params = params or FrictionParameters()
        
        # Platform-pair friction matrix
        self._friction_matrix: Dict[Tuple[PlatformType, PlatformType], float] = {}
        self._initialize_friction_matrix()
    
    def _initialize_friction_matrix(self) -> None:
        """Initialize platform-pair friction factors."""
        # Same-type pairs have highest friction (want to stay aligned)
        for ptype in PlatformType:
            self._friction_matrix[(ptype, ptype)] = 1.0
        
        # Cross-type pairs
        uav_types = [
            PlatformType.MICRO_UAV,
            PlatformType.SMALL_UAV,
            PlatformType.MEDIUM_UAV,
            PlatformType.LARGE_UAV,
        ]
        
        ugv_types = [
            PlatformType.SMALL_UGV,
            PlatformType.MEDIUM_UGV,
            PlatformType.LARGE_UGV,
        ]
        
        # UAV-UAV cross-type
        for i, type_i in enumerate(uav_types):
            for j, type_j in enumerate(uav_types):
                if i != j:
                    # Similar sizes have higher friction
                    size_diff = abs(i - j)
                    self._friction_matrix[(type_i, type_j)] = 1.0 / (1.0 + 0.2 * size_diff)
        
        # UGV-UGV cross-type
        for i, type_i in enumerate(ugv_types):
            for j, type_j in enumerate(ugv_types):
                if i != j:
                    size_diff = abs(i - j)
                    self._friction_matrix[(type_i, type_j)] = 1.0 / (1.0 + 0.3 * size_diff)
        
        # UAV-UGV cross-domain (lower friction - different dynamics)
        for uav_type in uav_types:
            for ugv_type in ugv_types:
                self._friction_matrix[(uav_type, ugv_type)] = 0.3
                self._friction_matrix[(ugv_type, uav_type)] = 0.3
        
        # USV interactions (minimal friction with other domains)
        usv_types = [PlatformType.SMALL_USV, PlatformType.MEDIUM_USV]
        for usv_type in usv_types:
            for other_type in PlatformType:
                if other_type not in usv_types:
                    self._friction_matrix[(usv_type, other_type)] = 0.1
                    self._friction_matrix[(other_type, usv_type)] = 0.1
    
    def get_friction_factor(
        self,
        type_i: PlatformType,
        type_j: PlatformType
    ) -> float:
        """Get friction factor for a platform pair."""
        return self._friction_matrix.get((type_i, type_j), 0.5)
    
    def compute_friction_radius(
        self,
        avoidance_radius: float
    ) -> float:
        """
        Compute friction activation radius.
        
        Friction is active within a larger radius than avoidance
        to provide smoother coordination.
        
        Args:
            avoidance_radius: Combined avoidance radius R₀(i,j)
            
        Returns:
            Friction radius R_friction
        """
        return self.params.friction_distance_factor * avoidance_radius
    
    def compute_distance_weight(
        self,
        distance: float,
        friction_radius: float,
        avoidance_radius: float
    ) -> float:
        """
        Compute distance-dependent friction weight.
        
        Weight is 0 outside friction_radius, increases linearly
        as distance decreases, and is capped at avoidance_radius.
        
        Args:
            distance: Distance between agents
            friction_radius: Friction activation radius
            avoidance_radius: Avoidance radius (full friction)
            
        Returns:
            Weight in [0, 1]
        """
        if distance >= friction_radius:
            return 0.0
        
        if distance <= avoidance_radius:
            return 1.0
        
        # Linear interpolation
        return (friction_radius - distance) / (friction_radius - avoidance_radius)
    
    def compute_friction_force(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        neighbor_type: PlatformType,
        avoidance_radius: float,
        dt: float = 0.1
    ) -> Vector3:
        """
        Compute friction force from a single neighbor.
        
        Args:
            agent_pos: Position of computing agent
            agent_vel: Velocity of computing agent
            agent_type: Platform type of computing agent
            neighbor_pos: Position of neighbor
            neighbor_vel: Velocity of neighbor
            neighbor_type: Platform type of neighbor
            avoidance_radius: Combined avoidance radius R₀(i,j)
            dt: Time step for deceleration limiting
            
        Returns:
            Friction velocity adjustment
        """
        # Compute distance
        diff = neighbor_pos - agent_pos
        distance = diff.norm()
        
        if distance < 1e-6:
            return Vector3(0, 0, 0)
        
        # Friction radius
        friction_radius = self.compute_friction_radius(avoidance_radius)
        
        # Distance weight
        weight = self.compute_distance_weight(distance, friction_radius, avoidance_radius)
        
        if weight < 1e-6:
            return Vector3(0, 0, 0)
        
        # Velocity difference
        vel_diff = neighbor_vel - agent_vel
        vel_diff_magnitude = vel_diff.norm()
        
        if vel_diff_magnitude < self.params.min_velocity_diff:
            return Vector3(0, 0, 0)
        
        # Platform-pair friction factor
        F_ij = self.get_friction_factor(agent_type, neighbor_type)
        
        # Platform-specific multipliers
        mult_i = self.params.get_friction_multiplier(agent_type)
        mult_j = self.params.get_friction_multiplier(neighbor_type)
        
        # Combined friction factor
        combined_factor = F_ij * mult_i * mult_j
        
        # Friction force
        friction = vel_diff * (self.params.friction_gain * weight * combined_factor)
        
        # Limit by maximum friction magnitude
        friction_magnitude = friction.norm()
        if friction_magnitude > self.params.max_friction_magnitude:
            friction = friction * (self.params.max_friction_magnitude / friction_magnitude)
        
        # Conservative deceleration limit
        # Ensure friction doesn't cause excessive deceleration
        max_delta_v = self.params.max_deceleration * dt
        if friction_magnitude > max_delta_v:
            friction = friction * (max_delta_v / friction_magnitude)
        
        return friction
    
    def compute_total_friction(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        neighbors: List[Tuple[Vector3, Vector3, PlatformType, float]],
        dt: float = 0.1
    ) -> Vector3:
        """
        Compute total friction force from all neighbors.
        
        Args:
            agent_pos: Position of computing agent
            agent_vel: Velocity of computing agent
            agent_type: Platform type of computing agent
            neighbors: List of (position, velocity, type, avoidance_radius) for each neighbor
            dt: Time step for deceleration limiting
            
        Returns:
            Total friction velocity adjustment
        """
        total_friction = Vector3(0, 0, 0)
        
        for neighbor_pos, neighbor_vel, neighbor_type, avoidance_radius in neighbors:
            friction = self.compute_friction_force(
                agent_pos, agent_vel, agent_type,
                neighbor_pos, neighbor_vel, neighbor_type,
                avoidance_radius, dt
            )
            total_friction = total_friction + friction
        
        # Limit total magnitude
        magnitude = total_friction.norm()
        if magnitude > self.params.max_friction_magnitude:
            total_friction = total_friction * (self.params.max_friction_magnitude / magnitude)
        
        # Apply deceleration limit to total
        max_delta_v = self.params.max_deceleration * dt
        if magnitude > max_delta_v:
            total_friction = total_friction * (max_delta_v / magnitude)
        
        return total_friction
    
    def compute_alignment_score(
        self,
        agent_vel: Vector3,
        neighbors: List[Tuple[Vector3, Vector3, PlatformType, float]]
    ) -> float:
        """
        Compute velocity alignment score with neighbors.
        
        Score of 1.0 means perfectly aligned, 0.0 means orthogonal,
        -1.0 means opposing velocities.
        
        Args:
            agent_vel: Velocity of computing agent
            neighbors: List of (position, velocity, type, avoidance_radius)
            
        Returns:
            Average alignment score [-1, 1]
        """
        if not neighbors:
            return 0.0
        
        agent_speed = agent_vel.norm()
        if agent_speed < 1e-6:
            return 0.0
        
        total_alignment = 0.0
        count = 0
        
        for _, neighbor_vel, _, _ in neighbors:
            neighbor_speed = neighbor_vel.norm()
            
            if neighbor_speed < 1e-6:
                continue
            
            # Cosine similarity
            alignment = agent_vel.dot(neighbor_vel) / (agent_speed * neighbor_speed)
            total_alignment += alignment
            count += 1
        
        if count == 0:
            return 0.0
        
        return total_alignment / count
    
    def should_yield(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        agent_priority: float,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        neighbor_type: PlatformType,
        neighbor_priority: float
    ) -> Tuple[bool, float]:
        """
        Determine if agent should yield to neighbor.
        
        Based on priority, platform type, and approach geometry.
        
        Args:
            agent_pos: Position of computing agent
            agent_vel: Velocity of computing agent
            agent_type: Platform type of computing agent
            agent_priority: Priority of computing agent
            neighbor_pos: Position of neighbor
            neighbor_vel: Velocity of neighbor
            neighbor_type: Platform type of neighbor
            neighbor_priority: Priority of neighbor
            
        Returns:
            Tuple of (should_yield, yield_factor)
        """
        # Priority comparison
        priority_diff = neighbor_priority - agent_priority
        
        # Size/type hierarchy
        type_hierarchy = {
            PlatformType.LARGE_UAV: 4,
            PlatformType.MEDIUM_UAV: 3,
            PlatformType.SMALL_UAV: 2,
            PlatformType.MICRO_UAV: 1,
            PlatformType.LARGE_UGV: 4,
            PlatformType.MEDIUM_UGV: 3,
            PlatformType.SMALL_UGV: 2,
            PlatformType.MEDIUM_USV: 3,
            PlatformType.SMALL_USV: 2,
        }
        
        type_diff = type_hierarchy.get(neighbor_type, 2) - type_hierarchy.get(agent_type, 2)
        
        # Approach geometry - who is "on the right" (like traffic rules)
        to_neighbor = neighbor_pos - agent_pos
        cross = agent_vel.cross(to_neighbor)
        
        # Positive cross.z means neighbor is on the right
        right_of_way = 1.0 if cross.z > 0 else -1.0
        
        # Combined yield score
        yield_score = (
            0.5 * priority_diff +          # Priority
            0.3 * type_diff / 3.0 +        # Size hierarchy
            0.2 * right_of_way             # Right of way
        )
        
        should_yield = yield_score > 0.1
        yield_factor = min(1.0, max(0.0, yield_score))
        
        return should_yield, yield_factor


@dataclass
class AdaptiveFrictionController:
    """
    Adaptive friction controller that adjusts friction based on situation.
    
    Increases friction in congested areas, decreases in open space.
    """
    
    base_friction_gain: float = 1.0
    congestion_threshold: int = 3  # Number of neighbors to trigger high congestion
    max_friction_multiplier: float = 2.0
    
    # History for adaptation
    recent_congestion_levels: List[float] = field(default_factory=list)
    history_length: int = 10
    
    def update_congestion(self, num_neighbors: int, average_distance: float) -> float:
        """
        Update congestion level and return adapted friction gain.
        
        Args:
            num_neighbors: Number of neighbors within friction radius
            average_distance: Average distance to neighbors
            
        Returns:
            Adapted friction gain multiplier
        """
        # Congestion metric
        if num_neighbors == 0:
            congestion = 0.0
        else:
            # More neighbors + closer = more congestion
            distance_factor = 1.0 / (1.0 + average_distance)
            congestion = num_neighbors * distance_factor
        
        # Update history
        self.recent_congestion_levels.append(congestion)
        if len(self.recent_congestion_levels) > self.history_length:
            self.recent_congestion_levels.pop(0)
        
        # Average congestion
        avg_congestion = np.mean(self.recent_congestion_levels)
        
        # Adapt friction
        if avg_congestion > self.congestion_threshold:
            # High congestion - increase friction for smoother flow
            multiplier = 1.0 + (avg_congestion - self.congestion_threshold) * 0.2
        else:
            # Low congestion - normal friction
            multiplier = 1.0
        
        return min(multiplier, self.max_friction_multiplier)
    
    def get_adapted_friction_gain(self) -> float:
        """Get current adapted friction gain."""
        if not self.recent_congestion_levels:
            return self.base_friction_gain
        
        avg_congestion = np.mean(self.recent_congestion_levels)
        
        if avg_congestion > self.congestion_threshold:
            multiplier = 1.0 + (avg_congestion - self.congestion_threshold) * 0.2
        else:
            multiplier = 1.0
        
        return self.base_friction_gain * min(multiplier, self.max_friction_multiplier)
