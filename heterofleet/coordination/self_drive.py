"""
Network-Energy-Aware Self-Drive for heterogeneous agents.

Implements the self-drive velocity selection algorithm that
combines goal-seeking with network quality and energy constraints.

Key features:
- Danger criteria (D1-D4) for collision avoidance
- Geometric criteria (C1-C5) for safe velocity selection
- Network quality constraints (N1)
- Energy conservation constraints (N2)

Based on HeteroFleet Architecture v1.0 and Decentralized Traffic Management paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3, CollisionEnvelope


@dataclass
class DangerCriteria:
    """
    Danger criteria for identifying threatening neighbors.
    
    Based on Decentralized Traffic Management paper:
    - D1: Collision course
    - D2: Relative velocity toward agent
    - D3: Contribution to collision
    - D4: Within prediction horizon
    """
    
    # D1: Distance threshold for collision course detection
    collision_distance_threshold: float = 0.5
    
    # D2: Relative velocity threshold (m/s)
    relative_velocity_threshold: float = 0.1
    
    # D3: Contribution angle threshold (radians)
    contribution_angle_threshold: float = np.pi / 4  # 45 degrees
    
    # D4: Prediction horizon (seconds)
    prediction_horizon: float = 3.0
    
    def check_d1_collision_course(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        combined_radius: float
    ) -> bool:
        """
        Check D1: Are agents on collision course?
        
        Projects future positions and checks if they intersect.
        """
        # Relative position and velocity
        rel_pos = neighbor_pos - agent_pos
        rel_vel = neighbor_vel - agent_vel
        
        # Time to closest approach
        rel_vel_sq = rel_vel.dot(rel_vel)
        if rel_vel_sq < 1e-10:
            return False  # No relative motion
        
        t_cpa = -rel_pos.dot(rel_vel) / rel_vel_sq
        
        if t_cpa < 0 or t_cpa > self.prediction_horizon:
            return False  # CPA in past or too far in future
        
        # Distance at CPA
        cpa_pos = rel_pos + rel_vel * t_cpa
        d_cpa = cpa_pos.norm()
        
        return d_cpa < combined_radius + self.collision_distance_threshold
    
    def check_d2_approaching(
        self,
        agent_pos: Vector3,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3
    ) -> bool:
        """
        Check D2: Is neighbor approaching?
        
        Checks if neighbor's velocity has component toward agent.
        """
        # Direction from neighbor to agent
        to_agent = agent_pos - neighbor_pos
        distance = to_agent.norm()
        
        if distance < 1e-6:
            return True  # Already at same position
        
        to_agent_unit = to_agent.normalized()
        
        # Component of neighbor velocity toward agent
        approach_speed = neighbor_vel.dot(to_agent_unit)
        
        return approach_speed > self.relative_velocity_threshold
    
    def check_d3_contributing(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        neighbor_pos: Vector3
    ) -> bool:
        """
        Check D3: Is agent contributing to collision?
        
        Checks if agent's velocity has component toward neighbor.
        """
        # Direction from agent to neighbor
        to_neighbor = neighbor_pos - agent_pos
        distance = to_neighbor.norm()
        
        if distance < 1e-6:
            return True
        
        to_neighbor_unit = to_neighbor.normalized()
        
        # Angle between velocity and direction to neighbor
        agent_speed = agent_vel.norm()
        if agent_speed < 1e-6:
            return False  # Agent not moving
        
        cos_angle = agent_vel.dot(to_neighbor_unit) / agent_speed
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return angle < self.contribution_angle_threshold
    
    def check_d4_in_horizon(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        max_speed: float
    ) -> bool:
        """
        Check D4: Is collision possible within horizon?
        
        Checks if agents can reach each other within prediction horizon.
        """
        distance = (neighbor_pos - agent_pos).norm()
        
        # Maximum closing speed
        max_closing = agent_vel.norm() + neighbor_vel.norm() + 2 * max_speed
        
        # Minimum time to reach
        if max_closing < 1e-6:
            return False
        
        min_time = distance / max_closing
        
        return min_time < self.prediction_horizon
    
    def is_threatening(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        combined_radius: float,
        max_speed: float
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check all danger criteria.
        
        Returns:
            Tuple of (is_threatening, criteria_results)
        """
        d1 = self.check_d1_collision_course(
            agent_pos, agent_vel, neighbor_pos, neighbor_vel, combined_radius
        )
        d2 = self.check_d2_approaching(agent_pos, neighbor_pos, neighbor_vel)
        d3 = self.check_d3_contributing(agent_pos, agent_vel, neighbor_pos)
        d4 = self.check_d4_in_horizon(agent_pos, agent_vel, neighbor_pos, neighbor_vel, max_speed)
        
        # A neighbor is threatening if all criteria are met
        is_threat = d1 and d2 and d3 and d4
        
        return is_threat, {"D1": d1, "D2": d2, "D3": d3, "D4": d4}


@dataclass
class GeometricCriteria:
    """
    Geometric criteria for safe velocity selection.
    
    Based on Decentralized Traffic Management paper:
    - C1: Outside collision cone
    - C2: Velocity magnitude feasible
    - C3: Within acceleration limits
    - C4: Respects priority
    - C5: Minimizes deviation from desired
    """
    
    # C2: Speed limits
    max_speed: float = 2.0
    min_speed: float = 0.0
    
    # C3: Acceleration limits (m/s per iteration)
    max_acceleration: float = 0.5
    max_deceleration: float = 0.5
    
    # C5: Deviation weight
    deviation_weight: float = 1.0
    
    # Resolution for velocity sampling
    velocity_resolution: int = 8  # Number of directions to sample
    speed_resolution: int = 5    # Number of speeds to sample
    
    def check_c1_outside_cone(
        self,
        candidate_vel: Vector3,
        agent_pos: Vector3,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        combined_radius: float,
        prediction_horizon: float
    ) -> bool:
        """
        Check C1: Is velocity outside collision cone?
        
        Collision cone is the set of velocities that lead to collision.
        """
        # Relative position
        rel_pos = neighbor_pos - agent_pos
        distance = rel_pos.norm()
        
        if distance < combined_radius:
            # Already colliding
            return False
        
        # Relative velocity
        rel_vel = neighbor_vel - candidate_vel
        
        # Time to closest approach
        rel_vel_sq = rel_vel.dot(rel_vel)
        if rel_vel_sq < 1e-10:
            # No relative motion - safe if not already colliding
            return True
        
        t_cpa = -rel_pos.dot(rel_vel) / rel_vel_sq
        
        if t_cpa < 0:
            # Diverging
            return True
        
        if t_cpa > prediction_horizon:
            # CPA beyond horizon
            return True
        
        # Distance at CPA
        cpa_pos = rel_pos + rel_vel * t_cpa
        d_cpa = cpa_pos.norm()
        
        return d_cpa >= combined_radius
    
    def check_c2_speed_feasible(self, candidate_vel: Vector3) -> bool:
        """Check C2: Is speed within limits?"""
        speed = candidate_vel.norm()
        return self.min_speed <= speed <= self.max_speed
    
    def check_c3_acceleration_feasible(
        self,
        candidate_vel: Vector3,
        current_vel: Vector3,
        dt: float
    ) -> bool:
        """Check C3: Is acceleration within limits?"""
        delta_v = candidate_vel - current_vel
        acceleration = delta_v.norm() / dt if dt > 0 else float('inf')
        
        # Check if accelerating or decelerating
        speed_change = candidate_vel.norm() - current_vel.norm()
        
        if speed_change >= 0:
            return acceleration <= self.max_acceleration / dt
        else:
            return acceleration <= self.max_deceleration / dt
    
    def compute_c5_deviation(
        self,
        candidate_vel: Vector3,
        desired_vel: Vector3
    ) -> float:
        """Compute C5: Deviation from desired velocity."""
        diff = candidate_vel - desired_vel
        return diff.norm() * self.deviation_weight
    
    def generate_candidate_velocities(
        self,
        current_vel: Vector3,
        desired_direction: Vector3,
        dt: float
    ) -> List[Vector3]:
        """
        Generate candidate velocities to evaluate.
        
        Samples velocities in a cone around the desired direction
        at various speeds.
        """
        candidates = []
        
        # Always include zero velocity
        candidates.append(Vector3(0, 0, 0))
        
        # Include current velocity
        candidates.append(current_vel)
        
        # Speeds to sample
        speeds = np.linspace(self.min_speed, self.max_speed, self.speed_resolution)
        
        # Directions to sample (in horizontal plane)
        angles = np.linspace(0, 2 * np.pi, self.velocity_resolution, endpoint=False)
        
        # Reference direction
        ref_dir = desired_direction.normalized() if desired_direction.norm() > 1e-6 else Vector3(1, 0, 0)
        
        for speed in speeds:
            for angle in angles:
                # Rotate reference direction
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                direction = Vector3(
                    ref_dir.x * cos_a - ref_dir.y * sin_a,
                    ref_dir.x * sin_a + ref_dir.y * cos_a,
                    ref_dir.z
                )
                
                candidate = direction * speed
                candidates.append(candidate)
        
        return candidates


@dataclass
class SelfDriveParameters:
    """Parameters for network-energy-aware self-drive."""
    
    # Goal-seeking gain
    goal_gain: float = 1.0
    
    # Network quality threshold
    min_network_quality: float = 0.3
    
    # Energy reserve fraction
    energy_reserve_fraction: float = 0.2
    
    # Network gradient following gain
    network_gradient_gain: float = 0.5
    
    # Energy conservation gain (reduces speed when low)
    energy_conservation_gain: float = 0.3
    
    # Maximum iterations for velocity refinement
    max_iterations: int = 10
    
    # Convergence threshold
    convergence_threshold: float = 0.01


class NetworkQualityMap:
    """
    Network quality map for estimating QoS at positions.
    
    Uses a simple grid-based model with interpolation.
    Can be updated via federated learning from agent observations.
    """
    
    def __init__(
        self,
        bounds: Tuple[Vector3, Vector3],
        resolution: float = 1.0,
        default_quality: float = 0.8
    ):
        """
        Initialize network quality map.
        
        Args:
            bounds: (min_corner, max_corner) of the map
            resolution: Grid cell size in meters
            default_quality: Default quality value
        """
        self.bounds = bounds
        self.resolution = resolution
        self.default_quality = default_quality
        
        min_corner, max_corner = bounds
        self.dims = (
            max(1, int(np.ceil((max_corner.x - min_corner.x) / resolution))),
            max(1, int(np.ceil((max_corner.y - min_corner.y) / resolution))),
            max(1, int(np.ceil((max_corner.z - min_corner.z) / resolution))),
        )
        
        # Quality grid
        self.quality_grid = np.ones(self.dims) * default_quality
        
        # Observation counts for federated averaging
        self.observation_counts = np.zeros(self.dims)
    
    def _position_to_index(self, position: Vector3) -> Tuple[int, int, int]:
        """Convert position to grid index."""
        min_corner = self.bounds[0]
        
        ix = int((position.x - min_corner.x) / self.resolution)
        iy = int((position.y - min_corner.y) / self.resolution)
        iz = int((position.z - min_corner.z) / self.resolution)
        
        # Clamp to valid range
        ix = max(0, min(self.dims[0] - 1, ix))
        iy = max(0, min(self.dims[1] - 1, iy))
        iz = max(0, min(self.dims[2] - 1, iz))
        
        return ix, iy, iz
    
    def get_quality(self, position: Vector3) -> float:
        """Get network quality at a position (with interpolation)."""
        ix, iy, iz = self._position_to_index(position)
        
        # Simple nearest-neighbor for now
        return float(self.quality_grid[ix, iy, iz])
    
    def get_gradient(self, position: Vector3) -> Vector3:
        """
        Get network quality gradient at a position.
        
        Points toward higher quality regions.
        """
        ix, iy, iz = self._position_to_index(position)
        
        # Compute gradient using central differences
        dx = dy = dz = 0.0
        
        if ix > 0 and ix < self.dims[0] - 1:
            dx = (self.quality_grid[ix + 1, iy, iz] - self.quality_grid[ix - 1, iy, iz]) / (2 * self.resolution)
        elif ix == 0:
            dx = (self.quality_grid[ix + 1, iy, iz] - self.quality_grid[ix, iy, iz]) / self.resolution
        else:
            dx = (self.quality_grid[ix, iy, iz] - self.quality_grid[ix - 1, iy, iz]) / self.resolution
        
        if iy > 0 and iy < self.dims[1] - 1:
            dy = (self.quality_grid[ix, iy + 1, iz] - self.quality_grid[ix, iy - 1, iz]) / (2 * self.resolution)
        elif iy == 0:
            dy = (self.quality_grid[ix, iy + 1, iz] - self.quality_grid[ix, iy, iz]) / self.resolution
        else:
            dy = (self.quality_grid[ix, iy, iz] - self.quality_grid[ix, iy - 1, iz]) / self.resolution
        
        if iz > 0 and iz < self.dims[2] - 1:
            dz = (self.quality_grid[ix, iy, iz + 1] - self.quality_grid[ix, iy, iz - 1]) / (2 * self.resolution)
        elif iz == 0 and self.dims[2] > 1:
            dz = (self.quality_grid[ix, iy, iz + 1] - self.quality_grid[ix, iy, iz]) / self.resolution
        elif self.dims[2] > 1:
            dz = (self.quality_grid[ix, iy, iz] - self.quality_grid[ix, iy, iz - 1]) / self.resolution
        
        return Vector3(dx, dy, dz)
    
    def update(self, position: Vector3, quality: float, weight: float = 1.0) -> None:
        """
        Update network quality at a position.
        
        Uses weighted averaging with existing value.
        """
        ix, iy, iz = self._position_to_index(position)
        
        # Weighted average
        old_count = self.observation_counts[ix, iy, iz]
        new_count = old_count + weight
        
        self.quality_grid[ix, iy, iz] = (
            self.quality_grid[ix, iy, iz] * old_count + quality * weight
        ) / new_count
        
        self.observation_counts[ix, iy, iz] = new_count
    
    def update_batch(
        self,
        positions: List[Vector3],
        qualities: List[float],
        weights: List[float] = None
    ) -> None:
        """Update multiple positions."""
        if weights is None:
            weights = [1.0] * len(positions)
        
        for pos, qual, weight in zip(positions, qualities, weights):
            self.update(pos, qual, weight)


class NetworkEnergyAwareSelfDrive:
    """
    Network-energy-aware self-drive velocity computation.
    
    Computes safe velocity toward goal while:
    - Avoiding collisions with threatening neighbors
    - Maintaining network connectivity
    - Conserving energy
    """
    
    def __init__(
        self,
        params: SelfDriveParameters = None,
        danger_criteria: DangerCriteria = None,
        geometric_criteria: GeometricCriteria = None,
        network_map: NetworkQualityMap = None
    ):
        """Initialize self-drive calculator."""
        self.params = params or SelfDriveParameters()
        self.danger_criteria = danger_criteria or DangerCriteria()
        self.geometric_criteria = geometric_criteria or GeometricCriteria()
        self.network_map = network_map
    
    def compute_goal_velocity(
        self,
        agent_pos: Vector3,
        target_pos: Vector3,
        max_speed: float
    ) -> Vector3:
        """Compute velocity toward goal."""
        to_goal = target_pos - agent_pos
        distance = to_goal.norm()
        
        if distance < 1e-6:
            return Vector3(0, 0, 0)
        
        # Speed proportional to distance, capped at max
        speed = min(self.params.goal_gain * distance, max_speed)
        
        return to_goal.normalized() * speed
    
    def identify_threatening_neighbors(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        neighbors: List[Tuple[Vector3, Vector3, float]],
        max_speed: float
    ) -> List[Tuple[Vector3, Vector3, float]]:
        """
        Identify threatening neighbors using danger criteria.
        
        Args:
            agent_pos: Agent position
            agent_vel: Agent velocity
            neighbors: List of (position, velocity, combined_radius)
            max_speed: Maximum agent speed
            
        Returns:
            List of threatening neighbors
        """
        threats = []
        
        for neighbor_pos, neighbor_vel, combined_radius in neighbors:
            is_threat, _ = self.danger_criteria.is_threatening(
                agent_pos, agent_vel,
                neighbor_pos, neighbor_vel,
                combined_radius, max_speed
            )
            
            if is_threat:
                threats.append((neighbor_pos, neighbor_vel, combined_radius))
        
        return threats
    
    def check_network_constraint(
        self,
        position: Vector3,
        required_quality: float
    ) -> bool:
        """
        Check N1: Network quality constraint.
        
        Args:
            position: Position to check
            required_quality: Required network quality
            
        Returns:
            True if constraint satisfied
        """
        if self.network_map is None:
            return True
        
        quality = self.network_map.get_quality(position)
        return quality >= required_quality
    
    def check_energy_constraint(
        self,
        agent_pos: Vector3,
        home_pos: Vector3,
        current_energy: float,
        energy_consumption_rate: float,
        speed: float
    ) -> bool:
        """
        Check N2: Energy constraint.
        
        Ensures enough energy to return home with reserve.
        
        Args:
            agent_pos: Current position
            home_pos: Home position
            current_energy: Current energy (Wh)
            energy_consumption_rate: Energy consumption (W)
            speed: Speed to check
            
        Returns:
            True if constraint satisfied
        """
        # Distance to home
        distance_to_home = (home_pos - agent_pos).norm()
        
        if speed < 1e-6:
            return True  # Stationary is always feasible
        
        # Time to return
        time_to_return = distance_to_home / speed
        
        # Energy needed
        energy_needed = energy_consumption_rate * time_to_return / 3600  # Convert W*s to Wh
        
        # Reserve
        energy_reserve = current_energy * self.params.energy_reserve_fraction
        
        return current_energy - energy_reserve >= energy_needed
    
    def select_safe_velocity(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        desired_vel: Vector3,
        threatening_neighbors: List[Tuple[Vector3, Vector3, float]],
        dt: float
    ) -> Vector3:
        """
        Select safe velocity using geometric criteria.
        
        Args:
            agent_pos: Agent position
            agent_vel: Current velocity
            desired_vel: Desired velocity toward goal
            threatening_neighbors: List of (position, velocity, combined_radius)
            dt: Time step
            
        Returns:
            Safe velocity
        """
        if not threatening_neighbors:
            # No threats - use desired velocity
            if self.geometric_criteria.check_c2_speed_feasible(desired_vel):
                if self.geometric_criteria.check_c3_acceleration_feasible(desired_vel, agent_vel, dt):
                    return desired_vel
        
        # Generate candidates
        candidates = self.geometric_criteria.generate_candidate_velocities(
            agent_vel, desired_vel, dt
        )
        
        # Evaluate candidates
        best_velocity = Vector3(0, 0, 0)
        best_score = float('inf')
        
        for candidate in candidates:
            # Check C2: Speed feasible
            if not self.geometric_criteria.check_c2_speed_feasible(candidate):
                continue
            
            # Check C3: Acceleration feasible
            if not self.geometric_criteria.check_c3_acceleration_feasible(candidate, agent_vel, dt):
                continue
            
            # Check C1: Outside all collision cones
            safe = True
            for neighbor_pos, neighbor_vel, combined_radius in threatening_neighbors:
                if not self.geometric_criteria.check_c1_outside_cone(
                    candidate, agent_pos,
                    neighbor_pos, neighbor_vel, combined_radius,
                    self.danger_criteria.prediction_horizon
                ):
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Compute C5: Deviation score
            deviation = self.geometric_criteria.compute_c5_deviation(candidate, desired_vel)
            
            if deviation < best_score:
                best_score = deviation
                best_velocity = candidate
        
        return best_velocity
    
    def compute_network_adjustment(
        self,
        agent_pos: Vector3,
        current_quality: float
    ) -> Vector3:
        """
        Compute velocity adjustment toward better network quality.
        
        Args:
            agent_pos: Agent position
            current_quality: Current network quality
            
        Returns:
            Velocity adjustment toward better network
        """
        if self.network_map is None:
            return Vector3(0, 0, 0)
        
        # Only adjust if quality is low
        if current_quality >= self.params.min_network_quality:
            return Vector3(0, 0, 0)
        
        # Get gradient
        gradient = self.network_map.get_gradient(agent_pos)
        
        # Scale by how much we need improvement
        quality_deficit = self.params.min_network_quality - current_quality
        
        return gradient * (self.params.network_gradient_gain * quality_deficit)
    
    def compute_energy_adjustment(
        self,
        energy_level: float,
        current_speed: float
    ) -> float:
        """
        Compute speed reduction for energy conservation.
        
        Args:
            energy_level: Current energy (0-1)
            current_speed: Current speed
            
        Returns:
            Adjusted speed
        """
        if energy_level > self.params.energy_reserve_fraction * 2:
            return current_speed
        
        # Linear reduction as energy decreases
        reduction = 1.0 - self.params.energy_conservation_gain * (
            1.0 - energy_level / (self.params.energy_reserve_fraction * 2)
        )
        
        return current_speed * max(0.1, reduction)
    
    def compute_self_drive_velocity(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        target_pos: Vector3,
        neighbors: List[Tuple[Vector3, Vector3, float]],
        platform_type: PlatformType,
        max_speed: float,
        energy_level: float,
        network_quality: float,
        home_pos: Vector3 = None,
        dt: float = 0.1
    ) -> Tuple[Vector3, Dict]:
        """
        Compute network-energy-aware self-drive velocity.
        
        This is the main entry point for velocity computation.
        
        Args:
            agent_pos: Current agent position
            agent_vel: Current agent velocity
            target_pos: Target/goal position
            neighbors: List of (position, velocity, combined_radius) for neighbors
            platform_type: Agent platform type
            max_speed: Maximum speed
            energy_level: Current energy level (0-1)
            network_quality: Current network quality (0-1)
            home_pos: Home position for energy constraint
            dt: Time step
            
        Returns:
            Tuple of (computed_velocity, debug_info)
        """
        debug_info = {
            "threatening_neighbors": 0,
            "network_constraint_active": False,
            "energy_constraint_active": False,
            "final_speed_reduction": 1.0,
        }
        
        # Step 1: Compute goal velocity
        goal_vel = self.compute_goal_velocity(agent_pos, target_pos, max_speed)
        
        # Step 2: Identify threatening neighbors
        threats = self.identify_threatening_neighbors(
            agent_pos, agent_vel, neighbors, max_speed
        )
        debug_info["threatening_neighbors"] = len(threats)
        
        # Step 3: Select safe velocity
        safe_vel = self.select_safe_velocity(
            agent_pos, agent_vel, goal_vel, threats, dt
        )
        
        # Step 4: Network quality adjustment
        if network_quality < self.params.min_network_quality:
            debug_info["network_constraint_active"] = True
            network_adj = self.compute_network_adjustment(agent_pos, network_quality)
            safe_vel = safe_vel + network_adj
        
        # Step 5: Energy conservation
        if energy_level < self.params.energy_reserve_fraction * 2:
            debug_info["energy_constraint_active"] = True
            original_speed = safe_vel.norm()
            adjusted_speed = self.compute_energy_adjustment(energy_level, original_speed)
            
            if original_speed > 1e-6:
                speed_factor = adjusted_speed / original_speed
                debug_info["final_speed_reduction"] = speed_factor
                safe_vel = safe_vel * speed_factor
        
        # Step 6: Final feasibility check
        if not self.geometric_criteria.check_c2_speed_feasible(safe_vel):
            # Clamp to max speed
            speed = safe_vel.norm()
            if speed > self.geometric_criteria.max_speed:
                safe_vel = safe_vel * (self.geometric_criteria.max_speed / speed)
        
        return safe_vel, debug_info
