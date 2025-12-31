"""
Heterogeneous Agent Interaction Model (HAIM) Coordinator.

This module provides the main coordinator that integrates:
- Extended repulsion with platform-pair specific gains
- Cross-platform friction with adaptive parameters
- Network-energy-aware self-drive
- Priority hierarchy management

The HAIM coordinator computes desired velocities for heterogeneous
agents considering all interaction forces and constraints.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

from heterofleet.core.platform import (
    PlatformType,
    PlatformSpecification,
    PlatformRegistry,
    PlatformFactory,
    Vector3,
    CollisionEnvelope,
)
from heterofleet.core.state import AgentState, AgentMode
from heterofleet.coordination.repulsion import (
    RepulsionCalculator,
    RepulsionGainMatrix,
    DownwashModel,
    SafetyMarginModel,
)
from heterofleet.coordination.friction import (
    FrictionCalculator,
    FrictionParameters,
    AdaptiveFrictionController,
)
from heterofleet.coordination.self_drive import (
    NetworkEnergyAwareSelfDrive,
    SelfDriveParameters,
    DangerCriteria,
    GeometricCriteria,
    NetworkQualityMap,
)
from heterofleet.coordination.priority import (
    PriorityManager,
    PriorityConfig,
    AgentPriorityInfo,
    PriorityResolution,
    DynamicPriorityAdjuster,
)


@dataclass
class InteractionForce:
    """Result of HAIM interaction force computation."""
    
    # Individual components
    repulsion: Vector3 = field(default_factory=Vector3)
    friction: Vector3 = field(default_factory=Vector3)
    self_drive: Vector3 = field(default_factory=Vector3)
    priority_adjustment: Vector3 = field(default_factory=Vector3)
    
    # Combined desired velocity
    desired_velocity: Vector3 = field(default_factory=Vector3)
    
    # Metadata
    num_neighbors: int = 0
    num_threatening_neighbors: int = 0
    priority: float = 0.5
    
    # Constraints status
    network_constraint_active: bool = False
    energy_constraint_active: bool = False
    
    @property
    def total_interaction(self) -> Vector3:
        """Get total interaction force (repulsion + friction)."""
        return self.repulsion + self.friction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repulsion": (self.repulsion.x, self.repulsion.y, self.repulsion.z),
            "friction": (self.friction.x, self.friction.y, self.friction.z),
            "self_drive": (self.self_drive.x, self.self_drive.y, self.self_drive.z),
            "priority_adjustment": (
                self.priority_adjustment.x,
                self.priority_adjustment.y,
                self.priority_adjustment.z
            ),
            "desired_velocity": (
                self.desired_velocity.x,
                self.desired_velocity.y,
                self.desired_velocity.z
            ),
            "num_neighbors": self.num_neighbors,
            "num_threatening_neighbors": self.num_threatening_neighbors,
            "priority": self.priority,
            "network_constraint_active": self.network_constraint_active,
            "energy_constraint_active": self.energy_constraint_active,
        }


@dataclass
class HAIMParameters:
    """Parameters for HAIM coordinator."""
    
    # Component weights for velocity combination
    repulsion_weight: float = 1.0
    friction_weight: float = 0.5
    self_drive_weight: float = 1.0
    priority_weight: float = 0.3
    
    # Velocity limits
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 3.0  # m/s²
    
    # Neighbor detection radius
    neighbor_radius: float = 5.0  # meters
    
    # Update frequency
    update_rate: float = 10.0  # Hz
    
    # Enable/disable components
    enable_repulsion: bool = True
    enable_friction: bool = True
    enable_self_drive: bool = True
    enable_priority: bool = True
    enable_network_aware: bool = True
    enable_energy_aware: bool = True
    
    # Smoothing
    velocity_smoothing: float = 0.3  # 0 = no smoothing, 1 = full smoothing


class NeighborTracker:
    """
    Tracks neighbors for each agent.
    
    Maintains neighbor lists and provides efficient lookup.
    """
    
    def __init__(self, max_distance: float = 5.0):
        """Initialize neighbor tracker."""
        self.max_distance = max_distance
        
        # Current neighbor lists
        self._neighbors: Dict[str, List[str]] = {}
        
        # Cached state data
        self._states: Dict[str, AgentState] = {}
        self._platforms: Dict[str, PlatformSpecification] = {}
    
    def update_agent(
        self,
        agent_id: str,
        state: AgentState,
        platform_spec: PlatformSpecification
    ) -> None:
        """Update agent state."""
        self._states[agent_id] = state
        self._platforms[agent_id] = platform_spec
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from tracker."""
        self._states.pop(agent_id, None)
        self._platforms.pop(agent_id, None)
        self._neighbors.pop(agent_id, None)
    
    def compute_neighbors(self) -> None:
        """Recompute all neighbor lists."""
        agent_ids = list(self._states.keys())
        
        self._neighbors = {aid: [] for aid in agent_ids}
        
        for i, aid_i in enumerate(agent_ids):
            state_i = self._states[aid_i]
            
            for j, aid_j in enumerate(agent_ids):
                if i >= j:
                    continue
                
                state_j = self._states[aid_j]
                
                # Compute distance
                distance = state_i.distance_to(state_j)
                
                if distance <= self.max_distance:
                    self._neighbors[aid_i].append(aid_j)
                    self._neighbors[aid_j].append(aid_i)
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """Get neighbor IDs for an agent."""
        return self._neighbors.get(agent_id, [])
    
    def get_neighbor_data(
        self,
        agent_id: str
    ) -> List[Tuple[AgentState, PlatformSpecification]]:
        """Get state and platform data for neighbors."""
        neighbor_ids = self.get_neighbors(agent_id)
        
        result = []
        for nid in neighbor_ids:
            if nid in self._states and nid in self._platforms:
                result.append((self._states[nid], self._platforms[nid]))
        
        return result
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get state for an agent."""
        return self._states.get(agent_id)
    
    def get_platform_spec(self, agent_id: str) -> Optional[PlatformSpecification]:
        """Get platform spec for an agent."""
        return self._platforms.get(agent_id)


class HAIMCoordinator:
    """
    Main coordinator for Heterogeneous Agent Interaction Model.
    
    Integrates all HAIM components to compute desired velocities
    for heterogeneous agents in a swarm.
    
    Usage:
        coordinator = HAIMCoordinator()
        
        # Update agent states
        for agent in agents:
            coordinator.update_agent(agent.id, agent.state, agent.platform_spec)
        
        # Compute desired velocities
        for agent in agents:
            force = coordinator.compute_interaction(agent.id, agent.target_position)
            agent.set_velocity(force.desired_velocity)
    """
    
    def __init__(
        self,
        params: HAIMParameters = None,
        platform_factory: PlatformFactory = None,
        network_map: NetworkQualityMap = None
    ):
        """
        Initialize HAIM coordinator.
        
        Args:
            params: HAIM parameters
            platform_factory: Factory for platform interaction parameters
            network_map: Network quality map for network-aware coordination
        """
        self.params = params or HAIMParameters()
        self.platform_factory = platform_factory or PlatformFactory()
        
        # Initialize components
        self.repulsion_calc = RepulsionCalculator(
            max_repulsion_magnitude=self.params.max_velocity,
        )
        
        self.friction_calc = FrictionCalculator(
            FrictionParameters(max_friction_magnitude=self.params.max_velocity * 0.5)
        )
        
        self.self_drive = NetworkEnergyAwareSelfDrive(
            params=SelfDriveParameters(),
            network_map=network_map,
        )
        
        self.priority_manager = PriorityManager()
        self.dynamic_priority = DynamicPriorityAdjuster(self.priority_manager)
        self.adaptive_friction = AdaptiveFrictionController()
        
        # Neighbor tracking
        self.neighbor_tracker = NeighborTracker(self.params.neighbor_radius)
        
        # Previous velocities for smoothing
        self._prev_velocities: Dict[str, Vector3] = {}
        
        # Statistics
        self._stats = {
            "total_updates": 0,
            "avg_neighbors": 0.0,
            "max_neighbors": 0,
            "avg_computation_time_ms": 0.0,
        }
    
    def update_agent(
        self,
        agent_id: str,
        state: AgentState,
        platform_spec: PlatformSpecification
    ) -> None:
        """
        Update agent state in the coordinator.
        
        Args:
            agent_id: Agent identifier
            state: Current agent state
            platform_spec: Platform specification
        """
        self.neighbor_tracker.update_agent(agent_id, state, platform_spec)
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from coordinator."""
        self.neighbor_tracker.remove_agent(agent_id)
        self._prev_velocities.pop(agent_id, None)
    
    def update_neighbors(self) -> None:
        """Recompute neighbor lists for all agents."""
        self.neighbor_tracker.compute_neighbors()
    
    def _get_collision_envelope(
        self,
        platform_spec: PlatformSpecification
    ) -> CollisionEnvelope:
        """Get collision envelope from platform spec."""
        return platform_spec.physical_properties.collision_envelope
    
    def _prepare_repulsion_data(
        self,
        agent_id: str,
        agent_state: AgentState,
        agent_spec: PlatformSpecification
    ) -> List[Tuple[Vector3, Vector3, PlatformType, CollisionEnvelope]]:
        """Prepare neighbor data for repulsion calculation."""
        neighbors = self.neighbor_tracker.get_neighbor_data(agent_id)
        
        result = []
        for neighbor_state, neighbor_spec in neighbors:
            result.append((
                neighbor_state.position_vec,
                neighbor_state.velocity_vec,
                neighbor_spec.platform_type,
                self._get_collision_envelope(neighbor_spec),
            ))
        
        return result
    
    def _prepare_friction_data(
        self,
        agent_id: str,
        agent_state: AgentState,
        agent_spec: PlatformSpecification
    ) -> List[Tuple[Vector3, Vector3, PlatformType, float]]:
        """Prepare neighbor data for friction calculation."""
        neighbors = self.neighbor_tracker.get_neighbor_data(agent_id)
        
        result = []
        for neighbor_state, neighbor_spec in neighbors:
            # Compute combined avoidance radius
            agent_envelope = self._get_collision_envelope(agent_spec)
            neighbor_envelope = self._get_collision_envelope(neighbor_spec)
            
            avoidance_radius = self.repulsion_calc.compute_combined_radius(
                agent_spec.platform_type,
                neighbor_spec.platform_type,
                agent_envelope,
                neighbor_envelope
            )
            
            result.append((
                neighbor_state.position_vec,
                neighbor_state.velocity_vec,
                neighbor_spec.platform_type,
                avoidance_radius,
            ))
        
        return result
    
    def _prepare_self_drive_data(
        self,
        agent_id: str,
        agent_state: AgentState,
        agent_spec: PlatformSpecification
    ) -> List[Tuple[Vector3, Vector3, float]]:
        """Prepare neighbor data for self-drive calculation."""
        neighbors = self.neighbor_tracker.get_neighbor_data(agent_id)
        
        result = []
        for neighbor_state, neighbor_spec in neighbors:
            agent_envelope = self._get_collision_envelope(agent_spec)
            neighbor_envelope = self._get_collision_envelope(neighbor_spec)
            
            combined_radius = self.repulsion_calc.compute_combined_radius(
                agent_spec.platform_type,
                neighbor_spec.platform_type,
                agent_envelope,
                neighbor_envelope
            )
            
            result.append((
                neighbor_state.position_vec,
                neighbor_state.velocity_vec,
                combined_radius,
            ))
        
        return result
    
    def _create_priority_info(
        self,
        agent_id: str,
        agent_state: AgentState,
        agent_spec: PlatformSpecification
    ) -> AgentPriorityInfo:
        """Create priority info for an agent."""
        return AgentPriorityInfo(
            agent_id=agent_id,
            platform_type=agent_spec.platform_type,
            position=agent_state.position_vec,
            velocity=agent_state.velocity_vec,
            mode=agent_state.mode,
            energy_level=agent_state.energy_level,
            task_priority=agent_state.priority,
        )
    
    def compute_interaction(
        self,
        agent_id: str,
        target_position: Vector3,
        home_position: Vector3 = None,
        dt: float = 0.1
    ) -> InteractionForce:
        """
        Compute HAIM interaction forces for an agent.
        
        This is the main entry point for computing desired velocity.
        
        Args:
            agent_id: Agent identifier
            target_position: Target/goal position
            home_position: Home position for energy constraint
            dt: Time step
            
        Returns:
            InteractionForce with all components and desired velocity
        """
        import time
        start_time = time.time()
        
        # Get agent data
        agent_state = self.neighbor_tracker.get_agent_state(agent_id)
        agent_spec = self.neighbor_tracker.get_platform_spec(agent_id)
        
        if agent_state is None or agent_spec is None:
            logger.warning(f"Agent {agent_id} not found in coordinator")
            return InteractionForce()
        
        agent_pos = agent_state.position_vec
        agent_vel = agent_state.velocity_vec
        agent_envelope = self._get_collision_envelope(agent_spec)
        
        result = InteractionForce()
        result.num_neighbors = len(self.neighbor_tracker.get_neighbors(agent_id))
        
        # Get max speed from platform spec
        max_speed = agent_spec.dynamic_properties.max_velocity.norm()
        
        # 1. Compute repulsion
        if self.params.enable_repulsion:
            repulsion_data = self._prepare_repulsion_data(agent_id, agent_state, agent_spec)
            result.repulsion = self.repulsion_calc.compute_total_repulsion(
                agent_pos, agent_vel,
                agent_spec.platform_type, agent_envelope,
                repulsion_data
            )
        
        # 2. Compute friction
        if self.params.enable_friction:
            friction_data = self._prepare_friction_data(agent_id, agent_state, agent_spec)
            
            # Adapt friction based on congestion
            if friction_data:
                avg_distance = np.mean([
                    (agent_pos - fp[0]).norm() for fp in friction_data
                ])
            else:
                avg_distance = self.params.neighbor_radius
            
            friction_multiplier = self.adaptive_friction.update_congestion(
                len(friction_data), avg_distance
            )
            
            result.friction = self.friction_calc.compute_total_friction(
                agent_pos, agent_vel, agent_spec.platform_type,
                friction_data, dt
            )
            result.friction = result.friction * friction_multiplier
        
        # 3. Compute self-drive
        if self.params.enable_self_drive:
            self_drive_data = self._prepare_self_drive_data(agent_id, agent_state, agent_spec)
            
            self_drive_vel, debug_info = self.self_drive.compute_self_drive_velocity(
                agent_pos, agent_vel, target_position,
                self_drive_data,
                agent_spec.platform_type,
                max_speed,
                agent_state.energy_level,
                agent_state.network_quality,
                home_position or Vector3(0, 0, 0),
                dt
            )
            
            result.self_drive = self_drive_vel
            result.num_threatening_neighbors = debug_info.get("threatening_neighbors", 0)
            result.network_constraint_active = debug_info.get("network_constraint_active", False)
            result.energy_constraint_active = debug_info.get("energy_constraint_active", False)
        
        # 4. Compute priority adjustment
        if self.params.enable_priority:
            agent_priority = self._create_priority_info(agent_id, agent_state, agent_spec)
            
            neighbor_priorities = []
            for neighbor_state, neighbor_spec in self.neighbor_tracker.get_neighbor_data(agent_id):
                neighbor_priority = self._create_priority_info(
                    neighbor_state.agent_id, neighbor_state, neighbor_spec
                )
                neighbor_priorities.append(neighbor_priority)
            
            result.priority = self.priority_manager.compute_priority(
                agent_priority, neighbor_priorities
            )
            
            # Adjust velocity based on priority
            for neighbor_priority in neighbor_priorities:
                resolution = self.priority_manager.resolve_conflict(
                    agent_priority, neighbor_priority
                )
                
                if resolution.yielder_id == agent_id:
                    # Need to adjust velocity
                    yield_factor = resolution.yield_factor * self.params.priority_weight
                    result.priority_adjustment = result.priority_adjustment - (
                        agent_vel * yield_factor * 0.3
                    )
        
        # 5. Combine all components
        desired_vel = (
            result.self_drive * self.params.self_drive_weight +
            result.repulsion * self.params.repulsion_weight +
            result.friction * self.params.friction_weight +
            result.priority_adjustment * self.params.priority_weight
        )
        
        # 6. Clamp velocity
        speed = desired_vel.norm()
        if speed > self.params.max_velocity:
            desired_vel = desired_vel * (self.params.max_velocity / speed)
        
        # 7. Apply smoothing
        if self.params.velocity_smoothing > 0:
            prev_vel = self._prev_velocities.get(agent_id, desired_vel)
            alpha = self.params.velocity_smoothing
            desired_vel = prev_vel * alpha + desired_vel * (1 - alpha)
        
        self._prev_velocities[agent_id] = desired_vel
        result.desired_velocity = desired_vel
        
        # Update statistics
        computation_time_ms = (time.time() - start_time) * 1000
        self._update_stats(result.num_neighbors, computation_time_ms)
        
        return result
    
    def compute_all_interactions(
        self,
        targets: Dict[str, Vector3],
        home_positions: Dict[str, Vector3] = None,
        dt: float = 0.1
    ) -> Dict[str, InteractionForce]:
        """
        Compute interactions for all agents.
        
        Args:
            targets: Dictionary of agent_id -> target position
            home_positions: Dictionary of agent_id -> home position
            dt: Time step
            
        Returns:
            Dictionary of agent_id -> InteractionForce
        """
        # First update neighbor lists
        self.update_neighbors()
        
        home_positions = home_positions or {}
        
        results = {}
        for agent_id, target in targets.items():
            home = home_positions.get(agent_id, Vector3(0, 0, 0))
            results[agent_id] = self.compute_interaction(agent_id, target, home, dt)
        
        return results
    
    def _update_stats(self, num_neighbors: int, computation_time_ms: float) -> None:
        """Update running statistics."""
        n = self._stats["total_updates"]
        
        # Running average
        self._stats["avg_neighbors"] = (
            self._stats["avg_neighbors"] * n + num_neighbors
        ) / (n + 1)
        
        self._stats["avg_computation_time_ms"] = (
            self._stats["avg_computation_time_ms"] * n + computation_time_ms
        ) / (n + 1)
        
        self._stats["max_neighbors"] = max(
            self._stats["max_neighbors"], num_neighbors
        )
        
        self._stats["total_updates"] = n + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_updates": 0,
            "avg_neighbors": 0.0,
            "max_neighbors": 0,
            "avg_computation_time_ms": 0.0,
        }
    
    def set_network_map(self, network_map: NetworkQualityMap) -> None:
        """Update network quality map."""
        self.self_drive.network_map = network_map
    
    def visualize_force_field(
        self,
        agent_type: PlatformType,
        bounds: Tuple[Vector3, Vector3],
        resolution: float = 0.2,
        neighbors: List[Tuple[Vector3, PlatformType]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate force field visualization data.
        
        Args:
            agent_type: Platform type to visualize for
            bounds: (min_corner, max_corner) of region
            resolution: Grid resolution
            neighbors: Static neighbor positions for visualization
            
        Returns:
            Tuple of (X, Y, force_magnitude) arrays
        """
        from heterofleet.core.platform import get_default_crazyflie_spec
        
        # Create temporary envelope
        default_spec = get_default_crazyflie_spec()
        agent_envelope = default_spec.physical_properties.collision_envelope
        
        min_corner, max_corner = bounds
        
        x = np.arange(min_corner.x, max_corner.x, resolution)
        y = np.arange(min_corner.y, max_corner.y, resolution)
        X, Y = np.meshgrid(x, y)
        
        force_magnitude = np.zeros_like(X)
        
        zero_vel = Vector3(0, 0, 0)
        
        if neighbors is None:
            neighbors = []
        
        # Prepare neighbor data
        neighbor_data = []
        for n_pos, n_type in neighbors:
            neighbor_data.append((
                n_pos, zero_vel, n_type, agent_envelope
            ))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = Vector3(X[i, j], Y[i, j], 0)
                
                repulsion = self.repulsion_calc.compute_total_repulsion(
                    pos, zero_vel, agent_type, agent_envelope, neighbor_data
                )
                
                force_magnitude[i, j] = repulsion.norm()
        
        return X, Y, force_magnitude
