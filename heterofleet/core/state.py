"""
State representations for heterogeneous autonomous vehicle swarms.

This module defines:
- AgentState: Individual agent state representation
- FleetState: Collective fleet state
- StateEstimate: Filtered/predicted state with uncertainty
- Supporting data structures for health, behavior, and anomalies

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from heterofleet.core.platform import (
    PlatformType,
    PlatformSpecification,
    Vector3,
    CollisionEnvelope,
)


class AgentMode(Enum):
    """Operational mode of an agent."""
    IDLE = auto()           # Stationary, waiting for commands
    TRANSIT = auto()        # Moving to target
    HOVER = auto()          # Hovering at position (UAVs)
    EXECUTE = auto()        # Executing task
    RETURN = auto()         # Returning to base
    LANDING = auto()        # Landing procedure
    TAKEOFF = auto()        # Takeoff procedure
    DOCKING = auto()        # Docking procedure
    EMERGENCY = auto()      # Emergency mode
    CHARGING = auto()       # Charging/refueling
    MAINTENANCE = auto()    # Under maintenance
    NAVIGATING = auto()     # Navigating to waypoint


# Alias for compatibility
OperationalMode = AgentMode


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    NONE = auto()
    TRAJECTORY_DEVIATION = auto()
    COMMUNICATION_LOSS = auto()
    MOTOR_FAULT = auto()
    SENSOR_FAULT = auto()
    BATTERY_CRITICAL = auto()
    COLLISION_IMMINENT = auto()
    NETWORK_DEGRADED = auto()
    UNEXPECTED_BEHAVIOR = auto()


@dataclass
class Quaternion:
    """Quaternion representation for orientation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> Quaternion:
        """Create from numpy array."""
        return cls(w=arr[0], x=arr[1], y=arr[2], z=arr[3])
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create from Euler angles (in radians)."""
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )
    
    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw) in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def normalize(self) -> Quaternion:
        """Return normalized quaternion."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            return Quaternion()
        return Quaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)
    
    def __mul__(self, other: Quaternion) -> Quaternion:
        """Quaternion multiplication."""
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        )
    
    def inverse(self) -> Quaternion:
        """Return inverse quaternion (conjugate for unit quaternion)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)


@dataclass
class BatteryState:
    """Battery/energy state of an agent."""
    current_level: float  # 0-1 (percentage)
    voltage: float  # Volts
    current: float  # Amps (negative = discharging)
    temperature: float = 25.0  # Celsius
    
    # Derived values
    discharge_rate: float = 0.0  # percentage per second
    estimated_remaining: float = float('inf')  # seconds
    
    # Health indicators
    cycle_count: int = 0
    health_percentage: float = 100.0  # State of health
    
    def update_estimates(self, capacity_wh: float, power_consumption: float) -> None:
        """Update discharge rate and remaining time estimates."""
        if capacity_wh > 0 and power_consumption > 0:
            # Discharge rate in percentage per second
            self.discharge_rate = (power_consumption / capacity_wh) / 3600 * 100
            # Remaining time
            remaining_wh = capacity_wh * self.current_level
            self.estimated_remaining = remaining_wh / power_consumption * 3600
        else:
            self.discharge_rate = 0.0
            self.estimated_remaining = float('inf')


@dataclass
class Orientation:
    """Simple orientation representation (Euler angles)."""
    roll: float = 0.0   # radians
    pitch: float = 0.0  # radians
    yaw: float = 0.0    # radians
    
    def to_quaternion(self) -> Quaternion:
        """Convert to quaternion."""
        cr = np.cos(self.roll / 2)
        sr = np.sin(self.roll / 2)
        cp = np.cos(self.pitch / 2)
        sp = np.sin(self.pitch / 2)
        cy = np.cos(self.yaw / 2)
        sy = np.sin(self.yaw / 2)
        
        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )


@dataclass
class EnergyState:
    """Simplified energy state."""
    battery_level: float = 1.0  # 0-1
    battery_voltage: float = 4.2  # Volts
    power_consumption: float = 0.0  # Watts
    estimated_remaining_time: float = float('inf')  # seconds


@dataclass
class SensorHealth:
    """Health status of sensors."""
    gps_quality: float = 1.0  # 0-1
    imu_health: float = 1.0  # 0-1
    camera_health: float = 1.0  # 0-1
    lidar_health: float = 1.0  # 0-1
    comm_quality: float = 1.0  # 0-1
    
    # Specific sensor values
    gps_satellites: int = 0
    gps_hdop: float = 99.0  # Horizontal dilution of precision
    
    def get_overall_health(self) -> float:
        """Get overall sensor health score."""
        weights = {
            'gps': 0.3,
            'imu': 0.3,
            'comm': 0.2,
            'camera': 0.1,
            'lidar': 0.1,
        }
        score = (
            weights['gps'] * self.gps_quality +
            weights['imu'] * self.imu_health +
            weights['comm'] * self.comm_quality +
            weights['camera'] * self.camera_health +
            weights['lidar'] * self.lidar_health
        )
        return score


@dataclass
class ActuatorHealth:
    """Health status of actuators/motors."""
    motor_health: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    response_time: float = 0.0  # seconds
    thrust_efficiency: float = 1.0  # 0-1
    
    def get_overall_health(self) -> float:
        """Get overall actuator health score."""
        if not self.motor_health:
            return 1.0
        return min(self.motor_health) * self.thrust_efficiency


@dataclass
class HealthIndicators:
    """Complete health indicators for an agent."""
    battery: BatteryState = field(default_factory=lambda: BatteryState(1.0, 3.7, 0.0))
    sensors: SensorHealth = field(default_factory=SensorHealth)
    actuators: ActuatorHealth = field(default_factory=ActuatorHealth)
    
    # System-level health
    cpu_usage: float = 0.0  # percentage
    memory_usage: float = 0.0  # percentage
    temperature: float = 25.0  # Celsius
    
    def get_overall_health(self) -> float:
        """Get overall system health score."""
        # Weighted combination
        battery_score = self.battery.current_level
        sensor_score = self.sensors.get_overall_health()
        actuator_score = self.actuators.get_overall_health()
        
        # Temperature penalty
        temp_penalty = 0.0
        if self.temperature > 60:
            temp_penalty = (self.temperature - 60) / 40  # Penalty above 60C
        
        overall = (
            0.4 * battery_score +
            0.3 * sensor_score +
            0.3 * actuator_score -
            temp_penalty
        )
        return max(0.0, min(1.0, overall))
    
    def is_critical(self) -> bool:
        """Check if any health indicator is critical."""
        return (
            self.battery.current_level < 0.1 or
            self.sensors.get_overall_health() < 0.3 or
            self.actuators.get_overall_health() < 0.3 or
            self.temperature > 80
        )


@dataclass
class AnomalyInfo:
    """Information about detected anomalies."""
    detected: bool = False
    type: AnomalyType = AnomalyType.NONE
    severity: float = 0.0  # 0-1
    timestamp: float = 0.0
    description: str = ""
    
    # Additional context
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    threshold: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class BehavioralState:
    """Behavioral/task state of an agent."""
    current_task_id: Optional[str] = None
    task_progress: float = 0.0  # 0-1
    current_mode: AgentMode = AgentMode.IDLE
    priority: float = 0.5  # 0-1
    
    # Target information
    target_position: Optional[Vector3] = None
    target_velocity: Optional[Vector3] = None
    
    # Timing
    task_start_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    
    # Coordination
    formation_position: Optional[int] = None  # Position in formation
    leader_id: Optional[str] = None  # ID of formation leader
    
    def get_time_in_task(self) -> float:
        """Get time spent in current task."""
        if self.task_start_time is None:
            return 0.0
        return time.time() - self.task_start_time


@dataclass
class StateEstimate:
    """
    Estimated state with uncertainty quantification.
    
    Used by the Digital Twin for state estimation and prediction.
    """
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    
    # Uncertainty (covariance matrices)
    position_covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.01
    )
    velocity_covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.1
    )
    
    # Full state covariance (6x6 for position + velocity)
    state_covariance: np.ndarray = field(
        default_factory=lambda: np.eye(6) * 0.01
    )
    
    # Confidence score
    confidence: float = 1.0  # 0-1
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_state_vector(self) -> np.ndarray:
        """Convert to state vector [x, y, z, vx, vy, vz]."""
        return np.concatenate([
            self.position.to_array(),
            self.velocity.to_array()
        ])
    
    @classmethod
    def from_state_vector(
        cls,
        state: np.ndarray,
        covariance: Optional[np.ndarray] = None
    ) -> StateEstimate:
        """Create from state vector."""
        estimate = cls(
            position=Vector3.from_array(state[:3]),
            velocity=Vector3.from_array(state[3:6])
        )
        if covariance is not None:
            estimate.state_covariance = covariance
            estimate.position_covariance = covariance[:3, :3]
            estimate.velocity_covariance = covariance[3:6, 3:6]
        return estimate
    
    def get_uncertainty_ellipsoid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get uncertainty ellipsoid parameters.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors) for position uncertainty
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.position_covariance)
        return eigenvalues, eigenvectors
    
    def mahalanobis_distance(self, other_position: Vector3) -> float:
        """Compute Mahalanobis distance to another position."""
        diff = (other_position - self.position).to_array()
        try:
            inv_cov = np.linalg.inv(self.position_covariance)
            return np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            return np.linalg.norm(diff)


@dataclass 
class PredictedTrajectory:
    """Predicted trajectory with uncertainty envelopes."""
    timestamps: List[float] = field(default_factory=list)
    positions: List[Vector3] = field(default_factory=list)
    velocities: List[Vector3] = field(default_factory=list)
    
    # Uncertainty envelopes (one CollisionEnvelope per timestep)
    uncertainty_envelopes: List[CollisionEnvelope] = field(default_factory=list)
    
    # Confidence decreases over prediction horizon
    confidence_values: List[float] = field(default_factory=list)
    
    @property
    def horizon(self) -> float:
        """Get prediction horizon in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    def get_position_at_time(self, t: float) -> Optional[Vector3]:
        """Interpolate position at given time."""
        if len(self.timestamps) < 2:
            return None
        
        # Find bracketing indices
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= t <= self.timestamps[i + 1]:
                # Linear interpolation
                alpha = (t - self.timestamps[i]) / (self.timestamps[i + 1] - self.timestamps[i])
                p1 = self.positions[i].to_array()
                p2 = self.positions[i + 1].to_array()
                return Vector3.from_array(p1 + alpha * (p2 - p1))
        
        return None


class AgentState(BaseModel):
    """
    Complete state representation for a heterogeneous agent.
    
    This is the main state class combining all state components.
    Used for both observed and estimated states.
    """
    
    # Identity
    agent_id: str
    platform_type: PlatformType
    platform_id: str  # Reference to platform specification
    
    # Kinematic state
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Timing
    timestamp: float = Field(default_factory=time.time)
    
    # Energy
    energy_level: float = 1.0  # 0-1
    
    # Network
    network_quality: float = 1.0  # 0-1
    
    # Priority
    priority: float = 0.5  # 0-1
    
    # Mode
    mode: AgentMode = AgentMode.IDLE
    
    # Target
    target_position: Optional[Tuple[float, float, float]] = None
    
    # Task
    current_task_id: Optional[str] = None
    task_progress: float = 0.0
    
    # Safety envelope parameters
    safety_envelope: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # Semi-axes
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        arbitrary_types_allowed = True
    
    @property
    def position_vec(self) -> Vector3:
        """Get position as Vector3."""
        return Vector3(*self.position)
    
    @property
    def velocity_vec(self) -> Vector3:
        """Get velocity as Vector3."""
        return Vector3(*self.velocity)
    
    @property
    def orientation_quat(self) -> Quaternion:
        """Get orientation as Quaternion."""
        return Quaternion(*self.orientation)
    
    @property
    def target_position_vec(self) -> Optional[Vector3]:
        """Get target position as Vector3."""
        if self.target_position is None:
            return None
        return Vector3(*self.target_position)
    
    def distance_to(self, other: AgentState) -> float:
        """Compute Euclidean distance to another agent."""
        return (self.position_vec - other.position_vec).norm()
    
    def ellipsoidal_distance_to(
        self,
        other: AgentState,
        combined_envelope: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute ellipsoidal distance to another agent.
        
        The ellipsoidal distance accounts for different safety envelopes.
        Distance < 1 means collision/overlap.
        
        Args:
            other: Other agent state
            combined_envelope: Combined envelope matrix Θ(i,j). If None, uses simple sum.
            
        Returns:
            Ellipsoidal distance (< 1 means collision)
        """
        diff = (self.position_vec - other.position_vec).to_array()
        
        if combined_envelope is None:
            # Simple sum of semi-axes
            a = self.safety_envelope[0] + other.safety_envelope[0]
            b = self.safety_envelope[1] + other.safety_envelope[1]
            c = self.safety_envelope[2] + other.safety_envelope[2]
            combined_envelope = np.diag([a**2, b**2, c**2])
        
        try:
            inv_envelope = np.linalg.inv(combined_envelope)
            return np.sqrt(diff @ inv_envelope @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean
            return np.linalg.norm(diff) / np.mean(self.safety_envelope)
    
    def to_broadcast_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for broadcasting."""
        return {
            "agent_id": self.agent_id,
            "platform_type": self.platform_type,
            "position": self.position,
            "velocity": self.velocity,
            "target_position": self.target_position,
            "priority": self.priority,
            "network_quality": self.network_quality,
            "energy_level": self.energy_level,
            "safety_envelope": self.safety_envelope,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_broadcast_dict(cls, data: Dict[str, Any], platform_id: str = "") -> AgentState:
        """Create from broadcast dictionary."""
        return cls(
            agent_id=data["agent_id"],
            platform_type=PlatformType(data["platform_type"]),
            platform_id=platform_id,
            position=tuple(data["position"]),
            velocity=tuple(data["velocity"]),
            target_position=tuple(data["target_position"]) if data.get("target_position") else None,
            priority=data.get("priority", 0.5),
            network_quality=data.get("network_quality", 1.0),
            energy_level=data.get("energy_level", 1.0),
            safety_envelope=tuple(data.get("safety_envelope", (0.1, 0.1, 0.1))),
            mode=AgentMode(data.get("mode", AgentMode.IDLE.value)),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class FleetState:
    """
    Collective state of the entire heterogeneous fleet.
    
    Aggregates individual agent states and provides fleet-level metrics.
    """
    
    # Individual agent states
    agents: Dict[str, AgentState] = field(default_factory=dict)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    # Fleet-level metrics (computed)
    _centroid: Optional[Vector3] = None
    _spread: Optional[float] = None
    _connectivity_graph: Optional[Dict[str, List[str]]] = None
    
    def add_agent(self, state: AgentState) -> None:
        """Add or update an agent state."""
        self.agents[state.agent_id] = state
        self._invalidate_cache()
    
    def remove_agent(self, agent_id: str) -> Optional[AgentState]:
        """Remove an agent and return its last state."""
        state = self.agents.pop(agent_id, None)
        self._invalidate_cache()
        return state
    
    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get state for a specific agent."""
        return self.agents.get(agent_id)
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._centroid = None
        self._spread = None
        self._connectivity_graph = None
    
    @property
    def num_agents(self) -> int:
        """Get number of agents in fleet."""
        return len(self.agents)
    
    @property
    def centroid(self) -> Vector3:
        """Compute fleet centroid."""
        if self._centroid is None:
            if self.num_agents == 0:
                self._centroid = Vector3()
            else:
                positions = [s.position_vec.to_array() for s in self.agents.values()]
                mean_pos = np.mean(positions, axis=0)
                self._centroid = Vector3.from_array(mean_pos)
        return self._centroid
    
    @property
    def spread(self) -> float:
        """Compute fleet spread (standard deviation of positions)."""
        if self._spread is None:
            if self.num_agents < 2:
                self._spread = 0.0
            else:
                positions = [s.position_vec.to_array() for s in self.agents.values()]
                self._spread = float(np.std(positions))
        return self._spread
    
    def get_agents_by_type(self, platform_type: PlatformType) -> List[AgentState]:
        """Get all agents of a specific platform type."""
        return [
            state for state in self.agents.values()
            if state.platform_type == platform_type
        ]
    
    def get_agents_in_mode(self, mode: AgentMode) -> List[AgentState]:
        """Get all agents in a specific mode."""
        return [
            state for state in self.agents.values()
            if state.mode == mode
        ]
    
    def get_agents_in_region(
        self,
        center: Vector3,
        radius: float
    ) -> List[AgentState]:
        """Get all agents within a spherical region."""
        results = []
        for state in self.agents.values():
            dist = (state.position_vec - center).norm()
            if dist <= radius:
                results.append(state)
        return results
    
    def compute_connectivity_graph(
        self,
        comm_range: float
    ) -> Dict[str, List[str]]:
        """
        Compute connectivity graph based on communication range.
        
        Args:
            comm_range: Communication range in meters
            
        Returns:
            Dictionary mapping agent_id to list of connected agent_ids
        """
        if self._connectivity_graph is not None:
            return self._connectivity_graph
        
        graph = {agent_id: [] for agent_id in self.agents}
        
        agent_list = list(self.agents.values())
        for i, state_i in enumerate(agent_list):
            for j, state_j in enumerate(agent_list):
                if i >= j:
                    continue
                
                dist = state_i.distance_to(state_j)
                if dist <= comm_range:
                    graph[state_i.agent_id].append(state_j.agent_id)
                    graph[state_j.agent_id].append(state_i.agent_id)
        
        self._connectivity_graph = graph
        return graph
    
    def compute_density_map(
        self,
        grid_resolution: float,
        bounds: Tuple[Vector3, Vector3]
    ) -> np.ndarray:
        """
        Compute density map of agent positions.
        
        Args:
            grid_resolution: Grid cell size in meters
            bounds: (min_corner, max_corner) of the region
            
        Returns:
            3D numpy array with agent counts per cell
        """
        min_corner, max_corner = bounds
        
        # Compute grid dimensions
        dims = [
            int(np.ceil((max_corner.x - min_corner.x) / grid_resolution)),
            int(np.ceil((max_corner.y - min_corner.y) / grid_resolution)),
            int(np.ceil((max_corner.z - min_corner.z) / grid_resolution)),
        ]
        dims = [max(1, d) for d in dims]
        
        density = np.zeros(dims)
        
        for state in self.agents.values():
            pos = state.position_vec
            
            # Compute grid indices
            ix = int((pos.x - min_corner.x) / grid_resolution)
            iy = int((pos.y - min_corner.y) / grid_resolution)
            iz = int((pos.z - min_corner.z) / grid_resolution)
            
            # Bounds check
            if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
                density[ix, iy, iz] += 1
        
        return density
    
    def get_minimum_separation(self) -> Tuple[float, str, str]:
        """
        Find minimum separation between any two agents.
        
        Returns:
            Tuple of (minimum_distance, agent_id_1, agent_id_2)
        """
        min_dist = float('inf')
        min_pair = ("", "")
        
        agent_list = list(self.agents.values())
        for i, state_i in enumerate(agent_list):
            for j, state_j in enumerate(agent_list):
                if i >= j:
                    continue
                
                dist = state_i.distance_to(state_j)
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (state_i.agent_id, state_j.agent_id)
        
        return min_dist, min_pair[0], min_pair[1]
    
    def get_collision_risks(
        self,
        threshold: float = 1.5
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs with ellipsoidal distance below threshold.
        
        Args:
            threshold: Distance threshold (< 1.0 is collision)
            
        Returns:
            List of (agent_id_1, agent_id_2, ellipsoidal_distance) tuples
        """
        risks = []
        
        agent_list = list(self.agents.values())
        for i, state_i in enumerate(agent_list):
            for j, state_j in enumerate(agent_list):
                if i >= j:
                    continue
                
                ellip_dist = state_i.ellipsoidal_distance_to(state_j)
                if ellip_dist < threshold:
                    risks.append((state_i.agent_id, state_j.agent_id, ellip_dist))
        
        # Sort by distance (most critical first)
        risks.sort(key=lambda x: x[2])
        return risks
    
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get summary statistics of fleet state."""
        if self.num_agents == 0:
            return {"num_agents": 0}
        
        # Count by type
        type_counts = {}
        for state in self.agents.values():
            ptype = state.platform_type.value if hasattr(state.platform_type, 'value') else str(state.platform_type)
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        # Count by mode
        mode_counts = {}
        for state in self.agents.values():
            mode = state.mode.name if hasattr(state.mode, 'name') else str(state.mode)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Energy statistics
        energy_levels = [s.energy_level for s in self.agents.values()]
        
        # Network statistics
        network_qualities = [s.network_quality for s in self.agents.values()]
        
        min_sep, agent1, agent2 = self.get_minimum_separation()
        
        return {
            "num_agents": self.num_agents,
            "type_counts": type_counts,
            "mode_counts": mode_counts,
            "centroid": (self.centroid.x, self.centroid.y, self.centroid.z),
            "spread": self.spread,
            "energy_mean": np.mean(energy_levels),
            "energy_min": min(energy_levels),
            "network_mean": np.mean(network_qualities),
            "network_min": min(network_qualities),
            "min_separation": min_sep,
            "closest_pair": (agent1, agent2),
            "timestamp": self.timestamp,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agents": {
                agent_id: state.dict()
                for agent_id, state in self.agents.items()
            },
            "timestamp": self.timestamp,
            "summary": self.get_fleet_summary(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FleetState:
        """Create from dictionary."""
        fleet = cls(timestamp=data.get("timestamp", time.time()))
        
        for agent_id, state_data in data.get("agents", {}).items():
            state = AgentState(**state_data)
            fleet.add_agent(state)
        
        return fleet


@dataclass
class AgentDigitalTwinState:
    """
    Complete digital twin state for an agent.
    
    Extends AgentState with estimation, prediction, health, and anomaly data.
    Used by the Hierarchical Digital Twin Architecture (HDTA).
    """
    
    # Identity
    agent_id: str
    platform_type: PlatformType
    platform_spec: Optional[PlatformSpecification] = None
    
    # Observed state (from physical agent)
    observed_state: Optional[AgentState] = None
    
    # Estimated state (filtered)
    estimated_state: Optional[StateEstimate] = None
    
    # Predicted trajectory
    prediction: Optional[PredictedTrajectory] = None
    
    # Health indicators
    health: HealthIndicators = field(default_factory=HealthIndicators)
    
    # Behavioral state
    behavior: BehavioralState = field(default_factory=BehavioralState)
    
    # Anomaly detection
    anomalies: AnomalyInfo = field(default_factory=AnomalyInfo)
    
    # Synchronization info
    last_sync_time: float = 0.0
    sync_latency: float = 0.0
    data_freshness: float = 1.0  # 0-1, decreases with age
    
    def update_from_observation(
        self,
        observation: AgentState,
        estimator_covariance: Optional[np.ndarray] = None
    ) -> None:
        """
        Update digital twin from new observation.
        
        Args:
            observation: New observed state
            estimator_covariance: State estimator covariance (if available)
        """
        current_time = time.time()
        
        # Update observed state
        self.observed_state = observation
        
        # Update sync info
        self.sync_latency = current_time - observation.timestamp
        self.last_sync_time = current_time
        self.data_freshness = 1.0
        
        # Update estimated state
        if self.estimated_state is None:
            self.estimated_state = StateEstimate(
                position=observation.position_vec,
                velocity=observation.velocity_vec,
                timestamp=observation.timestamp
            )
        else:
            # Simple update (would be replaced by Kalman filter in full implementation)
            self.estimated_state.position = observation.position_vec
            self.estimated_state.velocity = observation.velocity_vec
            self.estimated_state.timestamp = observation.timestamp
        
        if estimator_covariance is not None:
            self.estimated_state.state_covariance = estimator_covariance
        
        # Update behavioral state
        self.behavior.current_mode = observation.mode
        self.behavior.target_position = observation.target_position_vec
        self.behavior.priority = observation.priority
        
        if observation.current_task_id != self.behavior.current_task_id:
            self.behavior.current_task_id = observation.current_task_id
            self.behavior.task_start_time = current_time
        
        self.behavior.task_progress = observation.task_progress
        
        # Update health (energy)
        self.health.battery.current_level = observation.energy_level
    
    def decay_freshness(self, dt: float, decay_rate: float = 0.1) -> None:
        """Decay data freshness over time."""
        self.data_freshness *= np.exp(-decay_rate * dt)
    
    def check_anomalies(self) -> None:
        """Check for anomalies based on current state."""
        if self.observed_state is None or self.estimated_state is None:
            return
        
        # Check trajectory deviation
        predicted_pos = self.estimated_state.position
        observed_pos = self.observed_state.position_vec
        deviation = (predicted_pos - observed_pos).norm()
        
        # Threshold based on uncertainty
        threshold = 3 * np.sqrt(np.trace(self.estimated_state.position_covariance))
        
        if deviation > threshold:
            self.anomalies = AnomalyInfo(
                detected=True,
                type=AnomalyType.TRAJECTORY_DEVIATION,
                severity=min(1.0, deviation / threshold - 1.0),
                description=f"Position deviation: {deviation:.3f}m (threshold: {threshold:.3f}m)",
                expected_value=predicted_pos,
                actual_value=observed_pos,
                threshold=threshold
            )
        
        # Check battery critical
        elif self.health.battery.current_level < 0.1:
            self.anomalies = AnomalyInfo(
                detected=True,
                type=AnomalyType.BATTERY_CRITICAL,
                severity=1.0 - self.health.battery.current_level * 10,
                description=f"Battery critical: {self.health.battery.current_level*100:.1f}%"
            )
        
        # Check network quality
        elif self.observed_state.network_quality < 0.3:
            self.anomalies = AnomalyInfo(
                detected=True,
                type=AnomalyType.NETWORK_DEGRADED,
                severity=1.0 - self.observed_state.network_quality / 0.3,
                description=f"Network degraded: {self.observed_state.network_quality*100:.1f}%"
            )
        
        else:
            self.anomalies = AnomalyInfo(detected=False)
