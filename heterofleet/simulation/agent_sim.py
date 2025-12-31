"""
Simulated Agent for HeteroFleet.

Implements agent dynamics, sensor simulation, and
physical behavior modeling.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.core.state import AgentState, OperationalMode, Orientation, EnergyState
from heterofleet.simulation.environment import SimulationEnvironment


@dataclass
class AgentDynamics:
    """Dynamic parameters for agent simulation."""
    
    # Mass and inertia
    mass: float = 0.5  # kg
    moment_of_inertia: Vector3 = field(default_factory=lambda: Vector3(0.01, 0.01, 0.02))
    
    # Thrust and drag
    max_thrust: float = 10.0  # N
    drag_coefficient: float = 0.5
    lift_coefficient: float = 0.0  # For fixed-wing
    
    # Motor dynamics
    motor_time_constant: float = 0.05  # seconds
    num_motors: int = 4
    
    # Response characteristics
    velocity_time_constant: float = 0.3  # seconds
    angular_time_constant: float = 0.1  # seconds
    
    # Noise
    position_noise_std: float = 0.01  # meters
    velocity_noise_std: float = 0.02  # m/s
    
    @classmethod
    def for_platform(cls, platform_type: PlatformType) -> AgentDynamics:
        """Create dynamics for a platform type."""
        if platform_type in (PlatformType.MICRO_UAV, PlatformType.SMALL_UAV):
            return cls(
                mass=0.03 if platform_type == PlatformType.MICRO_UAV else 0.5,
                max_thrust=0.5 if platform_type == PlatformType.MICRO_UAV else 15.0,
                velocity_time_constant=0.2,
            )
        elif platform_type in (PlatformType.MEDIUM_UAV, PlatformType.LARGE_UAV):
            return cls(
                mass=3.0 if platform_type == PlatformType.MEDIUM_UAV else 15.0,
                max_thrust=60.0 if platform_type == PlatformType.MEDIUM_UAV else 300.0,
                velocity_time_constant=0.5,
            )
        elif platform_type in (PlatformType.SMALL_UGV, PlatformType.MEDIUM_UGV, PlatformType.LARGE_UGV):
            return cls(
                mass=1.0 if platform_type == PlatformType.SMALL_UGV else 5.0,
                max_thrust=20.0,
                velocity_time_constant=0.4,
                drag_coefficient=0.8,
            )
        else:
            return cls()


@dataclass
class SensorReading:
    """A sensor reading."""
    sensor_type: str
    value: Any
    timestamp: float
    noise_level: float = 0.0


class SensorSimulator:
    """
    Simulates sensor readings for an agent.
    """
    
    def __init__(
        self,
        platform_spec: PlatformSpecification,
        environment: SimulationEnvironment
    ):
        """
        Initialize sensor simulator.
        
        Args:
            platform_spec: Agent platform specification
            environment: Simulation environment
        """
        self.platform_spec = platform_spec
        self.environment = environment
        
        # Sensor parameters
        self._sensor_rates = {
            "position": 100.0,  # Hz
            "velocity": 100.0,
            "orientation": 200.0,
            "range": 20.0,
            "battery": 1.0,
        }
        
        self._last_readings: Dict[str, float] = {}
    
    def get_position_reading(
        self,
        true_position: Vector3,
        timestamp: float
    ) -> SensorReading:
        """Get simulated position sensor reading."""
        # Add noise
        noise = Vector3(
            np.random.normal(0, 0.01),
            np.random.normal(0, 0.01),
            np.random.normal(0, 0.02)
        )
        
        measured = Vector3(
            true_position.x + noise.x,
            true_position.y + noise.y,
            true_position.z + noise.z
        )
        
        return SensorReading(
            sensor_type="position",
            value=measured,
            timestamp=timestamp,
            noise_level=0.01
        )
    
    def get_velocity_reading(
        self,
        true_velocity: Vector3,
        timestamp: float
    ) -> SensorReading:
        """Get simulated velocity sensor reading."""
        noise = Vector3(
            np.random.normal(0, 0.02),
            np.random.normal(0, 0.02),
            np.random.normal(0, 0.02)
        )
        
        measured = Vector3(
            true_velocity.x + noise.x,
            true_velocity.y + noise.y,
            true_velocity.z + noise.z
        )
        
        return SensorReading(
            sensor_type="velocity",
            value=measured,
            timestamp=timestamp,
            noise_level=0.02
        )
    
    def get_range_readings(
        self,
        position: Vector3,
        orientation: Orientation,
        timestamp: float
    ) -> List[SensorReading]:
        """Get simulated range sensor readings."""
        readings = []
        sensor_range = self.platform_spec.sensor_range
        
        # Multi-ranger: front, back, left, right, up
        directions = [
            ("front", Vector3(1, 0, 0)),
            ("back", Vector3(-1, 0, 0)),
            ("left", Vector3(0, 1, 0)),
            ("right", Vector3(0, -1, 0)),
            ("up", Vector3(0, 0, 1)),
        ]
        
        for name, direction in directions:
            # Rotate direction by yaw
            yaw = orientation.yaw
            rotated = Vector3(
                direction.x * np.cos(yaw) - direction.y * np.sin(yaw),
                direction.x * np.sin(yaw) + direction.y * np.cos(yaw),
                direction.z
            )
            
            # Raycast
            hit = self.environment.raycast(position, rotated, sensor_range)
            
            if hit:
                hit_point, _ = hit
                distance = (hit_point - position).norm()
            else:
                distance = sensor_range
            
            # Add noise
            distance += np.random.normal(0, 0.02)
            distance = max(0.01, min(distance, sensor_range))
            
            readings.append(SensorReading(
                sensor_type=f"range_{name}",
                value=distance,
                timestamp=timestamp,
                noise_level=0.02
            ))
        
        return readings
    
    def get_battery_reading(
        self,
        true_level: float,
        timestamp: float
    ) -> SensorReading:
        """Get simulated battery sensor reading."""
        # Small noise and quantization
        noise = np.random.normal(0, 0.005)
        measured = np.clip(true_level + noise, 0, 1)
        
        return SensorReading(
            sensor_type="battery",
            value=measured,
            timestamp=timestamp,
            noise_level=0.005
        )


class SimulatedAgent:
    """
    Simulated agent for testing.
    
    Models agent dynamics, sensors, and behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        platform_spec: PlatformSpecification,
        environment: SimulationEnvironment,
        initial_position: Vector3 = None
    ):
        """
        Initialize simulated agent.
        
        Args:
            agent_id: Agent identifier
            platform_spec: Platform specification
            environment: Simulation environment
            initial_position: Starting position
        """
        self.agent_id = agent_id
        self.platform_spec = platform_spec
        self.environment = environment
        
        # Dynamics
        self.dynamics = AgentDynamics.for_platform(platform_spec.platform_type)
        
        # Sensors
        self.sensors = SensorSimulator(platform_spec, environment)
        
        # State
        self._position = initial_position or Vector3(0, 0, 0)
        self._velocity = Vector3(0, 0, 0)
        self._acceleration = Vector3(0, 0, 0)
        self._orientation = Orientation(0, 0, 0)
        self._angular_velocity = Vector3(0, 0, 0)
        
        # Energy
        self._battery_level = 1.0
        self._power_consumption = 0.0
        
        # Control
        self._target_velocity = Vector3(0, 0, 0)
        self._target_position: Optional[Vector3] = None
        self._mode = OperationalMode.IDLE
        
        # Time
        self._time = 0.0
    
    @property
    def position(self) -> Vector3:
        return self._position
    
    @property
    def velocity(self) -> Vector3:
        return self._velocity
    
    @property
    def orientation(self) -> Orientation:
        return self._orientation
    
    @property
    def battery_level(self) -> float:
        return self._battery_level
    
    @property
    def mode(self) -> OperationalMode:
        return self._mode
    
    def set_target_velocity(self, velocity: Vector3) -> None:
        """Set target velocity command."""
        # Clamp to max velocity
        max_vel = self.platform_spec.max_velocity
        vel_norm = velocity.norm()
        
        if vel_norm > max_vel:
            scale = max_vel / vel_norm
            velocity = Vector3(
                velocity.x * scale,
                velocity.y * scale,
                velocity.z * scale
            )
        
        self._target_velocity = velocity
        
        if velocity.norm() > 0.01:
            self._mode = OperationalMode.NAVIGATING
    
    def set_target_position(self, position: Vector3) -> None:
        """Set target position for navigation."""
        self._target_position = position
        self._mode = OperationalMode.NAVIGATING
    
    def update(self, dt: float) -> None:
        """
        Update agent state.
        
        Args:
            dt: Time step in seconds
        """
        self._time += dt
        
        # Position control if target set
        if self._target_position is not None:
            self._update_position_control()
        
        # Velocity control
        self._update_velocity(dt)
        
        # Update position
        self._update_position(dt)
        
        # Update orientation
        self._update_orientation(dt)
        
        # Update energy
        self._update_energy(dt)
        
        # Check bounds
        self._enforce_bounds()
        
        # Check for collisions
        self._check_collisions()
    
    def _update_position_control(self) -> None:
        """Update velocity target from position target."""
        if self._target_position is None:
            return
        
        # Simple P controller
        error = self._target_position - self._position
        dist = error.norm()
        
        if dist < 0.1:
            # Reached target
            self._target_velocity = Vector3(0, 0, 0)
            self._target_position = None
            self._mode = OperationalMode.IDLE
            return
        
        # Proportional gain
        kp = 1.0
        max_vel = self.platform_spec.max_velocity
        
        target_vel = Vector3(
            error.x * kp,
            error.y * kp,
            error.z * kp
        )
        
        # Clamp
        vel_norm = target_vel.norm()
        if vel_norm > max_vel:
            scale = max_vel / vel_norm
            target_vel = Vector3(
                target_vel.x * scale,
                target_vel.y * scale,
                target_vel.z * scale
            )
        
        self._target_velocity = target_vel
    
    def _update_velocity(self, dt: float) -> None:
        """Update velocity toward target."""
        tau = self.dynamics.velocity_time_constant
        
        # First-order dynamics
        alpha = dt / (tau + dt)
        
        self._velocity = Vector3(
            self._velocity.x + alpha * (self._target_velocity.x - self._velocity.x),
            self._velocity.y + alpha * (self._target_velocity.y - self._velocity.y),
            self._velocity.z + alpha * (self._target_velocity.z - self._velocity.z)
        )
        
        # Add wind effect for aerial platforms
        if self.platform_spec.platform_type.name.startswith("AERIAL"):
            wind = self.environment.get_wind_at(self._position)
            wind_effect = 0.1  # Wind influence factor
            self._velocity = Vector3(
                self._velocity.x + wind.x * wind_effect,
                self._velocity.y + wind.y * wind_effect,
                self._velocity.z + wind.z * wind_effect
            )
        
        # Add noise
        noise_std = self.dynamics.velocity_noise_std
        self._velocity = Vector3(
            self._velocity.x + np.random.normal(0, noise_std),
            self._velocity.y + np.random.normal(0, noise_std),
            self._velocity.z + np.random.normal(0, noise_std)
        )
    
    def _update_position(self, dt: float) -> None:
        """Update position from velocity."""
        self._position = Vector3(
            self._position.x + self._velocity.x * dt,
            self._position.y + self._velocity.y * dt,
            self._position.z + self._velocity.z * dt
        )
        
        # Add position noise
        noise_std = self.dynamics.position_noise_std
        self._position = Vector3(
            self._position.x + np.random.normal(0, noise_std),
            self._position.y + np.random.normal(0, noise_std),
            self._position.z + np.random.normal(0, noise_std)
        )
    
    def _update_orientation(self, dt: float) -> None:
        """Update orientation based on velocity direction."""
        # Yaw toward velocity direction
        vel_xy = np.sqrt(self._velocity.x**2 + self._velocity.y**2)
        
        if vel_xy > 0.1:
            target_yaw = np.arctan2(self._velocity.y, self._velocity.x)
            
            # Smooth yaw transition
            tau = self.dynamics.angular_time_constant
            alpha = dt / (tau + dt)
            
            # Handle wraparound
            yaw_diff = target_yaw - self._orientation.yaw
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            
            new_yaw = self._orientation.yaw + alpha * yaw_diff
            self._orientation = Orientation(0, 0, new_yaw)
    
    def _update_energy(self, dt: float) -> None:
        """Update energy consumption."""
        # Base consumption
        base_power = self.platform_spec.cruise_power_consumption
        
        # Velocity-dependent consumption
        vel_mag = self._velocity.norm()
        max_vel = self.platform_spec.max_velocity
        vel_factor = 1.0 + 2.0 * (vel_mag / max_vel) ** 2
        
        self._power_consumption = base_power * vel_factor
        
        # Consume energy
        battery_wh = self.platform_spec.battery_capacity_wh
        energy_consumed = self._power_consumption * dt / 3600  # Wh
        
        self._battery_level -= energy_consumed / battery_wh
        self._battery_level = max(0, self._battery_level)
        
        # Emergency mode if low battery
        if self._battery_level < 0.1:
            self._mode = OperationalMode.EMERGENCY
    
    def _enforce_bounds(self) -> None:
        """Enforce world bounds."""
        bounds_min = self.environment.config.bounds_min
        bounds_max = self.environment.config.bounds_max
        
        self._position = Vector3(
            np.clip(self._position.x, bounds_min.x, bounds_max.x),
            np.clip(self._position.y, bounds_min.y, bounds_max.y),
            np.clip(self._position.z, bounds_min.z, bounds_max.z)
        )
    
    def _check_collisions(self) -> None:
        """Check for collisions with obstacles."""
        radius = max(self.platform_spec.collision_envelope.semi_axes)
        
        for obstacle in self.environment.get_obstacles_near(self._position, radius + 1.0):
            if obstacle.distance_to_point(self._position) < radius:
                # Collision - push out
                diff = self._position - obstacle.position
                dist = diff.norm()
                if dist > 0.01:
                    push_dir = Vector3(diff.x / dist, diff.y / dist, diff.z / dist)
                    push_dist = radius - obstacle.distance_to_point(self._position) + 0.1
                    self._position = Vector3(
                        self._position.x + push_dir.x * push_dist,
                        self._position.y + push_dir.y * push_dist,
                        self._position.z + push_dir.z * push_dist
                    )
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return AgentState(
            agent_id=self.agent_id,
            platform_type=self.platform_spec.platform_type,
            platform_id=self.platform_spec.platform_id,
            position=(self._position.x, self._position.y, self._position.z),
            velocity=(self._velocity.x, self._velocity.y, self._velocity.z),
            orientation=(1.0, 0.0, 0.0, 0.0),  # Quaternion w,x,y,z - simplified
            mode=self._mode,
            energy_level=self._battery_level,
            timestamp=self._time,
        )
    
    def get_sensor_readings(self) -> Dict[str, SensorReading]:
        """Get all sensor readings."""
        return {
            "position": self.sensors.get_position_reading(self._position, self._time),
            "velocity": self.sensors.get_velocity_reading(self._velocity, self._time),
            "battery": self.sensors.get_battery_reading(self._battery_level, self._time),
        }
