"""
Heterogeneous agent implementation for autonomous vehicle swarms.

This module defines:
- HeterogeneousAgent: Main agent class with full capabilities
- AgentController: Low-level control interface
- AgentInterface: Hardware abstraction layer

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger

from heterofleet.core.platform import (
    PlatformType,
    PlatformSpecification,
    PlatformRegistry,
    PlatformFactory,
    Vector3,
    CollisionEnvelope,
    DomainType,
)
from heterofleet.core.state import (
    AgentState,
    AgentMode,
    StateEstimate,
    HealthIndicators,
    BehavioralState,
    AnomalyInfo,
    Quaternion,
    BatteryState,
)
from heterofleet.core.message import (
    MessageRouter,
    MessageType,
    StateBroadcastMessage,
    TrajectoryShareMessage,
    SafetyAlertMessage,
    EmergencyOverrideMessage,
)


class ControlMode(Enum):
    """Control mode for agent."""
    POSITION = auto()      # Position setpoint control
    VELOCITY = auto()      # Velocity setpoint control
    ATTITUDE = auto()      # Attitude (for UAVs) control
    TRAJECTORY = auto()    # Trajectory tracking
    MANUAL = auto()        # Manual/RC control


@dataclass
class ControlCommand:
    """Control command for agent execution."""
    mode: ControlMode = ControlMode.POSITION
    
    # Position mode
    target_position: Optional[Vector3] = None
    
    # Velocity mode
    target_velocity: Optional[Vector3] = None
    
    # Attitude mode (UAVs)
    target_attitude: Optional[Quaternion] = None
    target_thrust: Optional[float] = None
    
    # Trajectory mode
    trajectory_points: Optional[List[Tuple[float, Vector3, Vector3]]] = None  # (time, pos, vel)
    
    # Common
    yaw: float = 0.0  # Target yaw for position/velocity modes
    duration: float = 0.0  # Command duration (0 = until next command)
    
    # Safety
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    
    timestamp: float = field(default_factory=time.time)


class AgentInterface(ABC):
    """
    Abstract interface for agent hardware/simulation.
    
    Provides hardware abstraction for different platforms.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the agent hardware/simulation."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the agent."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass
    
    @abstractmethod
    def get_state(self) -> AgentState:
        """Get current state from sensors."""
        pass
    
    @abstractmethod
    def send_command(self, command: ControlCommand) -> bool:
        """Send control command to actuators."""
        pass
    
    @abstractmethod
    def arm(self) -> bool:
        """Arm the agent (enable motors)."""
        pass
    
    @abstractmethod
    def disarm(self) -> bool:
        """Disarm the agent (disable motors)."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Execute emergency stop."""
        pass
    
    @property
    @abstractmethod
    def is_armed(self) -> bool:
        """Check if agent is armed."""
        pass


class SimulatedAgentInterface(AgentInterface):
    """
    Simulated agent interface for testing.
    
    Provides a simple physics simulation for agent dynamics.
    """
    
    def __init__(
        self,
        agent_id: str,
        platform_spec: PlatformSpecification,
        initial_position: Vector3 = None,
        dt: float = 0.01
    ):
        """
        Initialize simulated agent.
        
        Args:
            agent_id: Agent identifier
            platform_spec: Platform specification
            initial_position: Initial position
            dt: Simulation timestep
        """
        self.agent_id = agent_id
        self.platform_spec = platform_spec
        self.dt = dt
        
        # State
        self._position = initial_position or Vector3(0, 0, 0)
        self._velocity = Vector3(0, 0, 0)
        self._acceleration = Vector3(0, 0, 0)
        self._orientation = Quaternion()
        self._angular_velocity = Vector3(0, 0, 0)
        
        # Energy
        self._battery_level = 1.0
        self._energy_capacity = platform_spec.energy_properties.capacity
        
        # Status
        self._connected = False
        self._armed = False
        self._last_update = time.time()
        
        # Current command
        self._current_command: Optional[ControlCommand] = None
        
        # Dynamics parameters
        dyn = platform_spec.dynamic_properties
        self._max_velocity = dyn.max_velocity
        self._max_acceleration = dyn.max_acceleration
        self._response_time = dyn.response_time
        
        # Noise parameters
        self._position_noise_std = 0.01  # 1cm
        self._velocity_noise_std = 0.05  # 5cm/s
    
    def connect(self) -> bool:
        """Connect to simulated agent."""
        self._connected = True
        self._last_update = time.time()
        logger.info(f"Simulated agent {self.agent_id} connected")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from simulated agent."""
        self._connected = False
        self._armed = False
        logger.info(f"Simulated agent {self.agent_id} disconnected")
    
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def is_armed(self) -> bool:
        return self._armed
    
    def arm(self) -> bool:
        if not self._connected:
            return False
        self._armed = True
        logger.debug(f"Agent {self.agent_id} armed")
        return True
    
    def disarm(self) -> bool:
        self._armed = False
        self._current_command = None
        logger.debug(f"Agent {self.agent_id} disarmed")
        return True
    
    def emergency_stop(self) -> bool:
        self._armed = False
        self._current_command = None
        self._velocity = Vector3(0, 0, 0)
        self._acceleration = Vector3(0, 0, 0)
        logger.warning(f"Agent {self.agent_id} emergency stop!")
        return True
    
    def get_state(self) -> AgentState:
        """Get current simulated state with noise."""
        self._update_simulation()
        
        # Add measurement noise
        noisy_position = Vector3(
            self._position.x + np.random.normal(0, self._position_noise_std),
            self._position.y + np.random.normal(0, self._position_noise_std),
            self._position.z + np.random.normal(0, self._position_noise_std)
        )
        
        noisy_velocity = Vector3(
            self._velocity.x + np.random.normal(0, self._velocity_noise_std),
            self._velocity.y + np.random.normal(0, self._velocity_noise_std),
            self._velocity.z + np.random.normal(0, self._velocity_noise_std)
        )
        
        # Determine mode
        if not self._armed:
            mode = AgentMode.IDLE
        elif self._current_command is not None:
            mode = AgentMode.TRANSIT
        else:
            mode = AgentMode.HOVER if self.platform_spec.platform_type.is_aerial else AgentMode.IDLE
        
        return AgentState(
            agent_id=self.agent_id,
            platform_type=self.platform_spec.platform_type,
            platform_id=self.platform_spec.platform_id,
            position=(noisy_position.x, noisy_position.y, noisy_position.z),
            velocity=(noisy_velocity.x, noisy_velocity.y, noisy_velocity.z),
            acceleration=(self._acceleration.x, self._acceleration.y, self._acceleration.z),
            orientation=(self._orientation.w, self._orientation.x, 
                        self._orientation.y, self._orientation.z),
            energy_level=self._battery_level,
            network_quality=0.9 + np.random.uniform(-0.1, 0.1),
            mode=mode,
            target_position=(self._current_command.target_position.x,
                           self._current_command.target_position.y,
                           self._current_command.target_position.z) 
                          if self._current_command and self._current_command.target_position else None,
            safety_envelope=self.platform_spec.physical_properties.collision_envelope.semi_axes,
            timestamp=time.time()
        )
    
    def send_command(self, command: ControlCommand) -> bool:
        """Execute control command."""
        if not self._armed:
            logger.warning(f"Cannot send command to disarmed agent {self.agent_id}")
            return False
        
        self._current_command = command
        return True
    
    def _update_simulation(self) -> None:
        """Update simulation state."""
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        if not self._armed or self._current_command is None:
            return
        
        cmd = self._current_command
        
        if cmd.mode == ControlMode.POSITION and cmd.target_position is not None:
            # Simple position control with velocity limiting
            error = cmd.target_position - self._position
            distance = error.norm()
            
            if distance < 0.01:
                # Arrived
                self._velocity = Vector3(0, 0, 0)
                self._acceleration = Vector3(0, 0, 0)
            else:
                # Compute desired velocity (proportional control)
                kp = 1.0 / self._response_time
                desired_velocity = error * kp
                
                # Limit velocity
                speed = desired_velocity.norm()
                max_speed = self._max_velocity.norm()
                if speed > max_speed:
                    desired_velocity = desired_velocity * (max_speed / speed)
                
                # Simple first-order response
                alpha = min(1.0, dt / self._response_time)
                self._velocity = self._velocity * (1 - alpha) + desired_velocity * alpha
        
        elif cmd.mode == ControlMode.VELOCITY and cmd.target_velocity is not None:
            # Velocity control
            desired_velocity = cmd.target_velocity
            
            # Limit velocity
            speed = desired_velocity.norm()
            max_speed = self._max_velocity.norm()
            if speed > max_speed:
                desired_velocity = desired_velocity * (max_speed / speed)
            
            alpha = min(1.0, dt / self._response_time)
            self._velocity = self._velocity * (1 - alpha) + desired_velocity * alpha
        
        # Integrate position
        self._position = self._position + self._velocity * dt
        
        # Enforce altitude constraints for aerial vehicles
        if self.platform_spec.platform_type.is_aerial:
            alt_min, alt_max = self.platform_spec.domain_constraints.altitude_range
            self._position.z = max(alt_min, min(alt_max, self._position.z))
        
        # Enforce ground constraint for ground vehicles
        if self.platform_spec.platform_type.is_ground:
            self._position.z = 0.0
            self._velocity.z = 0.0
        
        # Update battery (simplified model)
        energy = self.platform_spec.energy_properties
        if self.platform_spec.platform_type.is_aerial:
            power = energy.hover_power + 0.5 * (energy.cruise_power - energy.hover_power) * (self._velocity.norm() / self._max_velocity.norm())
        else:
            power = energy.cruise_power * (self._velocity.norm() / self._max_velocity.norm() + 0.1)
        
        energy_consumed = power * dt / 3600  # Wh
        self._battery_level -= energy_consumed / self._energy_capacity
        self._battery_level = max(0.0, self._battery_level)
    
    def set_position(self, position: Vector3) -> None:
        """Teleport agent to position (for testing)."""
        self._position = position
    
    def set_battery_level(self, level: float) -> None:
        """Set battery level (for testing)."""
        self._battery_level = max(0.0, min(1.0, level))


class AgentController:
    """
    Low-level controller for agent motion.
    
    Implements control laws for position, velocity, and trajectory tracking.
    """
    
    def __init__(
        self,
        platform_spec: PlatformSpecification,
        control_frequency: float = 50.0
    ):
        """
        Initialize controller.
        
        Args:
            platform_spec: Platform specification
            control_frequency: Control loop frequency in Hz
        """
        self.platform_spec = platform_spec
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        
        # Dynamics constraints
        dyn = platform_spec.dynamic_properties
        self.max_velocity = dyn.max_velocity.to_array()
        self.max_acceleration = dyn.max_acceleration.to_array()
        
        # PID gains (position control)
        self.kp_pos = np.array([2.0, 2.0, 2.0])  # Proportional
        self.ki_pos = np.array([0.1, 0.1, 0.1])  # Integral
        self.kd_pos = np.array([1.0, 1.0, 1.0])  # Derivative
        
        # Velocity control gains
        self.kp_vel = np.array([3.0, 3.0, 3.0])
        
        # State
        self._integral_error = np.zeros(3)
        self._last_error = np.zeros(3)
        self._last_position = None
        
        # Trajectory tracking
        self._trajectory: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None
        self._trajectory_start_time: Optional[float] = None
    
    def reset(self) -> None:
        """Reset controller state."""
        self._integral_error = np.zeros(3)
        self._last_error = np.zeros(3)
        self._last_position = None
        self._trajectory = None
        self._trajectory_start_time = None
    
    def set_trajectory(
        self,
        trajectory: List[Tuple[float, Vector3, Vector3]]
    ) -> None:
        """
        Set trajectory for tracking.
        
        Args:
            trajectory: List of (time, position, velocity) tuples
        """
        self._trajectory = [
            (t, pos.to_array(), vel.to_array())
            for t, pos, vel in trajectory
        ]
        self._trajectory_start_time = time.time()
    
    def compute_position_control(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute velocity command for position control.
        
        Uses PID control with velocity feedforward.
        
        Args:
            current_position: Current position [x, y, z]
            current_velocity: Current velocity [vx, vy, vz]
            target_position: Target position [x, y, z]
            target_velocity: Target velocity (feedforward) [vx, vy, vz]
            
        Returns:
            Commanded velocity [vx, vy, vz]
        """
        # Position error
        error = target_position - current_position
        
        # PID terms
        p_term = self.kp_pos * error
        
        self._integral_error += error * self.dt
        # Anti-windup
        self._integral_error = np.clip(self._integral_error, -1.0, 1.0)
        i_term = self.ki_pos * self._integral_error
        
        d_term = self.kd_pos * (error - self._last_error) / self.dt if self._last_error is not None else 0
        self._last_error = error
        
        # Compute commanded velocity
        cmd_velocity = p_term + i_term + d_term
        
        # Add feedforward
        if target_velocity is not None:
            cmd_velocity += target_velocity
        
        # Saturate velocity
        cmd_velocity = np.clip(cmd_velocity, -self.max_velocity, self.max_velocity)
        
        return cmd_velocity
    
    def compute_velocity_control(
        self,
        current_velocity: np.ndarray,
        target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration command for velocity control.
        
        Args:
            current_velocity: Current velocity [vx, vy, vz]
            target_velocity: Target velocity [vx, vy, vz]
            
        Returns:
            Commanded acceleration [ax, ay, az]
        """
        error = target_velocity - current_velocity
        cmd_acceleration = self.kp_vel * error
        
        # Saturate acceleration
        cmd_acceleration = np.clip(cmd_acceleration, -self.max_acceleration, self.max_acceleration)
        
        return cmd_acceleration
    
    def compute_trajectory_tracking(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_time: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute control for trajectory tracking.
        
        Args:
            current_position: Current position
            current_velocity: Current velocity
            current_time: Current time (if None, uses wall clock)
            
        Returns:
            Tuple of (commanded velocity, trajectory_complete)
        """
        if self._trajectory is None or len(self._trajectory) == 0:
            return np.zeros(3), True
        
        if current_time is None:
            current_time = time.time()
        
        # Compute trajectory time
        t = current_time - self._trajectory_start_time if self._trajectory_start_time else 0
        
        # Find current segment
        if t >= self._trajectory[-1][0]:
            # Trajectory complete, hold final position
            target_pos = self._trajectory[-1][1]
            target_vel = np.zeros(3)
            complete = True
        else:
            # Interpolate trajectory
            for i in range(len(self._trajectory) - 1):
                t0, pos0, vel0 = self._trajectory[i]
                t1, pos1, vel1 = self._trajectory[i + 1]
                
                if t0 <= t <= t1:
                    # Linear interpolation
                    alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
                    target_pos = pos0 + alpha * (pos1 - pos0)
                    target_vel = vel0 + alpha * (vel1 - vel0)
                    break
            else:
                # Before trajectory start
                target_pos = self._trajectory[0][1]
                target_vel = self._trajectory[0][2]
            complete = False
        
        # Compute control
        cmd_velocity = self.compute_position_control(
            current_position, current_velocity,
            target_pos, target_vel
        )
        
        return cmd_velocity, complete
    
    def compute_smooth_stop(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        stop_position: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute smooth stopping trajectory.
        
        Args:
            current_position: Current position
            current_velocity: Current velocity
            stop_position: Desired stop position (if None, stops at current + decel distance)
            
        Returns:
            Commanded velocity for smooth stop
        """
        speed = np.linalg.norm(current_velocity)
        
        if speed < 0.01:
            # Already stopped
            if stop_position is not None:
                return self.compute_position_control(
                    current_position, current_velocity, stop_position
                )
            return np.zeros(3)
        
        # Compute stopping distance
        max_decel = np.min(self.max_acceleration)
        stop_distance = speed**2 / (2 * max_decel)
        
        # Compute deceleration
        direction = current_velocity / speed
        if stop_position is not None:
            # Adjust for target position
            to_target = stop_position - current_position
            dist_to_target = np.linalg.norm(to_target)
            
            if dist_to_target < stop_distance:
                # Need to decelerate harder
                decel = speed**2 / (2 * max(dist_to_target, 0.01))
                cmd_velocity = current_velocity - decel * direction * self.dt
            else:
                # Normal approach
                cmd_velocity = self.compute_position_control(
                    current_position, current_velocity, stop_position
                )
        else:
            # Just decelerate
            decel = max_decel
            cmd_velocity = current_velocity - decel * direction * self.dt
        
        return cmd_velocity


class HeterogeneousAgent:
    """
    Main class representing a heterogeneous autonomous agent.
    
    Integrates:
    - Platform specification
    - State management
    - Control interface
    - Communication
    - Coordination behaviors
    """
    
    def __init__(
        self,
        agent_id: str,
        platform_spec: PlatformSpecification,
        interface: Optional[AgentInterface] = None,
        control_frequency: float = 50.0
    ):
        """
        Initialize heterogeneous agent.
        
        Args:
            agent_id: Unique identifier
            platform_spec: Platform specification
            interface: Hardware/simulation interface (if None, creates simulated)
            control_frequency: Control loop frequency in Hz
        """
        self.agent_id = agent_id
        self.platform_spec = platform_spec
        self.platform_type = platform_spec.platform_type
        self.control_frequency = control_frequency
        
        # Create interface if not provided
        if interface is None:
            self.interface = SimulatedAgentInterface(
                agent_id=agent_id,
                platform_spec=platform_spec,
                dt=1.0/control_frequency
            )
        else:
            self.interface = interface
        
        # Controller
        self.controller = AgentController(platform_spec, control_frequency)
        
        # State
        self._state: Optional[AgentState] = None
        self._state_estimate: Optional[StateEstimate] = None
        self._health = HealthIndicators()
        self._behavior = BehavioralState()
        self._anomalies = AnomalyInfo()
        
        # Communication
        self._agent_id_int = hash(agent_id) % (2**16)  # Convert to int for messaging
        self.message_router = MessageRouter(self._agent_id_int)
        
        # Neighbors (received state broadcasts)
        self._neighbors: Dict[int, StateBroadcastMessage] = {}
        self._neighbor_trajectories: Dict[int, TrajectoryShareMessage] = {}
        
        # Register message handlers
        self._setup_message_handlers()
        
        # Control state
        self._target_position: Optional[Vector3] = None
        self._target_velocity: Optional[Vector3] = None
        self._mode = AgentMode.IDLE
        
        # Threading
        self._running = False
        self._control_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._state_callbacks: List[Callable[[AgentState], None]] = []
        self._safety_callbacks: List[Callable[[SafetyAlertMessage], None]] = []
        
        logger.info(f"Created agent {agent_id} ({platform_spec.platform_type.value})")
    
    def _setup_message_handlers(self) -> None:
        """Setup message handlers."""
        self.message_router.register_handler(
            MessageType.STATE_BROADCAST,
            self._handle_state_broadcast
        )
        self.message_router.register_handler(
            MessageType.TRAJECTORY_SHARE,
            self._handle_trajectory_share
        )
        self.message_router.register_handler(
            MessageType.SAFETY_ALERT,
            self._handle_safety_alert
        )
        self.message_router.register_handler(
            MessageType.COLLISION_WARNING,
            self._handle_safety_alert
        )
        self.message_router.register_handler(
            MessageType.EMERGENCY_OVERRIDE,
            self._handle_emergency_override
        )
        self.message_router.register_handler(
            MessageType.EMERGENCY_STOP,
            self._handle_emergency_stop
        )
    
    def _handle_state_broadcast(self, msg: StateBroadcastMessage) -> None:
        """Handle received state broadcast from neighbor."""
        self._neighbors[msg.header.sender_id] = msg
    
    def _handle_trajectory_share(self, msg: TrajectoryShareMessage) -> None:
        """Handle received trajectory from neighbor."""
        self._neighbor_trajectories[msg.header.sender_id] = msg
    
    def _handle_safety_alert(self, msg: SafetyAlertMessage) -> None:
        """Handle safety alert."""
        logger.warning(f"Safety alert: {msg.description}")
        for callback in self._safety_callbacks:
            callback(msg)
    
    def _handle_emergency_override(self, msg: EmergencyOverrideMessage) -> None:
        """Handle emergency override command."""
        logger.warning(f"Emergency override: {msg.override_type} - {msg.reason}")
        
        if not msg.target_agents or self._agent_id_int in msg.target_agents:
            if msg.override_type == "return_home":
                self.return_to_home()
            elif msg.override_type == "hover":
                self.hover()
            elif msg.override_type == "land":
                self.land()
    
    def _handle_emergency_stop(self, msg: EmergencyOverrideMessage) -> None:
        """Handle emergency stop command."""
        logger.error(f"EMERGENCY STOP: {msg.reason}")
        self.emergency_stop()
    
    # ========== Lifecycle Methods ==========
    
    def connect(self) -> bool:
        """Connect to agent hardware/simulation."""
        return self.interface.connect()
    
    def disconnect(self) -> None:
        """Disconnect from agent."""
        self.stop()
        self.interface.disconnect()
    
    def start(self) -> None:
        """Start agent control loops."""
        if self._running:
            return
        
        self._running = True
        
        # Start control thread
        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        self._control_thread.start()
        
        # Start state broadcast thread
        self._state_thread = threading.Thread(
            target=self._state_broadcast_loop,
            daemon=True
        )
        self._state_thread.start()
        
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self) -> None:
        """Stop agent control loops."""
        self._running = False
        
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    def arm(self) -> bool:
        """Arm the agent."""
        result = self.interface.arm()
        if result:
            self._mode = AgentMode.HOVER if self.platform_type.is_aerial else AgentMode.IDLE
        return result
    
    def disarm(self) -> bool:
        """Disarm the agent."""
        result = self.interface.disarm()
        if result:
            self._mode = AgentMode.IDLE
        return result
    
    def emergency_stop(self) -> bool:
        """Execute emergency stop."""
        self._mode = AgentMode.EMERGENCY
        self._target_position = None
        self._target_velocity = None
        return self.interface.emergency_stop()
    
    # ========== Control Methods ==========
    
    def goto(
        self,
        position: Vector3,
        velocity: float = None,
        yaw: float = None
    ) -> None:
        """
        Command agent to go to position.
        
        Args:
            position: Target position
            velocity: Max velocity (if None, uses platform max)
            yaw: Target yaw angle (if None, maintains current)
        """
        self._target_position = position
        self._mode = AgentMode.TRANSIT
        
        # Create and send command
        cmd = ControlCommand(
            mode=ControlMode.POSITION,
            target_position=position,
            yaw=yaw or 0.0,
            max_velocity=velocity
        )
        self.interface.send_command(cmd)
    
    def set_velocity(self, velocity: Vector3) -> None:
        """
        Set velocity setpoint.
        
        Args:
            velocity: Target velocity vector
        """
        self._target_velocity = velocity
        self._mode = AgentMode.TRANSIT
        
        cmd = ControlCommand(
            mode=ControlMode.VELOCITY,
            target_velocity=velocity
        )
        self.interface.send_command(cmd)
    
    def follow_trajectory(
        self,
        trajectory: List[Tuple[float, Vector3, Vector3]]
    ) -> None:
        """
        Follow a trajectory.
        
        Args:
            trajectory: List of (time, position, velocity) tuples
        """
        self._mode = AgentMode.TRANSIT
        self.controller.set_trajectory(trajectory)
        
        cmd = ControlCommand(
            mode=ControlMode.TRAJECTORY,
            trajectory_points=trajectory
        )
        self.interface.send_command(cmd)
    
    def hover(self) -> None:
        """Hover at current position (UAVs) or stop (UGVs)."""
        if self._state is not None:
            self._target_position = self._state.position_vec
        self._target_velocity = Vector3(0, 0, 0)
        self._mode = AgentMode.HOVER if self.platform_type.is_aerial else AgentMode.IDLE
        
        cmd = ControlCommand(
            mode=ControlMode.VELOCITY,
            target_velocity=Vector3(0, 0, 0)
        )
        self.interface.send_command(cmd)
    
    def land(self) -> None:
        """Land the agent (UAVs only)."""
        if not self.platform_type.is_aerial:
            logger.warning(f"Land command ignored for ground vehicle {self.agent_id}")
            return
        
        self._mode = AgentMode.LANDING
        
        # Simple landing: descend at slow rate
        if self._state is not None:
            landing_position = Vector3(
                self._state.position[0],
                self._state.position[1],
                0.0  # Ground level
            )
            self.goto(landing_position, velocity=0.3)
    
    def takeoff(self, altitude: float = 1.0) -> None:
        """
        Takeoff to specified altitude (UAVs only).
        
        Args:
            altitude: Target altitude in meters
        """
        if not self.platform_type.is_aerial:
            logger.warning(f"Takeoff command ignored for ground vehicle {self.agent_id}")
            return
        
        self._mode = AgentMode.TAKEOFF
        
        if self._state is not None:
            takeoff_position = Vector3(
                self._state.position[0],
                self._state.position[1],
                altitude
            )
            self.goto(takeoff_position, velocity=0.5)
    
    def return_to_home(self, home_position: Vector3 = None) -> None:
        """
        Return to home position.
        
        Args:
            home_position: Home position (if None, uses origin)
        """
        home = home_position or Vector3(0, 0, 0.5 if self.platform_type.is_aerial else 0)
        self._mode = AgentMode.RETURN
        self.goto(home)
    
    # ========== State Methods ==========
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        self._state = self.interface.get_state()
        return self._state
    
    @property
    def state(self) -> Optional[AgentState]:
        """Get cached state."""
        return self._state
    
    @property
    def position(self) -> Optional[Vector3]:
        """Get current position."""
        if self._state is None:
            return None
        return self._state.position_vec
    
    @property
    def velocity(self) -> Optional[Vector3]:
        """Get current velocity."""
        if self._state is None:
            return None
        return self._state.velocity_vec
    
    @property
    def mode(self) -> AgentMode:
        """Get current mode."""
        return self._mode
    
    @property
    def is_armed(self) -> bool:
        """Check if agent is armed."""
        return self.interface.is_armed
    
    @property
    def energy_level(self) -> float:
        """Get current energy level (0-1)."""
        if self._state is None:
            return 1.0
        return self._state.energy_level
    
    def get_neighbors(self) -> Dict[int, StateBroadcastMessage]:
        """Get current neighbor states."""
        return self._neighbors.copy()
    
    def get_neighbor_trajectories(self) -> Dict[int, TrajectoryShareMessage]:
        """Get current neighbor trajectories."""
        return self._neighbor_trajectories.copy()
    
    # ========== Callback Methods ==========
    
    def register_state_callback(self, callback: Callable[[AgentState], None]) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)
    
    def register_safety_callback(self, callback: Callable[[SafetyAlertMessage], None]) -> None:
        """Register callback for safety alerts."""
        self._safety_callbacks.append(callback)
    
    # ========== Internal Loops ==========
    
    def _control_loop(self) -> None:
        """Main control loop (runs in separate thread)."""
        dt = 1.0 / self.control_frequency
        
        while self._running:
            try:
                loop_start = time.time()
                
                # Get current state
                state = self.get_state()
                
                # Update behavior state
                self._update_behavior(state)
                
                # Notify callbacks
                for callback in self._state_callbacks:
                    try:
                        callback(state)
                    except Exception as e:
                        logger.error(f"State callback error: {e}")
                
                # Control based on mode
                if self._mode == AgentMode.TRANSIT and self._target_position is not None:
                    current_pos = np.array(state.position)
                    current_vel = np.array(state.velocity)
                    target_pos = self._target_position.to_array()
                    
                    # Check if arrived
                    distance = np.linalg.norm(target_pos - current_pos)
                    if distance < 0.05:  # 5cm threshold
                        self._mode = AgentMode.HOVER if self.platform_type.is_aerial else AgentMode.IDLE
                
                # Sleep for remaining time
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(dt)
    
    def _state_broadcast_loop(self) -> None:
        """State broadcast loop (runs in separate thread)."""
        broadcast_interval = 0.1  # 10 Hz
        
        while self._running:
            try:
                loop_start = time.time()
                
                if self._state is not None:
                    # Create and queue broadcast message
                    msg = self.message_router.create_state_broadcast(
                        platform_type=self.platform_type,
                        position=self._state.position_vec,
                        velocity=self._state.velocity_vec,
                        target_position=self._target_position or Vector3(0, 0, 0),
                        priority=self._state.priority,
                        battery_percent=self._state.energy_level,
                        network_quality=self._state.network_quality,
                        safety_envelope=self._state.safety_envelope,
                        is_active=self.is_armed,
                        is_emergency=self._mode == AgentMode.EMERGENCY,
                        has_task=self._behavior.current_task_id is not None
                    )
                    self.message_router.queue_message(msg)
                
                # Clean up old neighbor data
                self._cleanup_neighbors()
                
                # Sleep
                elapsed = time.time() - loop_start
                time.sleep(max(0, broadcast_interval - elapsed))
                
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
                time.sleep(broadcast_interval)
    
    def _update_behavior(self, state: AgentState) -> None:
        """Update behavioral state from agent state."""
        self._behavior.current_mode = self._mode
        self._behavior.target_position = self._target_position
        self._behavior.priority = state.priority
        
        # Check for task completion
        if self._behavior.current_task_id is not None:
            if self._mode in (AgentMode.IDLE, AgentMode.HOVER):
                self._behavior.task_progress = 1.0
    
    def _cleanup_neighbors(self, max_age: float = 1.0) -> None:
        """Remove stale neighbor data."""
        current_time = time.time() * 1000  # ms
        
        # Clean state broadcasts
        stale_ids = []
        for sender_id, msg in self._neighbors.items():
            age = (current_time - msg.header.timestamp_ms) / 1000
            if age > max_age:
                stale_ids.append(sender_id)
        
        for sender_id in stale_ids:
            del self._neighbors[sender_id]
        
        # Clean trajectories
        stale_ids = []
        for sender_id, msg in self._neighbor_trajectories.items():
            age = (current_time - msg.header.timestamp_ms) / 1000
            if age > max_age * 2:  # Longer timeout for trajectories
                stale_ids.append(sender_id)
        
        for sender_id in stale_ids:
            del self._neighbor_trajectories[sender_id]
    
    # ========== Utility Methods ==========
    
    def distance_to(self, other: Union[HeterogeneousAgent, Vector3]) -> float:
        """Compute distance to another agent or position."""
        if self._state is None:
            return float('inf')
        
        if isinstance(other, HeterogeneousAgent):
            if other.state is None:
                return float('inf')
            other_pos = other.position
        else:
            other_pos = other
        
        return (self.position - other_pos).norm()
    
    def get_safety_envelope(self) -> CollisionEnvelope:
        """Get collision envelope for this agent."""
        return self.platform_spec.physical_properties.collision_envelope
    
    def __repr__(self) -> str:
        return f"HeterogeneousAgent(id={self.agent_id}, type={self.platform_type.value}, mode={self._mode.name})"


class AgentFactory:
    """
    Factory for creating heterogeneous agents.
    
    Handles platform registry lookup and agent instantiation.
    """
    
    def __init__(self, registry: Optional[PlatformRegistry] = None):
        """
        Initialize factory.
        
        Args:
            registry: Platform registry (if None, uses global)
        """
        self.registry = registry or PlatformRegistry()
    
    def create_agent(
        self,
        agent_id: str,
        platform_id: str,
        initial_position: Vector3 = None,
        use_simulation: bool = True,
        **kwargs
    ) -> HeterogeneousAgent:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique agent identifier
            platform_id: Platform specification ID
            initial_position: Initial position
            use_simulation: Whether to use simulated interface
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            New HeterogeneousAgent instance
        """
        # Get platform specification
        spec = self.registry.get(platform_id)
        if spec is None:
            raise ValueError(f"Unknown platform: {platform_id}")
        
        # Create interface
        if use_simulation:
            interface = SimulatedAgentInterface(
                agent_id=agent_id,
                platform_spec=spec,
                initial_position=initial_position
            )
        else:
            # TODO: Create hardware interface based on platform type
            raise NotImplementedError("Hardware interfaces not yet implemented")
        
        # Create agent
        agent = HeterogeneousAgent(
            agent_id=agent_id,
            platform_spec=spec,
            interface=interface,
            **kwargs
        )
        
        return agent
    
    def create_fleet(
        self,
        platform_counts: Dict[str, int],
        positions: Optional[Dict[str, List[Vector3]]] = None,
        use_simulation: bool = True
    ) -> List[HeterogeneousAgent]:
        """
        Create a fleet of agents.
        
        Args:
            platform_counts: Dict mapping platform_id to count
            positions: Optional dict mapping platform_id to list of positions
            use_simulation: Whether to use simulated interfaces
            
        Returns:
            List of created agents
        """
        agents = []
        
        for platform_id, count in platform_counts.items():
            platform_positions = positions.get(platform_id, []) if positions else []
            
            for i in range(count):
                agent_id = f"{platform_id}_{i}"
                
                # Get position if provided
                if i < len(platform_positions):
                    initial_pos = platform_positions[i]
                else:
                    # Random position
                    initial_pos = Vector3(
                        np.random.uniform(-2, 2),
                        np.random.uniform(-2, 2),
                        0.5 if self.registry.get(platform_id).platform_type.is_aerial else 0.0
                    )
                
                agent = self.create_agent(
                    agent_id=agent_id,
                    platform_id=platform_id,
                    initial_position=initial_pos,
                    use_simulation=use_simulation
                )
                agents.append(agent)
        
        return agents
