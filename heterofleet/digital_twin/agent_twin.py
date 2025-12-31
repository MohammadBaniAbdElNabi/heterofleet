"""
Agent-level Digital Twin for individual robot state management.

Maintains real-time state representation, prediction models,
and historical data for each agent.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.core.state import AgentState, OperationalMode


class TwinStatus(Enum):
    """Status of digital twin synchronization."""
    SYNCHRONIZED = auto()
    STALE = auto()
    DISCONNECTED = auto()
    INITIALIZING = auto()


@dataclass
class AgentTwinState:
    """Complete state representation for agent twin."""
    
    # Identity
    agent_id: str = ""
    platform_type: PlatformType = PlatformType.MICRO_UAV
    
    # Kinematic state
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    acceleration: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    orientation: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))  # roll, pitch, yaw
    angular_velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    
    # Energy state
    battery_level: float = 1.0  # 0-1
    battery_voltage: float = 4.2
    power_consumption: float = 0.0  # Watts
    estimated_flight_time: float = 0.0  # seconds
    
    # Operational state
    mode: OperationalMode = OperationalMode.IDLE
    current_task_id: Optional[str] = None
    target_position: Optional[Vector3] = None
    
    # Health metrics
    motor_health: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    sensor_health: Dict[str, float] = field(default_factory=dict)
    communication_quality: float = 1.0
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    update_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "platform_type": self.platform_type.name,
            "position": [self.position.x, self.position.y, self.position.z],
            "velocity": [self.velocity.x, self.velocity.y, self.velocity.z],
            "battery_level": self.battery_level,
            "mode": self.mode.name,
            "current_task": self.current_task_id,
            "timestamp": self.timestamp,
        }
    
    def copy(self) -> AgentTwinState:
        """Create a deep copy of the state."""
        state = AgentTwinState(
            agent_id=self.agent_id,
            platform_type=self.platform_type,
            position=Vector3(self.position.x, self.position.y, self.position.z),
            velocity=Vector3(self.velocity.x, self.velocity.y, self.velocity.z),
            acceleration=Vector3(self.acceleration.x, self.acceleration.y, self.acceleration.z),
            orientation=Vector3(self.orientation.x, self.orientation.y, self.orientation.z),
            angular_velocity=Vector3(self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z),
            battery_level=self.battery_level,
            battery_voltage=self.battery_voltage,
            power_consumption=self.power_consumption,
            estimated_flight_time=self.estimated_flight_time,
            mode=self.mode,
            current_task_id=self.current_task_id,
            target_position=Vector3(self.target_position.x, self.target_position.y, self.target_position.z) if self.target_position else None,
            motor_health=self.motor_health.copy(),
            sensor_health=self.sensor_health.copy(),
            communication_quality=self.communication_quality,
            timestamp=self.timestamp,
            update_count=self.update_count,
        )
        return state


class StatePredictor:
    """
    State prediction model for agent twin.
    
    Uses kinematic models and historical data for prediction.
    """
    
    def __init__(self, model_type: str = "kinematic"):
        """
        Initialize state predictor.
        
        Args:
            model_type: Type of prediction model ("kinematic", "learned")
        """
        self.model_type = model_type
        self._history: deque = deque(maxlen=100)
    
    def add_observation(self, state: AgentTwinState) -> None:
        """Add state observation for learning."""
        self._history.append(state.copy())
    
    def predict(self, current_state: AgentTwinState, dt: float) -> AgentTwinState:
        """
        Predict future state.
        
        Args:
            current_state: Current state
            dt: Time horizon for prediction
            
        Returns:
            Predicted state
        """
        if self.model_type == "kinematic":
            return self._predict_kinematic(current_state, dt)
        else:
            return self._predict_kinematic(current_state, dt)
    
    def _predict_kinematic(self, state: AgentTwinState, dt: float) -> AgentTwinState:
        """Simple kinematic prediction."""
        predicted = state.copy()
        
        # Position prediction: p = p0 + v*dt + 0.5*a*dt^2
        predicted.position = Vector3(
            state.position.x + state.velocity.x * dt + 0.5 * state.acceleration.x * dt * dt,
            state.position.y + state.velocity.y * dt + 0.5 * state.acceleration.y * dt * dt,
            state.position.z + state.velocity.z * dt + 0.5 * state.acceleration.z * dt * dt
        )
        
        # Velocity prediction: v = v0 + a*dt
        predicted.velocity = Vector3(
            state.velocity.x + state.acceleration.x * dt,
            state.velocity.y + state.acceleration.y * dt,
            state.velocity.z + state.acceleration.z * dt
        )
        
        # Battery prediction (simple linear model)
        energy_rate = state.power_consumption / (state.battery_voltage * 3.7)  # Approximate Ah
        predicted.battery_level = max(0, state.battery_level - energy_rate * dt / 3600)
        
        predicted.timestamp = state.timestamp + dt
        
        return predicted
    
    def predict_trajectory(
        self,
        current_state: AgentTwinState,
        horizon: float,
        dt: float = 0.1
    ) -> List[AgentTwinState]:
        """Predict trajectory over time horizon."""
        trajectory = [current_state.copy()]
        
        state = current_state.copy()
        for t in np.arange(dt, horizon + dt, dt):
            state = self.predict(state, dt)
            trajectory.append(state)
        
        return trajectory


class AgentTwin:
    """
    Digital twin for a single agent.
    
    Maintains synchronized state with physical agent,
    provides prediction, and tracks historical data.
    """
    
    def __init__(
        self,
        agent_id: str,
        platform_spec: PlatformSpecification,
        history_size: int = 1000,
        stale_threshold: float = 1.0
    ):
        """
        Initialize agent twin.
        
        Args:
            agent_id: Agent identifier
            platform_spec: Platform specification
            history_size: Size of state history buffer
            stale_threshold: Time threshold for stale status (seconds)
        """
        self.agent_id = agent_id
        self.platform_spec = platform_spec
        self.stale_threshold = stale_threshold
        
        # Current state
        self._state = AgentTwinState(
            agent_id=agent_id,
            platform_type=platform_spec.platform_type
        )
        
        # State history
        self._history: deque = deque(maxlen=history_size)
        
        # Prediction
        self._predictor = StatePredictor()
        
        # Synchronization
        self._status = TwinStatus.INITIALIZING
        self._last_sync_time = 0.0
        self._sync_count = 0
        
        # Callbacks
        self._state_callbacks: List[Callable[[AgentTwinState], None]] = []
        self._anomaly_callbacks: List[Callable[[str, Any], None]] = []
        
        # Anomaly detection
        self._anomaly_thresholds = {
            "position_jump": 1.0,  # meters
            "velocity_spike": 5.0,  # m/s
            "battery_drop": 0.1,   # 10%
        }
    
    @property
    def state(self) -> AgentTwinState:
        """Get current state."""
        return self._state
    
    @property
    def status(self) -> TwinStatus:
        """Get synchronization status."""
        if self._status == TwinStatus.INITIALIZING:
            return self._status
        
        time_since_sync = time.time() - self._last_sync_time
        
        if time_since_sync > self.stale_threshold * 10:
            return TwinStatus.DISCONNECTED
        elif time_since_sync > self.stale_threshold:
            return TwinStatus.STALE
        else:
            return TwinStatus.SYNCHRONIZED
    
    def update_state(self, new_state: AgentTwinState) -> None:
        """
        Update twin with new state from physical agent.
        
        Args:
            new_state: New state observation
        """
        # Check for anomalies
        self._check_anomalies(new_state)
        
        # Store in history
        self._history.append(self._state.copy())
        
        # Update current state
        self._state = new_state
        self._state.update_count = self._sync_count
        
        # Update predictor
        self._predictor.add_observation(new_state)
        
        # Update sync status
        self._last_sync_time = time.time()
        self._sync_count += 1
        self._status = TwinStatus.SYNCHRONIZED
        
        # Trigger callbacks
        for callback in self._state_callbacks:
            callback(new_state)
    
    def update_from_agent_state(self, agent_state: AgentState) -> None:
        """Update from core AgentState object."""
        # Handle orientation as tuple (quaternion w,x,y,z) - convert to euler approx
        if isinstance(agent_state.orientation, tuple):
            # Simple approximation - just use zero for now
            orientation = Vector3(0, 0, 0)
        else:
            orientation = Vector3(
                getattr(agent_state.orientation, 'roll', 0),
                getattr(agent_state.orientation, 'pitch', 0),
                getattr(agent_state.orientation, 'yaw', 0)
            )
        
        # Handle energy level
        if hasattr(agent_state, 'energy') and agent_state.energy:
            battery = agent_state.energy.battery_level
        else:
            battery = getattr(agent_state, 'energy_level', 1.0)
        
        # Handle position/velocity as tuples
        if isinstance(agent_state.position, tuple):
            pos = Vector3(*agent_state.position)
        else:
            pos = agent_state.position
            
        if isinstance(agent_state.velocity, tuple):
            vel = Vector3(*agent_state.velocity)
        else:
            vel = agent_state.velocity
        
        new_state = AgentTwinState(
            agent_id=self.agent_id,
            platform_type=self.platform_spec.platform_type,
            position=pos,
            velocity=vel,
            orientation=orientation,
            battery_level=battery,
            mode=agent_state.mode,
            timestamp=agent_state.timestamp,
        )
        self.update_state(new_state)
    
    def predict_state(self, dt: float) -> AgentTwinState:
        """Predict state at future time."""
        return self._predictor.predict(self._state, dt)
    
    def predict_trajectory(self, horizon: float, dt: float = 0.1) -> List[AgentTwinState]:
        """Predict trajectory over time horizon."""
        return self._predictor.predict_trajectory(self._state, horizon, dt)
    
    def get_history(self, duration: float = None) -> List[AgentTwinState]:
        """Get state history."""
        if duration is None:
            return list(self._history)
        
        cutoff = time.time() - duration
        return [s for s in self._history if s.timestamp >= cutoff]
    
    def register_state_callback(self, callback: Callable[[AgentTwinState], None]) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)
    
    def register_anomaly_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Register callback for anomaly detection."""
        self._anomaly_callbacks.append(callback)
    
    def _check_anomalies(self, new_state: AgentTwinState) -> None:
        """Check for anomalies in state update."""
        if self._status == TwinStatus.INITIALIZING:
            return
        
        old = self._state
        
        # Position jump
        pos_diff = (new_state.position - old.position).norm()
        if pos_diff > self._anomaly_thresholds["position_jump"]:
            self._trigger_anomaly("position_jump", {
                "old": old.position, "new": new_state.position, "diff": pos_diff
            })
        
        # Velocity spike
        vel_diff = (new_state.velocity - old.velocity).norm()
        if vel_diff > self._anomaly_thresholds["velocity_spike"]:
            self._trigger_anomaly("velocity_spike", {
                "old": old.velocity, "new": new_state.velocity, "diff": vel_diff
            })
        
        # Battery drop
        battery_diff = old.battery_level - new_state.battery_level
        if battery_diff > self._anomaly_thresholds["battery_drop"]:
            self._trigger_anomaly("battery_drop", {
                "old": old.battery_level, "new": new_state.battery_level, "diff": battery_diff
            })
    
    def _trigger_anomaly(self, anomaly_type: str, data: Dict[str, Any]) -> None:
        """Trigger anomaly callbacks."""
        logger.warning(f"Anomaly detected for {self.agent_id}: {anomaly_type}")
        for callback in self._anomaly_callbacks:
            callback(anomaly_type, data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get twin statistics."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.name,
            "sync_count": self._sync_count,
            "last_sync": self._last_sync_time,
            "history_size": len(self._history),
            "battery_level": self._state.battery_level,
            "mode": self._state.mode.name,
        }
    
    def estimate_remaining_energy(self) -> float:
        """Estimate remaining energy in Wh."""
        battery_wh = self.platform_spec.battery_capacity_wh
        return battery_wh * self._state.battery_level
    
    def estimate_range(self) -> float:
        """Estimate remaining range in meters."""
        remaining_energy = self.estimate_remaining_energy()
        cruise_power = self.platform_spec.cruise_power_consumption
        
        if cruise_power <= 0:
            return float('inf')
        
        remaining_time = remaining_energy / cruise_power * 3600  # seconds
        cruise_speed = self.platform_spec.max_velocity * 0.7  # Assume 70% of max
        
        return remaining_time * cruise_speed
