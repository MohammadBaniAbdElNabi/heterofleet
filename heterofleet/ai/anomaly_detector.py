"""
Anomaly Detection for HeteroFleet.

Implements real-time anomaly detection for fleet operations
using statistical and ML-based methods.

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

from heterofleet.core.platform import Vector3
from heterofleet.core.state import AgentState


class AnomalyType(Enum):
    """Types of anomalies."""
    POSITION_JUMP = auto()
    VELOCITY_SPIKE = auto()
    BATTERY_ANOMALY = auto()
    COMMUNICATION_LOSS = auto()
    TRAJECTORY_DEVIATION = auto()
    FORMATION_BREAK = auto()
    SENSOR_FAULT = auto()
    MOTOR_FAILURE = auto()
    UNEXPECTED_STOP = auto()
    COLLISION_RISK = auto()


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class AnomalyEvent:
    """An detected anomaly event."""
    
    anomaly_id: str = ""
    anomaly_type: AnomalyType = AnomalyType.POSITION_JUMP
    severity: AnomalySeverity = AnomalySeverity.WARNING
    
    # Context
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Detection details
    expected_value: Any = None
    actual_value: Any = None
    deviation: float = 0.0
    confidence: float = 0.0
    
    # Description
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "type": self.anomaly_type.name,
            "severity": self.severity.name,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "deviation": self.deviation,
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detector."""
    
    # Window sizes
    history_window: int = 100
    short_window: int = 10
    
    # Thresholds
    position_jump_threshold: float = 1.0  # meters
    velocity_spike_threshold: float = 5.0  # m/s
    battery_drop_threshold: float = 0.1  # 10%
    trajectory_deviation_threshold: float = 2.0  # meters
    
    # Statistical parameters
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Learning rate for adaptive thresholds
    adaptation_rate: float = 0.01


class StatisticalDetector:
    """
    Statistical anomaly detector.
    
    Uses z-score, IQR, and moving average methods.
    """
    
    def __init__(self, window_size: int = 100, zscore_threshold: float = 3.0):
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self._history: deque = deque(maxlen=window_size)
        
        # Running statistics
        self._mean = 0.0
        self._variance = 0.0
        self._count = 0
    
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update with new value and check for anomaly.
        
        Returns:
            Tuple of (is_anomaly, zscore)
        """
        self._history.append(value)
        
        if len(self._history) < 5:
            return False, 0.0
        
        # Compute statistics
        values = np.array(self._history)
        mean = np.mean(values)
        std = np.std(values)
        
        if std < 1e-6:
            return False, 0.0
        
        zscore = abs(value - mean) / std
        is_anomaly = zscore > self.zscore_threshold
        
        return is_anomaly, zscore
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if len(self._history) < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        values = np.array(self._history)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }


class TrajectoryPredictor:
    """
    Simple trajectory predictor for deviation detection.
    """
    
    def __init__(self, horizon: float = 1.0):
        self.horizon = horizon
        self._last_position: Optional[Vector3] = None
        self._last_velocity: Optional[Vector3] = None
        self._last_time: float = 0.0
    
    def predict(self, dt: float) -> Optional[Vector3]:
        """Predict position at future time."""
        if self._last_position is None or self._last_velocity is None:
            return None
        
        return Vector3(
            self._last_position.x + self._last_velocity.x * dt,
            self._last_position.y + self._last_velocity.y * dt,
            self._last_position.z + self._last_velocity.z * dt
        )
    
    def update(self, position: Vector3, velocity: Vector3, timestamp: float) -> float:
        """
        Update with actual position and return prediction error.
        
        Returns:
            Prediction error (distance between predicted and actual)
        """
        error = 0.0
        
        if self._last_position is not None:
            dt = timestamp - self._last_time
            predicted = self.predict(dt)
            
            if predicted:
                error = (position - predicted).norm()
        
        self._last_position = position
        self._last_velocity = velocity
        self._last_time = timestamp
        
        return error


class AnomalyDetector:
    """
    Fleet-wide anomaly detector.
    
    Monitors agent states and detects various types of anomalies.
    """
    
    def __init__(self, config: AnomalyDetectorConfig = None):
        """
        Initialize anomaly detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or AnomalyDetectorConfig()
        
        # Per-agent detectors
        self._velocity_detectors: Dict[str, StatisticalDetector] = {}
        self._battery_detectors: Dict[str, StatisticalDetector] = {}
        self._trajectory_predictors: Dict[str, TrajectoryPredictor] = {}
        
        # State history
        self._last_states: Dict[str, AgentState] = {}
        self._state_history: Dict[str, deque] = {}
        
        # Anomaly history
        self._anomaly_history: List[AnomalyEvent] = []
        self._anomaly_count = 0
        
        # Callbacks
        self._callbacks: List[Callable[[AnomalyEvent], None]] = []
    
    def register_callback(self, callback: Callable[[AnomalyEvent], None]) -> None:
        """Register callback for anomaly events."""
        self._callbacks.append(callback)
    
    def process_state(self, state: AgentState) -> List[AnomalyEvent]:
        """
        Process agent state and detect anomalies.
        
        Args:
            state: Current agent state
            
        Returns:
            List of detected anomalies
        """
        agent_id = state.agent_id
        anomalies = []
        
        # Initialize detectors if needed
        if agent_id not in self._velocity_detectors:
            self._initialize_agent(agent_id)
        
        # Get previous state
        prev_state = self._last_states.get(agent_id)
        
        if prev_state is not None:
            # Check position jump
            anomaly = self._check_position_jump(agent_id, prev_state, state)
            if anomaly:
                anomalies.append(anomaly)
            
            # Check velocity spike
            anomaly = self._check_velocity_spike(agent_id, state)
            if anomaly:
                anomalies.append(anomaly)
            
            # Check battery anomaly
            anomaly = self._check_battery_anomaly(agent_id, prev_state, state)
            if anomaly:
                anomalies.append(anomaly)
            
            # Check trajectory deviation
            anomaly = self._check_trajectory_deviation(agent_id, state)
            if anomaly:
                anomalies.append(anomaly)
            
            # Check unexpected stop
            anomaly = self._check_unexpected_stop(agent_id, prev_state, state)
            if anomaly:
                anomalies.append(anomaly)
        
        # Update history
        self._last_states[agent_id] = state
        self._state_history[agent_id].append(state)
        
        # Update trajectory predictor
        self._trajectory_predictors[agent_id].update(
            state.position, state.velocity, state.timestamp
        )
        
        # Trigger callbacks
        for anomaly in anomalies:
            self._anomaly_history.append(anomaly)
            for callback in self._callbacks:
                callback(anomaly)
        
        return anomalies
    
    def _initialize_agent(self, agent_id: str) -> None:
        """Initialize detectors for an agent."""
        self._velocity_detectors[agent_id] = StatisticalDetector(
            window_size=self.config.history_window,
            zscore_threshold=self.config.zscore_threshold
        )
        self._battery_detectors[agent_id] = StatisticalDetector(
            window_size=self.config.history_window
        )
        self._trajectory_predictors[agent_id] = TrajectoryPredictor()
        self._state_history[agent_id] = deque(maxlen=self.config.history_window)
    
    def _check_position_jump(
        self,
        agent_id: str,
        prev_state: AgentState,
        state: AgentState
    ) -> Optional[AnomalyEvent]:
        """Check for sudden position jumps."""
        dt = state.timestamp - prev_state.timestamp
        if dt <= 0:
            return None
        
        distance = (state.position - prev_state.position).norm()
        expected_distance = prev_state.velocity.norm() * dt * 1.5  # 50% margin
        
        if distance > max(expected_distance, self.config.position_jump_threshold):
            self._anomaly_count += 1
            return AnomalyEvent(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.POSITION_JUMP,
                severity=AnomalySeverity.WARNING,
                agent_id=agent_id,
                expected_value=expected_distance,
                actual_value=distance,
                deviation=distance - expected_distance,
                confidence=min(1.0, (distance - expected_distance) / expected_distance) if expected_distance > 0 else 0.5,
                description=f"Position jumped {distance:.2f}m (expected {expected_distance:.2f}m)"
            )
        
        return None
    
    def _check_velocity_spike(
        self,
        agent_id: str,
        state: AgentState
    ) -> Optional[AnomalyEvent]:
        """Check for velocity spikes."""
        velocity_mag = state.velocity.norm()
        
        is_anomaly, zscore = self._velocity_detectors[agent_id].update(velocity_mag)
        
        if is_anomaly and velocity_mag > self.config.velocity_spike_threshold:
            stats = self._velocity_detectors[agent_id].get_statistics()
            self._anomaly_count += 1
            return AnomalyEvent(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.VELOCITY_SPIKE,
                severity=AnomalySeverity.WARNING,
                agent_id=agent_id,
                expected_value=stats["mean"],
                actual_value=velocity_mag,
                deviation=zscore,
                confidence=min(1.0, zscore / 5.0),
                description=f"Velocity spike: {velocity_mag:.2f} m/s (z-score: {zscore:.2f})"
            )
        
        return None
    
    def _check_battery_anomaly(
        self,
        agent_id: str,
        prev_state: AgentState,
        state: AgentState
    ) -> Optional[AnomalyEvent]:
        """Check for battery anomalies."""
        dt = state.timestamp - prev_state.timestamp
        if dt <= 0:
            return None
        
        battery_drop = prev_state.energy.battery_level - state.energy.battery_level
        battery_drop_rate = battery_drop / dt  # per second
        
        # Check for sudden large drop
        if battery_drop > self.config.battery_drop_threshold:
            self._anomaly_count += 1
            return AnomalyEvent(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.BATTERY_ANOMALY,
                severity=AnomalySeverity.CRITICAL if battery_drop > 0.2 else AnomalySeverity.WARNING,
                agent_id=agent_id,
                expected_value=0.001,  # Normal rate
                actual_value=battery_drop,
                deviation=battery_drop,
                confidence=min(1.0, battery_drop / 0.2),
                description=f"Battery dropped {battery_drop*100:.1f}% in {dt:.1f}s"
            )
        
        # Check for battery increase (sensor fault)
        if battery_drop < -0.01:
            self._anomaly_count += 1
            return AnomalyEvent(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.SENSOR_FAULT,
                severity=AnomalySeverity.WARNING,
                agent_id=agent_id,
                expected_value=prev_state.energy.battery_level,
                actual_value=state.energy.battery_level,
                deviation=abs(battery_drop),
                confidence=0.8,
                description=f"Battery level increased (sensor fault?)"
            )
        
        return None
    
    def _check_trajectory_deviation(
        self,
        agent_id: str,
        state: AgentState
    ) -> Optional[AnomalyEvent]:
        """Check for trajectory deviation."""
        predictor = self._trajectory_predictors[agent_id]
        error = predictor.update(state.position, state.velocity, state.timestamp)
        
        if error > self.config.trajectory_deviation_threshold:
            self._anomaly_count += 1
            return AnomalyEvent(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.TRAJECTORY_DEVIATION,
                severity=AnomalySeverity.WARNING,
                agent_id=agent_id,
                expected_value=0.0,
                actual_value=error,
                deviation=error,
                confidence=min(1.0, error / (self.config.trajectory_deviation_threshold * 2)),
                description=f"Trajectory deviation: {error:.2f}m from predicted"
            )
        
        return None
    
    def _check_unexpected_stop(
        self,
        agent_id: str,
        prev_state: AgentState,
        state: AgentState
    ) -> Optional[AnomalyEvent]:
        """Check for unexpected stops."""
        prev_vel = prev_state.velocity.norm()
        curr_vel = state.velocity.norm()
        
        # Was moving, now stopped
        if prev_vel > 0.5 and curr_vel < 0.1:
            # Check if mode indicates intentional stop
            if state.mode.name not in ["IDLE", "HOVERING", "LANDED"]:
                self._anomaly_count += 1
                return AnomalyEvent(
                    anomaly_id=f"anomaly_{self._anomaly_count}",
                    anomaly_type=AnomalyType.UNEXPECTED_STOP,
                    severity=AnomalySeverity.WARNING,
                    agent_id=agent_id,
                    expected_value=prev_vel,
                    actual_value=curr_vel,
                    deviation=prev_vel - curr_vel,
                    confidence=0.7,
                    description=f"Unexpected stop: velocity dropped from {prev_vel:.2f} to {curr_vel:.2f} m/s"
                )
        
        return None
    
    def check_fleet_anomalies(
        self,
        states: Dict[str, AgentState]
    ) -> List[AnomalyEvent]:
        """
        Check for fleet-wide anomalies.
        
        Args:
            states: Dictionary of agent states
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Process each agent
        for agent_id, state in states.items():
            agent_anomalies = self.process_state(state)
            anomalies.extend(agent_anomalies)
        
        # Check for formation breaks
        # (would need formation tracking)
        
        # Check for collision risks
        positions = [(aid, s.position) for aid, s in states.items()]
        for i, (id_i, pos_i) in enumerate(positions):
            for j, (id_j, pos_j) in enumerate(positions):
                if i >= j:
                    continue
                
                dist = (pos_i - pos_j).norm()
                if dist < 0.5:  # Critical distance
                    self._anomaly_count += 1
                    anomalies.append(AnomalyEvent(
                        anomaly_id=f"anomaly_{self._anomaly_count}",
                        anomaly_type=AnomalyType.COLLISION_RISK,
                        severity=AnomalySeverity.CRITICAL,
                        agent_id=f"{id_i},{id_j}",
                        expected_value=1.0,
                        actual_value=dist,
                        deviation=1.0 - dist,
                        confidence=0.9,
                        description=f"Collision risk: {id_i} and {id_j} are {dist:.2f}m apart"
                    ))
        
        return anomalies
    
    def get_anomaly_history(
        self,
        agent_id: str = None,
        anomaly_type: AnomalyType = None,
        limit: int = None
    ) -> List[AnomalyEvent]:
        """Get filtered anomaly history."""
        anomalies = self._anomaly_history
        
        if agent_id:
            anomalies = [a for a in anomalies if a.agent_id == agent_id]
        
        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]
        
        if limit:
            anomalies = anomalies[-limit:]
        
        return anomalies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        type_counts = {}
        for anomaly in self._anomaly_history:
            type_name = anomaly.anomaly_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_anomalies": len(self._anomaly_history),
            "by_type": type_counts,
            "agents_monitored": len(self._velocity_detectors),
        }
