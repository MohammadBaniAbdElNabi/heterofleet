"""
Base Hardware Interface for HeteroFleet.

Defines abstract interface for hardware platforms.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import time
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.core.state import AgentState, OperationalMode


class ConnectionState(Enum):
    """Hardware connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


class HardwareStatus(Enum):
    """Hardware operational status."""
    UNKNOWN = auto()
    READY = auto()
    BUSY = auto()
    WARNING = auto()
    ERROR = auto()
    EMERGENCY = auto()


@dataclass
class TelemetryData:
    """Telemetry data from hardware."""
    
    # Position
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    
    # Orientation (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Battery
    battery_voltage: float = 0.0
    battery_percentage: float = 0.0
    
    # Motors
    motor_powers: List[float] = field(default_factory=list)
    
    # Sensors
    range_front: float = 0.0
    range_back: float = 0.0
    range_left: float = 0.0
    range_right: float = 0.0
    range_up: float = 0.0
    
    # Status
    timestamp: float = field(default_factory=time.time)
    is_flying: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [self.position.x, self.position.y, self.position.z],
            "velocity": [self.velocity.x, self.velocity.y, self.velocity.z],
            "orientation": [self.roll, self.pitch, self.yaw],
            "battery": self.battery_percentage,
            "timestamp": self.timestamp,
        }


class HardwareInterface(ABC):
    """
    Abstract base class for hardware interfaces.
    
    Defines common interface for all hardware platforms.
    """
    
    def __init__(self, agent_id: str, platform_type: PlatformType):
        """
        Initialize hardware interface.
        
        Args:
            agent_id: Agent identifier
            platform_type: Type of platform
        """
        self.agent_id = agent_id
        self.platform_type = platform_type
        
        self._connection_state = ConnectionState.DISCONNECTED
        self._hardware_status = HardwareStatus.UNKNOWN
        self._telemetry = TelemetryData()
        
        # Callbacks
        self._telemetry_callbacks: List[Callable[[TelemetryData], None]] = []
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
    
    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state
    
    @property
    def hardware_status(self) -> HardwareStatus:
        return self._hardware_status
    
    @property
    def telemetry(self) -> TelemetryData:
        return self._telemetry
    
    @property
    def is_connected(self) -> bool:
        return self._connection_state == ConnectionState.CONNECTED
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to hardware."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from hardware."""
        pass
    
    @abstractmethod
    async def arm(self) -> bool:
        """Arm the platform."""
        pass
    
    @abstractmethod
    async def disarm(self) -> bool:
        """Disarm the platform."""
        pass
    
    @abstractmethod
    async def takeoff(self, altitude: float = 1.0) -> bool:
        """Take off (for aerial platforms)."""
        pass
    
    @abstractmethod
    async def land(self) -> bool:
        """Land (for aerial platforms)."""
        pass
    
    @abstractmethod
    async def set_velocity(self, velocity: Vector3) -> bool:
        """Set velocity command."""
        pass
    
    @abstractmethod
    async def set_position(self, position: Vector3) -> bool:
        """Set position command."""
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """Emergency stop."""
        pass
    
    @abstractmethod
    async def get_telemetry(self) -> TelemetryData:
        """Get current telemetry."""
        pass
    
    def register_telemetry_callback(self, callback: Callable[[TelemetryData], None]) -> None:
        """Register callback for telemetry updates."""
        self._telemetry_callbacks.append(callback)
    
    def register_state_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)
    
    def _update_connection_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        old_state = self._connection_state
        self._connection_state = state
        
        if old_state != state:
            for callback in self._state_callbacks:
                callback(state)
    
    def _update_telemetry(self, telemetry: TelemetryData) -> None:
        """Update telemetry and notify callbacks."""
        self._telemetry = telemetry
        
        for callback in self._telemetry_callbacks:
            callback(telemetry)
    
    def to_agent_state(self) -> AgentState:
        """Convert telemetry to AgentState."""
        from heterofleet.core.state import Orientation, EnergyState
        
        return AgentState(
            agent_id=self.agent_id,
            position=self._telemetry.position,
            velocity=self._telemetry.velocity,
            orientation=Orientation(
                roll=self._telemetry.roll,
                pitch=self._telemetry.pitch,
                yaw=self._telemetry.yaw
            ),
            energy=EnergyState(
                battery_level=self._telemetry.battery_percentage / 100.0,
                battery_voltage=self._telemetry.battery_voltage,
            ),
            mode=self._get_operational_mode(),
            timestamp=self._telemetry.timestamp,
        )
    
    def _get_operational_mode(self) -> OperationalMode:
        """Convert hardware status to operational mode."""
        if self._hardware_status == HardwareStatus.EMERGENCY:
            return OperationalMode.EMERGENCY
        elif self._hardware_status == HardwareStatus.ERROR:
            return OperationalMode.EMERGENCY
        elif self._telemetry.is_flying:
            return OperationalMode.NAVIGATING
        else:
            return OperationalMode.IDLE
