"""
MentorPi Interface for HeteroFleet.

Provides interface to MentorPi ground vehicles.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.hardware.base import (
    HardwareInterface, ConnectionState, HardwareStatus, TelemetryData
)


@dataclass
class MentorPiConfig:
    """Configuration for MentorPi interface."""
    
    # Connection
    host: str = "192.168.1.100"
    port: int = 8080
    
    # Vehicle parameters
    wheel_base: float = 0.2  # meters
    max_speed: float = 0.5   # m/s
    max_turn_rate: float = 1.0  # rad/s
    
    # Sensors
    use_lidar: bool = True
    use_camera: bool = True
    use_imu: bool = True
    
    # Control
    control_frequency: int = 20  # Hz
    
    # Safety
    low_battery_threshold: float = 10.0  # Volts


class MentorPiInterface(HardwareInterface):
    """
    Interface to MentorPi ground vehicle.
    
    Supports:
    - Differential drive control
    - Mecanum wheel control
    - Sensor integration (LiDAR, camera, IMU)
    - Telemetry streaming
    """
    
    def __init__(self, agent_id: str, config: MentorPiConfig = None):
        """
        Initialize MentorPi interface.
        
        Args:
            agent_id: Agent identifier
            config: MentorPi configuration
        """
        super().__init__(agent_id, PlatformType.SMALL_UGV)
        
        self.config = config or MentorPiConfig()
        
        # Connection
        self._socket = None
        self._reader = None
        self._writer = None
        
        # State
        self._is_armed = False
        self._is_moving = False
        
        # Telemetry
        self._position = Vector3(0, 0, 0)
        self._velocity = Vector3(0, 0, 0)
        self._yaw = 0.0
        self._battery_voltage = 12.0
        
        # Sensors
        self._lidar_ranges: List[float] = []
        self._imu_data: Dict[str, float] = {}
        
        # Control
        self._target_velocity = Vector3(0, 0, 0)
        self._control_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """Connect to MentorPi."""
        try:
            self._update_connection_state(ConnectionState.CONNECTING)
            
            # Try to connect via TCP
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self.config.host, self.config.port
                )
                logger.info(f"Connected to MentorPi at {self.config.host}:{self.config.port}")
            except Exception as e:
                logger.warning(f"Could not connect to MentorPi: {e}. Running in simulation mode.")
            
            self._update_connection_state(ConnectionState.CONNECTED)
            self._hardware_status = HardwareStatus.READY
            
            # Start control loop
            self._control_task = asyncio.create_task(self._control_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MentorPi: {e}")
            self._update_connection_state(ConnectionState.ERROR)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MentorPi."""
        # Stop control loop
        if self._control_task:
            self._control_task.cancel()
            self._control_task = None
        
        # Stop motors
        await self._send_command("STOP")
        
        # Close connection
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        
        self._reader = None
        self._writer = None
        
        self._update_connection_state(ConnectionState.DISCONNECTED)
        logger.info(f"Disconnected from MentorPi {self.agent_id}")
    
    async def arm(self) -> bool:
        """Arm the MentorPi."""
        if not self.is_connected:
            return False
        
        # Send arm command
        await self._send_command("ARM")
        
        self._is_armed = True
        logger.info(f"MentorPi {self.agent_id} armed")
        return True
    
    async def disarm(self) -> bool:
        """Disarm the MentorPi."""
        # Stop motors
        await self._send_command("STOP")
        
        self._is_armed = False
        self._is_moving = False
        logger.info(f"MentorPi {self.agent_id} disarmed")
        return True
    
    async def takeoff(self, altitude: float = 1.0) -> bool:
        """Not applicable for ground vehicle."""
        logger.warning("Takeoff not applicable for ground vehicle")
        return False
    
    async def land(self) -> bool:
        """Not applicable for ground vehicle."""
        logger.warning("Land not applicable for ground vehicle")
        return False
    
    async def set_velocity(self, velocity: Vector3) -> bool:
        """Set velocity command."""
        if not self._is_armed:
            return False
        
        # Clamp velocity
        max_vel = self.config.max_speed
        vel_norm = velocity.norm()
        if vel_norm > max_vel:
            scale = max_vel / vel_norm
            velocity = Vector3(
                velocity.x * scale,
                velocity.y * scale,
                0  # Ground vehicle
            )
        
        self._target_velocity = velocity
        
        # Send to hardware
        await self._send_velocity_command(velocity)
        
        return True
    
    async def set_position(self, position: Vector3) -> bool:
        """Set position command (requires position feedback)."""
        if not self._is_armed:
            return False
        
        # Simple position control - compute velocity toward target
        error = position - self._position
        error.z = 0  # Ground vehicle
        
        dist = error.norm()
        if dist < 0.05:
            # At target
            await self.set_velocity(Vector3(0, 0, 0))
            return True
        
        # Move toward target
        kp = 1.0
        velocity = Vector3(error.x * kp, error.y * kp, 0)
        
        return await self.set_velocity(velocity)
    
    async def emergency_stop(self) -> bool:
        """Emergency stop."""
        await self._send_command("EMERGENCY_STOP")
        
        self._is_moving = False
        self._is_armed = False
        self._target_velocity = Vector3(0, 0, 0)
        self._hardware_status = HardwareStatus.EMERGENCY
        
        logger.warning(f"MentorPi {self.agent_id} EMERGENCY STOP")
        return True
    
    async def get_telemetry(self) -> TelemetryData:
        """Get current telemetry."""
        return TelemetryData(
            position=self._position,
            velocity=self._velocity,
            roll=0.0,
            pitch=0.0,
            yaw=self._yaw,
            battery_voltage=self._battery_voltage,
            battery_percentage=self._estimate_battery_percentage(),
            timestamp=time.time(),
            is_flying=False,
        )
    
    def _estimate_battery_percentage(self) -> float:
        """Estimate battery percentage."""
        # Assuming 3S LiPo (12.6V full, 9V empty)
        v = self._battery_voltage
        if v >= 12.6:
            return 100.0
        elif v <= 9.0:
            return 0.0
        else:
            return (v - 9.0) / 3.6 * 100.0
    
    async def _send_command(self, command: str) -> bool:
        """Send command to MentorPi."""
        if self._writer is None:
            # Simulated mode
            logger.debug(f"Simulated command: {command}")
            return True
        
        try:
            self._writer.write(f"{command}\n".encode())
            await self._writer.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    async def _send_velocity_command(self, velocity: Vector3) -> bool:
        """Send velocity command to MentorPi."""
        # Convert to wheel velocities for differential drive
        # Forward velocity and angular velocity
        v_linear = velocity.x  # Forward
        v_angular = velocity.y * 2  # Simplified turning
        
        # Compute wheel velocities
        wheel_base = self.config.wheel_base
        v_left = v_linear - v_angular * wheel_base / 2
        v_right = v_linear + v_angular * wheel_base / 2
        
        command = f"VELOCITY {v_left:.3f} {v_right:.3f}"
        return await self._send_command(command)
    
    async def _control_loop(self) -> None:
        """Control loop for continuous updates."""
        interval = 1.0 / self.config.control_frequency
        
        while True:
            try:
                # Request telemetry
                await self._request_telemetry()
                
                # Check battery
                if self._battery_voltage < self.config.low_battery_threshold:
                    self._hardware_status = HardwareStatus.WARNING
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Control loop error: {e}")
    
    async def _request_telemetry(self) -> None:
        """Request telemetry from MentorPi."""
        if self._writer is None:
            # Simulated - update position based on velocity
            dt = 1.0 / self.config.control_frequency
            self._position = Vector3(
                self._position.x + self._target_velocity.x * dt,
                self._position.y + self._target_velocity.y * dt,
                0
            )
            self._velocity = self._target_velocity
            return
        
        # Request and parse telemetry
        await self._send_command("TELEMETRY")
        
        try:
            data = await asyncio.wait_for(self._reader.readline(), timeout=0.1)
            self._parse_telemetry(data.decode())
        except asyncio.TimeoutError:
            pass
    
    def _parse_telemetry(self, data: str) -> None:
        """Parse telemetry data."""
        try:
            parts = data.strip().split()
            if len(parts) >= 7:
                self._position = Vector3(
                    float(parts[0]),
                    float(parts[1]),
                    0
                )
                self._velocity = Vector3(
                    float(parts[2]),
                    float(parts[3]),
                    0
                )
                self._yaw = float(parts[4])
                self._battery_voltage = float(parts[5])
        except Exception as e:
            logger.warning(f"Failed to parse telemetry: {e}")
    
    # Mecanum wheel control
    async def set_mecanum_velocity(
        self,
        vx: float,
        vy: float,
        omega: float
    ) -> bool:
        """
        Set mecanum wheel velocity (for omnidirectional movement).
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            omega: Angular velocity (rad/s)
        """
        if not self._is_armed:
            return False
        
        command = f"MECANUM {vx:.3f} {vy:.3f} {omega:.3f}"
        return await self._send_command(command)
    
    # Sensor access
    async def get_lidar_scan(self) -> List[float]:
        """Get LiDAR scan data."""
        if not self.config.use_lidar:
            return []
        
        await self._send_command("LIDAR")
        
        if self._reader:
            try:
                data = await asyncio.wait_for(self._reader.readline(), timeout=0.5)
                self._lidar_ranges = [float(x) for x in data.decode().strip().split()]
            except:
                pass
        
        return self._lidar_ranges
    
    async def get_camera_frame(self) -> Optional[bytes]:
        """Get camera frame."""
        if not self.config.use_camera:
            return None
        
        # Would implement camera frame retrieval
        return None
