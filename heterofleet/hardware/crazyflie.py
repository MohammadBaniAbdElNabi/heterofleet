"""
Crazyflie Interface for HeteroFleet.

Provides interface to Crazyflie micro drones using cflib.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.hardware.base import (
    HardwareInterface, ConnectionState, HardwareStatus, TelemetryData
)

# Try to import cflib (optional dependency)
try:
    import cflib
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.positioning.motion_commander import MotionCommander
    from cflib.positioning.position_hl_commander import PositionHlCommander
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    logger.warning("cflib not available - Crazyflie hardware interface will be simulated")


@dataclass
class CrazyflieConfig:
    """Configuration for Crazyflie interface."""
    
    uri: str = "radio://0/80/2M/E7E7E7E7E7"
    
    # Flight parameters
    default_height: float = 0.5  # meters
    max_velocity: float = 0.5   # m/s
    
    # Lighthouse
    use_lighthouse: bool = True
    
    # Multi-ranger
    use_multiranger: bool = True
    
    # Flow deck
    use_flowdeck: bool = False
    
    # Logging
    log_frequency: int = 50  # Hz
    
    # Safety
    low_battery_threshold: float = 3.0  # Volts


class CrazyflieInterface(HardwareInterface):
    """
    Interface to Crazyflie micro drone.
    
    Supports:
    - Position control via Lighthouse or Flow deck
    - Velocity commands
    - Multi-ranger obstacle sensing
    - Telemetry streaming
    """
    
    def __init__(self, agent_id: str, config: CrazyflieConfig = None):
        """
        Initialize Crazyflie interface.
        
        Args:
            agent_id: Agent identifier
            config: Crazyflie configuration
        """
        super().__init__(agent_id, PlatformType.MICRO_UAV)
        
        self.config = config or CrazyflieConfig()
        
        # Crazyflie objects
        self._cf: Optional[Any] = None
        self._scf: Optional[Any] = None
        self._mc: Optional[Any] = None
        self._pc: Optional[Any] = None
        
        # State
        self._is_armed = False
        self._is_flying = False
        self._target_position: Optional[Vector3] = None
        
        # Telemetry
        self._position = Vector3(0, 0, 0)
        self._velocity = Vector3(0, 0, 0)
        self._battery_voltage = 4.2
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        
        # Range sensors
        self._ranges = {"front": 0.0, "back": 0.0, "left": 0.0, "right": 0.0, "up": 0.0}
    
    async def connect(self) -> bool:
        """Connect to Crazyflie."""
        if not CFLIB_AVAILABLE:
            logger.info(f"Simulating connection to {self.config.uri}")
            self._update_connection_state(ConnectionState.CONNECTED)
            self._hardware_status = HardwareStatus.READY
            return True
        
        try:
            self._update_connection_state(ConnectionState.CONNECTING)
            
            # Initialize cflib
            cflib.crtp.init_drivers()
            
            # Connect
            self._cf = Crazyflie(rw_cache='./cache')
            self._scf = SyncCrazyflie(self.config.uri, cf=self._cf)
            self._scf.open_link()
            
            # Set up logging
            self._setup_logging()
            
            # Check positioning system
            if self.config.use_lighthouse:
                await self._setup_lighthouse()
            
            self._update_connection_state(ConnectionState.CONNECTED)
            self._hardware_status = HardwareStatus.READY
            
            logger.info(f"Connected to Crazyflie at {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Crazyflie: {e}")
            self._update_connection_state(ConnectionState.ERROR)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Crazyflie."""
        if self._is_flying:
            await self.land()
        
        if CFLIB_AVAILABLE and self._scf:
            self._scf.close_link()
        
        self._cf = None
        self._scf = None
        self._mc = None
        self._pc = None
        
        self._update_connection_state(ConnectionState.DISCONNECTED)
        logger.info(f"Disconnected from Crazyflie {self.agent_id}")
    
    async def arm(self) -> bool:
        """Arm the Crazyflie."""
        if not self.is_connected:
            return False
        
        self._is_armed = True
        logger.info(f"Crazyflie {self.agent_id} armed")
        return True
    
    async def disarm(self) -> bool:
        """Disarm the Crazyflie."""
        if self._is_flying:
            await self.land()
        
        self._is_armed = False
        logger.info(f"Crazyflie {self.agent_id} disarmed")
        return True
    
    async def takeoff(self, altitude: float = None) -> bool:
        """Take off to specified altitude."""
        if not self._is_armed:
            logger.warning("Cannot takeoff - not armed")
            return False
        
        altitude = altitude or self.config.default_height
        
        if CFLIB_AVAILABLE and self._scf:
            try:
                self._pc = PositionHlCommander(self._scf)
                self._pc.take_off(altitude)
                self._is_flying = True
                self._position.z = altitude
                logger.info(f"Crazyflie {self.agent_id} took off to {altitude}m")
                return True
            except Exception as e:
                logger.error(f"Takeoff failed: {e}")
                return False
        else:
            # Simulated
            self._is_flying = True
            self._position.z = altitude
            return True
    
    async def land(self) -> bool:
        """Land the Crazyflie."""
        if not self._is_flying:
            return True
        
        if CFLIB_AVAILABLE and self._pc:
            try:
                self._pc.land()
                self._is_flying = False
                self._position.z = 0.0
                logger.info(f"Crazyflie {self.agent_id} landed")
                return True
            except Exception as e:
                logger.error(f"Landing failed: {e}")
                return False
        else:
            self._is_flying = False
            self._position.z = 0.0
            return True
    
    async def set_velocity(self, velocity: Vector3) -> bool:
        """Set velocity command."""
        if not self._is_flying:
            return False
        
        # Clamp velocity
        max_vel = self.config.max_velocity
        vel_norm = velocity.norm()
        if vel_norm > max_vel:
            scale = max_vel / vel_norm
            velocity = Vector3(
                velocity.x * scale,
                velocity.y * scale,
                velocity.z * scale
            )
        
        if CFLIB_AVAILABLE and self._cf:
            try:
                # Send velocity setpoint
                self._cf.commander.send_velocity_world_setpoint(
                    velocity.x, velocity.y, velocity.z, 0  # yaw rate
                )
                return True
            except Exception as e:
                logger.error(f"Set velocity failed: {e}")
                return False
        else:
            # Simulated
            self._velocity = velocity
            return True
    
    async def set_position(self, position: Vector3) -> bool:
        """Set position command."""
        if not self._is_flying:
            return False
        
        if CFLIB_AVAILABLE and self._pc:
            try:
                self._pc.go_to(position.x, position.y, position.z)
                self._target_position = position
                return True
            except Exception as e:
                logger.error(f"Set position failed: {e}")
                return False
        else:
            self._position = position
            self._target_position = position
            return True
    
    async def emergency_stop(self) -> bool:
        """Emergency stop - cut motors."""
        if CFLIB_AVAILABLE and self._cf:
            self._cf.commander.send_stop_setpoint()
        
        self._is_flying = False
        self._is_armed = False
        self._hardware_status = HardwareStatus.EMERGENCY
        
        logger.warning(f"Crazyflie {self.agent_id} EMERGENCY STOP")
        return True
    
    async def get_telemetry(self) -> TelemetryData:
        """Get current telemetry."""
        return TelemetryData(
            position=self._position,
            velocity=self._velocity,
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw,
            battery_voltage=self._battery_voltage,
            battery_percentage=self._estimate_battery_percentage(),
            range_front=self._ranges["front"],
            range_back=self._ranges["back"],
            range_left=self._ranges["left"],
            range_right=self._ranges["right"],
            range_up=self._ranges["up"],
            timestamp=time.time(),
            is_flying=self._is_flying,
        )
    
    def _estimate_battery_percentage(self) -> float:
        """Estimate battery percentage from voltage."""
        # LiPo voltage curve approximation
        v = self._battery_voltage
        if v >= 4.2:
            return 100.0
        elif v <= 3.0:
            return 0.0
        else:
            return (v - 3.0) / 1.2 * 100.0
    
    def _setup_logging(self) -> None:
        """Set up telemetry logging."""
        if not CFLIB_AVAILABLE or not self._cf:
            return
        
        # State estimate logging
        lg_state = LogConfig(name='StateEstimate', period_in_ms=1000 // self.config.log_frequency)
        lg_state.add_variable('stateEstimate.x', 'float')
        lg_state.add_variable('stateEstimate.y', 'float')
        lg_state.add_variable('stateEstimate.z', 'float')
        lg_state.add_variable('stateEstimate.vx', 'float')
        lg_state.add_variable('stateEstimate.vy', 'float')
        lg_state.add_variable('stateEstimate.vz', 'float')
        
        try:
            self._cf.log.add_config(lg_state)
            lg_state.data_received_cb.add_callback(self._state_callback)
            lg_state.start()
        except Exception as e:
            logger.warning(f"Could not add state logging: {e}")
        
        # Stabilizer logging
        lg_stab = LogConfig(name='Stabilizer', period_in_ms=1000 // self.config.log_frequency)
        lg_stab.add_variable('stabilizer.roll', 'float')
        lg_stab.add_variable('stabilizer.pitch', 'float')
        lg_stab.add_variable('stabilizer.yaw', 'float')
        
        try:
            self._cf.log.add_config(lg_stab)
            lg_stab.data_received_cb.add_callback(self._stab_callback)
            lg_stab.start()
        except Exception as e:
            logger.warning(f"Could not add stabilizer logging: {e}")
        
        # Battery logging
        lg_batt = LogConfig(name='Battery', period_in_ms=500)
        lg_batt.add_variable('pm.vbat', 'float')
        
        try:
            self._cf.log.add_config(lg_batt)
            lg_batt.data_received_cb.add_callback(self._battery_callback)
            lg_batt.start()
        except Exception as e:
            logger.warning(f"Could not add battery logging: {e}")
        
        # Multi-ranger logging
        if self.config.use_multiranger:
            lg_range = LogConfig(name='Range', period_in_ms=100)
            lg_range.add_variable('range.front', 'uint16_t')
            lg_range.add_variable('range.back', 'uint16_t')
            lg_range.add_variable('range.left', 'uint16_t')
            lg_range.add_variable('range.right', 'uint16_t')
            lg_range.add_variable('range.up', 'uint16_t')
            
            try:
                self._cf.log.add_config(lg_range)
                lg_range.data_received_cb.add_callback(self._range_callback)
                lg_range.start()
            except Exception as e:
                logger.warning(f"Could not add range logging: {e}")
    
    def _state_callback(self, timestamp, data, logconf) -> None:
        """Handle state estimate data."""
        self._position = Vector3(
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z']
        )
        self._velocity = Vector3(
            data['stateEstimate.vx'],
            data['stateEstimate.vy'],
            data['stateEstimate.vz']
        )
        self._update_telemetry_sync()
    
    def _stab_callback(self, timestamp, data, logconf) -> None:
        """Handle stabilizer data."""
        import math
        self._roll = math.radians(data['stabilizer.roll'])
        self._pitch = math.radians(data['stabilizer.pitch'])
        self._yaw = math.radians(data['stabilizer.yaw'])
    
    def _battery_callback(self, timestamp, data, logconf) -> None:
        """Handle battery data."""
        self._battery_voltage = data['pm.vbat']
        
        # Check low battery
        if self._battery_voltage < self.config.low_battery_threshold:
            self._hardware_status = HardwareStatus.WARNING
            logger.warning(f"Low battery: {self._battery_voltage}V")
    
    def _range_callback(self, timestamp, data, logconf) -> None:
        """Handle range sensor data."""
        self._ranges["front"] = data['range.front'] / 1000.0  # mm to m
        self._ranges["back"] = data['range.back'] / 1000.0
        self._ranges["left"] = data['range.left'] / 1000.0
        self._ranges["right"] = data['range.right'] / 1000.0
        self._ranges["up"] = data['range.up'] / 1000.0
    
    def _update_telemetry_sync(self) -> None:
        """Update telemetry synchronously."""
        telemetry = TelemetryData(
            position=self._position,
            velocity=self._velocity,
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw,
            battery_voltage=self._battery_voltage,
            battery_percentage=self._estimate_battery_percentage(),
            range_front=self._ranges["front"],
            range_back=self._ranges["back"],
            range_left=self._ranges["left"],
            range_right=self._ranges["right"],
            range_up=self._ranges["up"],
            timestamp=time.time(),
            is_flying=self._is_flying,
        )
        self._update_telemetry(telemetry)
    
    async def _setup_lighthouse(self) -> None:
        """Set up Lighthouse positioning."""
        if not CFLIB_AVAILABLE:
            return
        
        logger.info("Setting up Lighthouse positioning")
        # Lighthouse setup would go here


class CrazyflieSwarm:
    """
    Manager for multiple Crazyflies.
    """
    
    def __init__(self):
        """Initialize swarm manager."""
        self._drones: Dict[str, CrazyflieInterface] = {}
    
    def add_drone(self, agent_id: str, uri: str) -> CrazyflieInterface:
        """Add a drone to the swarm."""
        config = CrazyflieConfig(uri=uri)
        drone = CrazyflieInterface(agent_id, config)
        self._drones[agent_id] = drone
        return drone
    
    def remove_drone(self, agent_id: str) -> None:
        """Remove a drone from the swarm."""
        self._drones.pop(agent_id, None)
    
    def get_drone(self, agent_id: str) -> Optional[CrazyflieInterface]:
        """Get a drone by ID."""
        return self._drones.get(agent_id)
    
    def get_all_drones(self) -> List[CrazyflieInterface]:
        """Get all drones."""
        return list(self._drones.values())
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all drones."""
        results = {}
        for agent_id, drone in self._drones.items():
            results[agent_id] = await drone.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect from all drones."""
        for drone in self._drones.values():
            await drone.disconnect()
    
    async def takeoff_all(self, altitude: float = 0.5) -> Dict[str, bool]:
        """Take off all drones."""
        results = {}
        for agent_id, drone in self._drones.items():
            results[agent_id] = await drone.takeoff(altitude)
        return results
    
    async def land_all(self) -> Dict[str, bool]:
        """Land all drones."""
        results = {}
        for agent_id, drone in self._drones.items():
            results[agent_id] = await drone.land()
        return results
    
    async def emergency_stop_all(self) -> None:
        """Emergency stop all drones."""
        for drone in self._drones.values():
            await drone.emergency_stop()
    
    def get_telemetry_all(self) -> Dict[str, TelemetryData]:
        """Get telemetry from all drones."""
        return {aid: drone.telemetry for aid, drone in self._drones.items()}
