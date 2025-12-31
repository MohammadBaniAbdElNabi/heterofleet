"""
Platform specifications and registry for heterogeneous autonomous vehicles.

This module defines:
- Platform types (UAVs, UGVs, USVs)
- Physical, dynamic, and operational properties
- Platform registry for managing specifications
- Platform factory for creating platform instances

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from loguru import logger


class PlatformType(Enum):
    """
    Enumeration of supported platform types.
    
    Covers aerial, ground, and surface vehicles of various sizes.
    """
    # Aerial platforms (UAVs)
    MICRO_UAV = "micro_uav"      # e.g., Crazyflie 2.1 (~30g)
    SMALL_UAV = "small_uav"      # e.g., DJI Mini (~250g)
    MEDIUM_UAV = "medium_uav"    # e.g., DJI Matrice (~3kg)
    LARGE_UAV = "large_uav"      # e.g., Custom heavy-lift (>5kg)
    
    # Ground platforms (UGVs)
    SMALL_UGV = "small_ugv"      # e.g., MentorPi (~1.5kg)
    MEDIUM_UGV = "medium_ugv"    # e.g., Jackal (~17kg)
    LARGE_UGV = "large_ugv"      # e.g., Husky (~50kg)
    
    # Surface platforms (USVs)
    SMALL_USV = "small_usv"      # Small surface vehicle
    MEDIUM_USV = "medium_usv"    # Medium surface vehicle
    
    @property
    def is_aerial(self) -> bool:
        """Check if platform is aerial (UAV)."""
        return self in {
            PlatformType.MICRO_UAV,
            PlatformType.SMALL_UAV,
            PlatformType.MEDIUM_UAV,
            PlatformType.LARGE_UAV,
        }
    
    @property
    def is_ground(self) -> bool:
        """Check if platform is ground-based (UGV)."""
        return self in {
            PlatformType.SMALL_UGV,
            PlatformType.MEDIUM_UGV,
            PlatformType.LARGE_UGV,
        }
    
    @property
    def is_surface(self) -> bool:
        """Check if platform is surface-based (USV)."""
        return self in {
            PlatformType.SMALL_USV,
            PlatformType.MEDIUM_USV,
        }
    
    @property
    def domain(self) -> DomainType:
        """Get the operational domain of the platform."""
        if self.is_aerial:
            return DomainType.AERIAL
        elif self.is_ground:
            return DomainType.GROUND
        elif self.is_surface:
            return DomainType.SURFACE
        else:
            return DomainType.AERIAL  # Default


class DomainType(Enum):
    """Operational domain for platforms."""
    AERIAL = "aerial"
    GROUND = "ground"
    SURFACE = "surface"
    UNDERWATER = "underwater"


class CollisionEnvelopeType(Enum):
    """Type of collision envelope geometry."""
    ELLIPSOID = "ellipsoid"
    CYLINDER = "cylinder"
    BOX = "box"
    SPHERE = "sphere"


@dataclass
class Vector3:
    """3D vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __post_init__(self):
        """Convert to float if needed."""
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> Vector3:
        """Create from numpy array."""
        return cls(x=arr[0], y=arr[1], z=arr[2])
    
    @classmethod
    def from_list(cls, lst: List[float]) -> Vector3:
        """Create from list."""
        return cls(x=lst[0], y=lst[1], z=lst[2])
    
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)
    
    def norm(self) -> float:
        """Compute Euclidean norm."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> Vector3:
        """Return normalized vector."""
        n = self.norm()
        if n < 1e-10:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / n, self.y / n, self.z / n)
    
    def dot(self, other: Vector3) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: Vector3) -> Vector3:
        """Cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass
class CollisionEnvelope:
    """
    Collision envelope for a platform.
    
    Supports ellipsoid, cylinder, box, and sphere geometries.
    """
    type: CollisionEnvelopeType = CollisionEnvelopeType.ELLIPSOID
    semi_axes: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # a, b, c in meters
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the collision envelope matrix Θ.
        
        For ellipsoid: Θ = diag(a², b², c²)
        Used in ellipsoidal distance computation: ||p||_Θ⁻¹
        """
        a, b, c = self.semi_axes
        return np.diag([a**2, b**2, c**2])
    
    def get_inverse_matrix(self) -> np.ndarray:
        """Get the inverse of the collision envelope matrix."""
        a, b, c = self.semi_axes
        return np.diag([1.0/a**2, 1.0/b**2, 1.0/c**2])
    
    def contains_point(self, point: Vector3, center: Vector3 = None) -> bool:
        """Check if a point is inside the collision envelope."""
        if center is None:
            center = Vector3(0, 0, 0)
        
        diff = (point - center).to_array()
        
        if self.type == CollisionEnvelopeType.ELLIPSOID:
            # Check (x/a)² + (y/b)² + (z/c)² <= 1
            a, b, c = self.semi_axes
            return (diff[0]/a)**2 + (diff[1]/b)**2 + (diff[2]/c)**2 <= 1.0
        
        elif self.type == CollisionEnvelopeType.SPHERE:
            r = self.semi_axes[0]  # Assume sphere uses first axis as radius
            return np.sum(diff**2) <= r**2
        
        elif self.type == CollisionEnvelopeType.CYLINDER:
            # Cylinder aligned with z-axis
            r = self.semi_axes[0]  # radius
            h = self.semi_axes[2]  # half-height
            return diff[0]**2 + diff[1]**2 <= r**2 and abs(diff[2]) <= h
        
        elif self.type == CollisionEnvelopeType.BOX:
            a, b, c = self.semi_axes
            return abs(diff[0]) <= a and abs(diff[1]) <= b and abs(diff[2]) <= c
        
        return False
    
    def compute_combined_envelope(
        self,
        other: CollisionEnvelope,
        downwash_factor: float = 0.0
    ) -> np.ndarray:
        """
        Compute combined collision envelope matrix Θ(i,j) for two platforms.
        
        Θ(i,j) = diag((a_i + a_j)², (b_i + b_j)², (c_i + c_j + d_w)²)
        
        Args:
            other: Other platform's collision envelope
            downwash_factor: Additional vertical separation for UAV-UAV pairs
            
        Returns:
            3x3 combined envelope matrix
        """
        a1, b1, c1 = self.semi_axes
        a2, b2, c2 = other.semi_axes
        
        combined_a = a1 + a2
        combined_b = b1 + b2
        combined_c = c1 + c2 + downwash_factor
        
        return np.diag([combined_a**2, combined_b**2, combined_c**2])


@dataclass
class PhysicalProperties:
    """Physical properties of a platform."""
    mass: float  # kg
    length: float  # m
    width: float  # m
    height: float  # m
    collision_envelope: CollisionEnvelope = field(default_factory=CollisionEnvelope)
    inertia_matrix: Optional[np.ndarray] = None  # 3x3 inertia matrix
    
    def __post_init__(self):
        """Initialize default inertia if not provided."""
        if self.inertia_matrix is None:
            # Approximate as uniform box
            m = self.mass
            l, w, h = self.length, self.width, self.height
            self.inertia_matrix = np.diag([
                m * (w**2 + h**2) / 12,
                m * (l**2 + h**2) / 12,
                m * (l**2 + w**2) / 12
            ])


@dataclass
class DynamicProperties:
    """Dynamic/kinematic properties of a platform."""
    max_velocity: Vector3  # m/s in x, y, z
    max_acceleration: Vector3  # m/s²
    max_angular_velocity: float  # rad/s
    response_time: float  # seconds (closed-loop response time)
    
    # Optional advanced dynamics
    drag_coefficient: float = 0.0  # Drag coefficient
    motor_time_constant: float = 0.02  # Motor response time constant
    
    # Control system matrices (for linear approximation)
    A_matrix: Optional[np.ndarray] = None  # State transition matrix
    B_matrix: Optional[np.ndarray] = None  # Control input matrix
    
    def get_max_velocity_magnitude(self) -> float:
        """Get maximum velocity magnitude."""
        return self.max_velocity.norm()
    
    def get_max_acceleration_magnitude(self) -> float:
        """Get maximum acceleration magnitude."""
        return self.max_acceleration.norm()


@dataclass
class DomainConstraints:
    """Operational domain constraints for a platform."""
    domain: DomainType
    altitude_range: Tuple[float, float]  # (min, max) in meters
    terrain_capability: List[str] = field(default_factory=list)  # e.g., ["flat", "rough"]
    
    # Arena bounds (if applicable)
    arena_min: Optional[Vector3] = None
    arena_max: Optional[Vector3] = None
    
    # No-fly zones or restricted areas
    restricted_zones: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_position_valid(self, position: Vector3) -> bool:
        """Check if a position is valid for this platform."""
        # Check altitude constraints
        if self.domain == DomainType.AERIAL:
            if position.z < self.altitude_range[0] or position.z > self.altitude_range[1]:
                return False
        elif self.domain == DomainType.GROUND:
            # Ground vehicles should be at ground level (z ≈ 0)
            if abs(position.z - self.altitude_range[0]) > 0.5:
                return False
        
        # Check arena bounds
        if self.arena_min is not None and self.arena_max is not None:
            if (position.x < self.arena_min.x or position.x > self.arena_max.x or
                position.y < self.arena_min.y or position.y > self.arena_max.y or
                position.z < self.arena_min.z or position.z > self.arena_max.z):
                return False
        
        return True


@dataclass
class CommunicationProperties:
    """Communication capabilities of a platform."""
    protocols: List[str]  # e.g., ["wifi", "lte", "lora", "uwb"]
    range: float  # meters
    bandwidth: float  # Mbps
    latency: float  # ms typical
    
    # Protocol-specific parameters
    wifi_frequency: Optional[float] = 2.4  # GHz
    lte_band: Optional[str] = None
    uwb_channel: Optional[int] = None
    
    def get_effective_range(self, protocol: str) -> float:
        """Get effective communication range for a specific protocol."""
        protocol_ranges = {
            "wifi": self.range,
            "lte": self.range * 10,  # LTE typically has longer range
            "lora": self.range * 5,  # LoRa has good range
            "uwb": min(self.range, 100),  # UWB is typically shorter range
            "crazyradio": min(self.range, 100),
        }
        return protocol_ranges.get(protocol, self.range)


@dataclass
class EnergyProperties:
    """Energy/battery properties of a platform."""
    capacity: float  # Wh
    hover_power: float  # W (for UAVs, 0 for UGVs)
    cruise_power: float  # W at nominal speed
    max_power: float  # W
    
    # Battery characteristics
    voltage_nominal: float = 3.7  # V per cell
    num_cells: int = 1
    discharge_rate: float = 1.0  # C rating
    
    def compute_flight_time(self, power_consumption: float) -> float:
        """Compute estimated operation time given power consumption."""
        if power_consumption <= 0:
            return float('inf')
        return self.capacity / power_consumption * 3600  # Convert to seconds
    
    def compute_energy_for_distance(
        self,
        distance: float,
        speed: float,
        is_aerial: bool = True
    ) -> float:
        """
        Compute energy required to travel a distance.
        
        Args:
            distance: Distance in meters
            speed: Speed in m/s
            is_aerial: Whether platform is aerial
            
        Returns:
            Energy in Wh
        """
        if speed <= 0:
            return float('inf')
        
        time_seconds = distance / speed
        
        if is_aerial:
            # For UAVs, include hover power component
            avg_power = self.hover_power + (self.cruise_power - self.hover_power) * 0.5
        else:
            avg_power = self.cruise_power
        
        energy_wh = avg_power * time_seconds / 3600
        return energy_wh


@dataclass
class SensorProperties:
    """Sensor capabilities of a platform."""
    positioning: List[str]  # e.g., ["gps", "uwb", "vicon", "slam"]
    perception: List[str]  # e.g., ["camera", "lidar", "radar", "sonar"]
    
    # Sensor-specific parameters
    gps_accuracy: float = 2.0  # meters
    uwb_accuracy: float = 0.1  # meters
    camera_resolution: Optional[Tuple[int, int]] = None
    lidar_range: Optional[float] = None
    
    def get_best_positioning_accuracy(self) -> float:
        """Get the best available positioning accuracy."""
        accuracies = {
            "vicon": 0.001,  # mm-level
            "lighthouse": 0.01,  # cm-level
            "uwb": self.uwb_accuracy,
            "flow_deck": 0.05,
            "slam": 0.1,
            "gps": self.gps_accuracy,
            "wheel_odometry": 0.2,
        }
        
        best = float('inf')
        for sensor in self.positioning:
            if sensor in accuracies:
                best = min(best, accuracies[sensor])
        
        return best if best < float('inf') else self.gps_accuracy


@dataclass
class CapabilityProperties:
    """Special capabilities of a platform."""
    payload_capacity: float  # kg
    manipulation: bool = False  # Has manipulator/gripper
    can_land: bool = True  # Can land (UAVs)
    can_dock: bool = False  # Can dock with other platforms
    can_carry_other: bool = False  # Can carry smaller platforms
    
    # Task-specific capabilities
    capable_tasks: List[str] = field(default_factory=list)  # e.g., ["delivery", "surveillance"]


class PlatformSpecification(BaseModel):
    """
    Complete specification for a platform type.
    
    This is the main class that combines all platform properties.
    Supports serialization to/from YAML for configuration.
    """
    
    platform_id: str
    platform_type: PlatformType
    
    # Component properties (stored as dicts for Pydantic compatibility)
    physical: Dict[str, Any] = Field(default_factory=dict)
    dynamic: Dict[str, Any] = Field(default_factory=dict)
    domain: Dict[str, Any] = Field(default_factory=dict)
    communication: Dict[str, Any] = Field(default_factory=dict)
    energy: Dict[str, Any] = Field(default_factory=dict)
    sensors: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    
    # Cached dataclass instances
    _physical_props: Optional[PhysicalProperties] = None
    _dynamic_props: Optional[DynamicProperties] = None
    _domain_props: Optional[DomainConstraints] = None
    _comm_props: Optional[CommunicationProperties] = None
    _energy_props: Optional[EnergyProperties] = None
    _sensor_props: Optional[SensorProperties] = None
    _capability_props: Optional[CapabilityProperties] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        # Keep enum as enum instead of using values
        use_enum_values = False
    
    @property
    def physical_properties(self) -> PhysicalProperties:
        """Get physical properties as dataclass."""
        if self._physical_props is None:
            p = self.physical
            envelope = CollisionEnvelope(
                type=CollisionEnvelopeType(p.get("collision_envelope", {}).get("type", "ellipsoid")),
                semi_axes=tuple(p.get("collision_envelope", {}).get("semi_axes", [0.1, 0.1, 0.1]))
            )
            self._physical_props = PhysicalProperties(
                mass=p.get("mass", 1.0),
                length=p.get("dimensions", {}).get("length", 0.1),
                width=p.get("dimensions", {}).get("width", 0.1),
                height=p.get("dimensions", {}).get("height", 0.1),
                collision_envelope=envelope
            )
        return self._physical_props
    
    @property
    def dynamic_properties(self) -> DynamicProperties:
        """Get dynamic properties as dataclass."""
        if self._dynamic_props is None:
            d = self.dynamic
            max_vel = d.get("max_velocity", [1.0, 1.0, 1.0])
            max_acc = d.get("max_acceleration", [2.0, 2.0, 2.0])
            self._dynamic_props = DynamicProperties(
                max_velocity=Vector3.from_list(max_vel),
                max_acceleration=Vector3.from_list(max_acc),
                max_angular_velocity=d.get("max_angular_velocity", 3.14),
                response_time=d.get("response_time", 0.1)
            )
        return self._dynamic_props
    
    @property
    def domain_constraints(self) -> DomainConstraints:
        """Get domain constraints as dataclass."""
        if self._domain_props is None:
            d = self.domain
            self._domain_props = DomainConstraints(
                domain=DomainType(d.get("domain", "aerial")),
                altitude_range=tuple(d.get("altitude_range", [0.0, 10.0])),
                terrain_capability=d.get("terrain_capability", [])
            )
        return self._domain_props
    
    @property
    def communication_properties(self) -> CommunicationProperties:
        """Get communication properties as dataclass."""
        if self._comm_props is None:
            c = self.communication
            self._comm_props = CommunicationProperties(
                protocols=c.get("protocols", ["wifi"]),
                range=c.get("range", 100.0),
                bandwidth=c.get("bandwidth", 10.0),
                latency=c.get("latency", 10.0)
            )
        return self._comm_props
    
    @property
    def energy_properties(self) -> EnergyProperties:
        """Get energy properties as dataclass."""
        if self._energy_props is None:
            e = self.energy
            model = e.get("consumption_model", {})
            self._energy_props = EnergyProperties(
                capacity=e.get("capacity", 10.0),
                hover_power=model.get("hover", 0.0),
                cruise_power=model.get("cruise", 5.0),
                max_power=model.get("max_power", 20.0)
            )
        return self._energy_props
    
    @property
    def sensor_properties(self) -> SensorProperties:
        """Get sensor properties as dataclass."""
        if self._sensor_props is None:
            s = self.sensors
            self._sensor_props = SensorProperties(
                positioning=s.get("positioning", ["gps"]),
                perception=s.get("perception", [])
            )
        return self._sensor_props
    
    @property
    def capability_properties(self) -> CapabilityProperties:
        """Get capability properties as dataclass."""
        if self._capability_props is None:
            c = self.capabilities
            self._capability_props = CapabilityProperties(
                payload_capacity=c.get("payload_capacity", 0.0),
                manipulation=c.get("manipulation", False),
                can_land=c.get("can_land", True),
                can_dock=c.get("can_dock", False)
            )
        return self._capability_props
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PlatformSpecification:
        """Load platform specification from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert platform_type string to enum if needed
        if isinstance(data.get("platform_type"), str):
            data["platform_type"] = PlatformType(data["platform_type"])
        
        return cls(**data)
    
    @classmethod
    def from_platform_type(cls, platform_type: PlatformType, platform_id: str = None) -> PlatformSpecification:
        """Create a default specification for a platform type."""
        platform_id = platform_id or f"{platform_type.value}_default"
        
        # Default configurations by platform type
        defaults = {
            PlatformType.MICRO_UAV: {
                "physical": {
                    "mass": 0.03,
                    "dimensions": {"length": 0.1, "width": 0.1, "height": 0.05},
                    "collision_envelope": {"type": "ellipsoid", "semi_axes": [0.05, 0.05, 0.03]}
                },
                "dynamic": {
                    "max_velocity": [1.0, 1.0, 0.5],
                    "max_acceleration": [2.0, 2.0, 1.0],
                    "max_angular_velocity": 3.14
                },
                "energy": {
                    "battery_capacity_wh": 0.3,
                    "cruise_power": 2.0,
                    "max_power": 5.0
                },
                "sensors": {"range": 4.0},
            },
            PlatformType.SMALL_UAV: {
                "physical": {
                    "mass": 0.5,
                    "dimensions": {"length": 0.3, "width": 0.3, "height": 0.15},
                    "collision_envelope": {"type": "ellipsoid", "semi_axes": [0.15, 0.15, 0.08]}
                },
                "dynamic": {
                    "max_velocity": [3.0, 3.0, 1.5],
                    "max_acceleration": [4.0, 4.0, 2.0],
                    "max_angular_velocity": 2.0
                },
                "energy": {
                    "battery_capacity_wh": 20.0,
                    "cruise_power": 15.0,
                    "max_power": 60.0
                },
                "sensors": {"range": 10.0},
            },
            PlatformType.SMALL_UGV: {
                "physical": {
                    "mass": 1.5,
                    "dimensions": {"length": 0.25, "width": 0.2, "height": 0.15},
                    "collision_envelope": {"type": "box", "semi_axes": [0.125, 0.1, 0.075]}
                },
                "dynamic": {
                    "max_velocity": [0.5, 0.3, 0.0],
                    "max_acceleration": [1.0, 0.5, 0.0],
                    "max_angular_velocity": 1.5
                },
                "energy": {
                    "battery_capacity_wh": 30.0,
                    "cruise_power": 5.0,
                    "max_power": 20.0
                },
                "sensors": {"range": 5.0},
            },
        }
        
        # Get defaults or use generic
        config = defaults.get(platform_type, defaults[PlatformType.SMALL_UAV])
        
        return cls(
            platform_id=platform_id,
            platform_type=platform_type,
            physical=config.get("physical", {}),
            dynamic=config.get("dynamic", {}),
            energy=config.get("energy", {}),
            sensors=config.get("sensors", {}),
        )
    
    # Convenience properties for common values
    @property
    def max_velocity(self) -> float:
        """Get max velocity magnitude."""
        dp = self.dynamic_properties
        return max(dp.max_velocity.x, dp.max_velocity.y, dp.max_velocity.z)
    
    @property
    def collision_envelope(self) -> CollisionEnvelope:
        """Get collision envelope."""
        return self.physical_properties.collision_envelope
    
    @property
    def battery_capacity_wh(self) -> float:
        """Get battery capacity in Wh."""
        return self.energy.get("battery_capacity_wh", 10.0)
    
    @property
    def cruise_power_consumption(self) -> float:
        """Get cruise power consumption in W."""
        return self.energy.get("cruise_power", 10.0)
    
    @property
    def sensor_range(self) -> float:
        """Get sensor range in m."""
        return self.sensors.get("range", 5.0)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save platform specification to YAML file."""
        data = self.dict()
        data["platform_type"] = self.platform_type.value
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


class PlatformRegistry:
    """
    Registry for managing platform specifications.
    
    Singleton pattern to ensure consistent platform specs across the system.
    """
    
    _instance: Optional[PlatformRegistry] = None
    _specifications: Dict[str, PlatformSpecification] = {}
    
    def __new__(cls) -> PlatformRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._specifications = {}
        return cls._instance
    
    def register(self, spec: PlatformSpecification) -> None:
        """Register a platform specification."""
        self._specifications[spec.platform_id] = spec
        logger.info(f"Registered platform: {spec.platform_id} ({spec.platform_type.value})")
    
    def get(self, platform_id: str) -> Optional[PlatformSpecification]:
        """Get a platform specification by ID."""
        return self._specifications.get(platform_id)
    
    def get_by_type(self, platform_type: PlatformType) -> List[PlatformSpecification]:
        """Get all specifications for a platform type."""
        return [
            spec for spec in self._specifications.values()
            if spec.platform_type == platform_type
        ]
    
    def load_from_directory(self, directory: Union[str, Path]) -> int:
        """
        Load all platform specifications from a directory.
        
        Args:
            directory: Path to directory containing YAML files
            
        Returns:
            Number of specifications loaded
        """
        directory = Path(directory)
        count = 0
        
        for yaml_file in directory.glob("*.yaml"):
            try:
                spec = PlatformSpecification.from_yaml(yaml_file)
                self.register(spec)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        return count
    
    def list_platforms(self) -> List[str]:
        """List all registered platform IDs."""
        return list(self._specifications.keys())
    
    def clear(self) -> None:
        """Clear all registered specifications."""
        self._specifications.clear()


class PlatformFactory:
    """
    Factory for creating platform-related instances.
    
    Creates agents, digital twins, controllers, and dynamics models
    based on platform specifications.
    """
    
    def __init__(self, registry: Optional[PlatformRegistry] = None):
        """
        Initialize factory with a registry.
        
        Args:
            registry: Platform registry to use. If None, uses global registry.
        """
        self.registry = registry or PlatformRegistry()
    
    def get_platform_spec(self, platform_id: str) -> PlatformSpecification:
        """Get platform specification, raising error if not found."""
        spec = self.registry.get(platform_id)
        if spec is None:
            raise ValueError(f"Unknown platform: {platform_id}")
        return spec
    
    def create_dynamics_model(
        self,
        platform_id: str
    ) -> Dict[str, Any]:
        """
        Create dynamics model parameters for a platform.
        
        Returns a dictionary with A, B matrices for linear approximation
        and nonlinear dynamics parameters.
        """
        spec = self.get_platform_spec(platform_id)
        dyn = spec.dynamic_properties
        phys = spec.physical_properties
        
        # Simple double integrator model for position control
        # State: [x, y, z, vx, vy, vz]
        # Input: [ax, ay, az]
        dt = spec.dynamic_properties.response_time
        
        A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        
        B = np.array([
            [0.5*dt**2, 0, 0],
            [0, 0.5*dt**2, 0],
            [0, 0, 0.5*dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ])
        
        return {
            "A": A,
            "B": B,
            "mass": phys.mass,
            "inertia": phys.inertia_matrix,
            "max_velocity": dyn.max_velocity.to_array(),
            "max_acceleration": dyn.max_acceleration.to_array(),
            "max_angular_velocity": dyn.max_angular_velocity,
            "response_time": dyn.response_time,
            "drag_coefficient": dyn.drag_coefficient,
        }
    
    def get_interaction_parameters(
        self,
        platform_type_i: PlatformType,
        platform_type_j: PlatformType
    ) -> Dict[str, float]:
        """
        Get interaction parameters for a pair of platform types.
        
        Used in HAIM (Heterogeneous Agent Interaction Model).
        
        Args:
            platform_type_i: First platform type
            platform_type_j: Second platform type
            
        Returns:
            Dictionary with repulsion gain, friction parameters, etc.
        """
        # Platform-pair repulsion gain matrix (from architecture spec)
        repulsion_matrix = {
            (PlatformType.MICRO_UAV, PlatformType.MICRO_UAV): 1.0,
            (PlatformType.MICRO_UAV, PlatformType.SMALL_UAV): 1.2,
            (PlatformType.MICRO_UAV, PlatformType.MEDIUM_UAV): 1.5,
            (PlatformType.MICRO_UAV, PlatformType.LARGE_UAV): 2.0,
            (PlatformType.MICRO_UAV, PlatformType.SMALL_UGV): 0.8,
            (PlatformType.MICRO_UAV, PlatformType.MEDIUM_UGV): 1.0,
            
            (PlatformType.SMALL_UAV, PlatformType.SMALL_UAV): 1.0,
            (PlatformType.SMALL_UAV, PlatformType.MEDIUM_UAV): 1.3,
            (PlatformType.SMALL_UAV, PlatformType.LARGE_UAV): 1.8,
            (PlatformType.SMALL_UAV, PlatformType.SMALL_UGV): 0.9,
            (PlatformType.SMALL_UAV, PlatformType.MEDIUM_UGV): 1.1,
            
            (PlatformType.MEDIUM_UAV, PlatformType.MEDIUM_UAV): 1.0,
            (PlatformType.MEDIUM_UAV, PlatformType.LARGE_UAV): 1.5,
            (PlatformType.MEDIUM_UAV, PlatformType.SMALL_UGV): 1.0,
            (PlatformType.MEDIUM_UAV, PlatformType.MEDIUM_UGV): 1.2,
            
            (PlatformType.LARGE_UAV, PlatformType.LARGE_UAV): 1.0,
            (PlatformType.LARGE_UAV, PlatformType.SMALL_UGV): 1.2,
            (PlatformType.LARGE_UAV, PlatformType.MEDIUM_UGV): 1.4,
            
            (PlatformType.SMALL_UGV, PlatformType.SMALL_UGV): 1.0,
            (PlatformType.SMALL_UGV, PlatformType.MEDIUM_UGV): 1.2,
            
            (PlatformType.MEDIUM_UGV, PlatformType.MEDIUM_UGV): 1.0,
        }
        
        # Get repulsion gain (symmetric lookup)
        key = (platform_type_i, platform_type_j)
        if key not in repulsion_matrix:
            key = (platform_type_j, platform_type_i)
        
        p_rep = repulsion_matrix.get(key, 1.0)
        
        # Downwash factor for UAV-UAV pairs
        downwash = 0.0
        if platform_type_i.is_aerial and platform_type_j.is_aerial:
            # Larger UAVs have more downwash effect
            size_factor_i = {
                PlatformType.MICRO_UAV: 0.02,
                PlatformType.SMALL_UAV: 0.1,
                PlatformType.MEDIUM_UAV: 0.3,
                PlatformType.LARGE_UAV: 0.5,
            }
            size_factor_j = size_factor_i.copy()
            downwash = size_factor_i.get(platform_type_i, 0.1) + size_factor_j.get(platform_type_j, 0.1)
        
        # Safety margin based on platform types
        base_safety = 0.2  # 20cm base safety margin
        if platform_type_i.is_aerial != platform_type_j.is_aerial:
            # Cross-domain (UAV-UGV) needs larger margin
            base_safety = 0.5
        
        return {
            "repulsion_gain": p_rep,
            "downwash_factor": downwash,
            "safety_margin": base_safety,
            "friction_distance_factor": 1.5,  # R_friction = 1.5 * R_avoid
        }


# Default platform specifications for common platforms
def get_default_crazyflie_spec() -> PlatformSpecification:
    """Get default Crazyflie 2.1 specification."""
    return PlatformSpecification(
        platform_id="crazyflie_2_1",
        platform_type=PlatformType.MICRO_UAV,
        physical={
            "mass": 0.027,
            "dimensions": {"length": 0.092, "width": 0.092, "height": 0.029},
            "collision_envelope": {
                "type": "ellipsoid",
                "semi_axes": [0.05, 0.05, 0.03]
            }
        },
        dynamic={
            "max_velocity": [2.0, 2.0, 1.0],
            "max_acceleration": [6.0, 6.0, 4.0],
            "max_angular_velocity": 6.28,
            "response_time": 0.1
        },
        domain={
            "domain": "aerial",
            "altitude_range": [0.3, 3.0],
            "terrain_capability": []
        },
        communication={
            "protocols": ["crazyradio", "wifi"],
            "range": 100,
            "bandwidth": 2.0,
            "latency": 10
        },
        energy={
            "capacity": 0.94,
            "consumption_model": {
                "hover": 4.5,
                "cruise": 5.5,
                "max_power": 12.0
            }
        },
        sensors={
            "positioning": ["lighthouse", "uwb", "flow_deck"],
            "perception": ["multiranger"]
        },
        capabilities={
            "payload_capacity": 0.015,
            "manipulation": False,
            "can_land": True,
            "can_dock": False
        }
    )


def get_default_mentorpi_spec() -> PlatformSpecification:
    """Get default MentorPi UGV specification."""
    return PlatformSpecification(
        platform_id="mentorpi",
        platform_type=PlatformType.SMALL_UGV,
        physical={
            "mass": 1.5,
            "dimensions": {"length": 0.20, "width": 0.15, "height": 0.12},
            "collision_envelope": {
                "type": "cylinder",
                "semi_axes": [0.12, 0.12, 0.06]
            }
        },
        dynamic={
            "max_velocity": [0.5, 0.0, 0.0],
            "max_acceleration": [1.0, 0.0, 0.0],
            "max_angular_velocity": 2.0,
            "response_time": 0.2
        },
        domain={
            "domain": "ground",
            "altitude_range": [0.0, 0.0],
            "terrain_capability": ["flat", "rough"]
        },
        communication={
            "protocols": ["wifi"],
            "range": 50,
            "bandwidth": 10.0,
            "latency": 20
        },
        energy={
            "capacity": 22.2,
            "consumption_model": {
                "hover": 0.0,
                "cruise": 5.0,
                "max_power": 15.0
            }
        },
        sensors={
            "positioning": ["uwb", "wheel_odometry"],
            "perception": ["camera", "ultrasonic"]
        },
        capabilities={
            "payload_capacity": 0.5,
            "manipulation": True,
            "can_land": False,
            "can_dock": True
        }
    )


def register_default_platforms() -> None:
    """Register default platform specifications in the global registry."""
    registry = PlatformRegistry()
    registry.register(get_default_crazyflie_spec())
    registry.register(get_default_mentorpi_spec())
    logger.info("Registered default platform specifications")
