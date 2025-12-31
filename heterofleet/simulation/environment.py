"""
Simulation Environment for HeteroFleet.

Defines the simulated world including obstacles, boundaries,
communication zones, and environmental conditions.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import Vector3


class ObstacleType(Enum):
    """Types of obstacles."""
    STATIC = auto()
    DYNAMIC = auto()
    NO_FLY_ZONE = auto()
    BUILDING = auto()
    TERRAIN = auto()


@dataclass
class Obstacle:
    """An obstacle in the environment."""
    
    obstacle_id: str = ""
    obstacle_type: ObstacleType = ObstacleType.STATIC
    
    # Geometry
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    dimensions: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))  # half-extents
    radius: float = 0.0  # For spherical obstacles
    is_spherical: bool = False
    
    # For dynamic obstacles
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    
    # Properties
    is_solid: bool = True
    height: float = 0.0  # For 2.5D representation
    
    def contains_point(self, point: Vector3) -> bool:
        """Check if point is inside obstacle."""
        if self.is_spherical:
            dist = (point - self.position).norm()
            return dist <= self.radius
        else:
            # Axis-aligned bounding box
            dx = abs(point.x - self.position.x)
            dy = abs(point.y - self.position.y)
            dz = abs(point.z - self.position.z)
            return (dx <= self.dimensions.x and 
                    dy <= self.dimensions.y and 
                    dz <= self.dimensions.z)
    
    def distance_to_point(self, point: Vector3) -> float:
        """Get distance from point to obstacle surface."""
        if self.is_spherical:
            return max(0, (point - self.position).norm() - self.radius)
        else:
            # Distance to AABB
            dx = max(0, abs(point.x - self.position.x) - self.dimensions.x)
            dy = max(0, abs(point.y - self.position.y) - self.dimensions.y)
            dz = max(0, abs(point.z - self.position.z) - self.dimensions.z)
            return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def update(self, dt: float) -> None:
        """Update dynamic obstacle."""
        if self.obstacle_type == ObstacleType.DYNAMIC:
            self.position = Vector3(
                self.position.x + self.velocity.x * dt,
                self.position.y + self.velocity.y * dt,
                self.position.z + self.velocity.z * dt
            )


@dataclass
class CommunicationZone:
    """A zone with specific communication properties."""
    
    zone_id: str = ""
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    radius: float = 10.0
    
    # Communication quality (0-1)
    quality: float = 1.0
    
    # Interference
    interference_level: float = 0.0
    
    def get_quality_at(self, point: Vector3) -> float:
        """Get communication quality at a point."""
        dist = (point - self.position).norm()
        if dist > self.radius:
            return 0.0
        
        # Linear falloff
        factor = 1.0 - dist / self.radius
        return self.quality * factor * (1.0 - self.interference_level)


@dataclass
class EnvironmentConfig:
    """Configuration for simulation environment."""
    
    # World bounds
    bounds_min: Vector3 = field(default_factory=lambda: Vector3(-50, -50, 0))
    bounds_max: Vector3 = field(default_factory=lambda: Vector3(50, 50, 20))
    
    # Grid resolution for spatial queries
    grid_resolution: float = 1.0
    
    # Environmental conditions
    wind_velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    wind_variability: float = 0.1
    
    # Communication
    base_comm_quality: float = 1.0
    comm_range: float = 50.0
    
    # Physics
    gravity: float = 9.81
    air_density: float = 1.225  # kg/m^3


class SimulationEnvironment:
    """
    Simulation environment for heterogeneous fleet testing.
    
    Manages obstacles, communication zones, and environmental conditions.
    """
    
    def __init__(self, config: EnvironmentConfig = None):
        """
        Initialize simulation environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or EnvironmentConfig()
        
        # Obstacles
        self._obstacles: Dict[str, Obstacle] = {}
        
        # Communication zones
        self._comm_zones: Dict[str, CommunicationZone] = {}
        
        # Spatial index for obstacles
        self._obstacle_grid: Dict[Tuple[int, int, int], List[str]] = {}
        
        # Time
        self._time = 0.0
    
    @property
    def time(self) -> float:
        """Get current simulation time."""
        return self._time
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the environment."""
        self._obstacles[obstacle.obstacle_id] = obstacle
        self._update_obstacle_grid(obstacle)
    
    def remove_obstacle(self, obstacle_id: str) -> None:
        """Remove an obstacle from the environment."""
        if obstacle_id in self._obstacles:
            obstacle = self._obstacles.pop(obstacle_id)
            self._remove_from_grid(obstacle)
    
    def get_obstacle(self, obstacle_id: str) -> Optional[Obstacle]:
        """Get obstacle by ID."""
        return self._obstacles.get(obstacle_id)
    
    def get_all_obstacles(self) -> List[Obstacle]:
        """Get all obstacles."""
        return list(self._obstacles.values())
    
    def add_comm_zone(self, zone: CommunicationZone) -> None:
        """Add a communication zone."""
        self._comm_zones[zone.zone_id] = zone
    
    def remove_comm_zone(self, zone_id: str) -> None:
        """Remove a communication zone."""
        self._comm_zones.pop(zone_id, None)
    
    def _update_obstacle_grid(self, obstacle: Obstacle) -> None:
        """Update spatial grid for obstacle."""
        cell_size = self.config.grid_resolution
        
        # Get cells covered by obstacle
        if obstacle.is_spherical:
            min_x = int((obstacle.position.x - obstacle.radius) / cell_size)
            max_x = int((obstacle.position.x + obstacle.radius) / cell_size)
            min_y = int((obstacle.position.y - obstacle.radius) / cell_size)
            max_y = int((obstacle.position.y + obstacle.radius) / cell_size)
            min_z = int((obstacle.position.z - obstacle.radius) / cell_size)
            max_z = int((obstacle.position.z + obstacle.radius) / cell_size)
        else:
            min_x = int((obstacle.position.x - obstacle.dimensions.x) / cell_size)
            max_x = int((obstacle.position.x + obstacle.dimensions.x) / cell_size)
            min_y = int((obstacle.position.y - obstacle.dimensions.y) / cell_size)
            max_y = int((obstacle.position.y + obstacle.dimensions.y) / cell_size)
            min_z = int((obstacle.position.z - obstacle.dimensions.z) / cell_size)
            max_z = int((obstacle.position.z + obstacle.dimensions.z) / cell_size)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    cell = (x, y, z)
                    if cell not in self._obstacle_grid:
                        self._obstacle_grid[cell] = []
                    if obstacle.obstacle_id not in self._obstacle_grid[cell]:
                        self._obstacle_grid[cell].append(obstacle.obstacle_id)
    
    def _remove_from_grid(self, obstacle: Obstacle) -> None:
        """Remove obstacle from spatial grid."""
        for cell in self._obstacle_grid.values():
            if obstacle.obstacle_id in cell:
                cell.remove(obstacle.obstacle_id)
    
    def is_position_valid(self, position: Vector3, radius: float = 0.0) -> bool:
        """Check if position is valid (within bounds, not in obstacle)."""
        # Check bounds
        if not self._is_within_bounds(position):
            return False
        
        # Check obstacles
        for obstacle in self._obstacles.values():
            dist = obstacle.distance_to_point(position)
            if dist < radius:
                return False
        
        return True
    
    def _is_within_bounds(self, position: Vector3) -> bool:
        """Check if position is within world bounds."""
        return (self.config.bounds_min.x <= position.x <= self.config.bounds_max.x and
                self.config.bounds_min.y <= position.y <= self.config.bounds_max.y and
                self.config.bounds_min.z <= position.z <= self.config.bounds_max.z)
    
    def get_obstacles_near(self, position: Vector3, radius: float) -> List[Obstacle]:
        """Get obstacles within radius of position."""
        result = []
        
        for obstacle in self._obstacles.values():
            dist = obstacle.distance_to_point(position)
            if dist <= radius:
                result.append(obstacle)
        
        return result
    
    def raycast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 100.0
    ) -> Optional[Tuple[Vector3, str]]:
        """
        Cast ray and find first obstacle hit.
        
        Returns:
            Tuple of (hit_point, obstacle_id) or None
        """
        # Normalize direction
        dir_norm = direction.norm()
        if dir_norm < 1e-6:
            return None
        direction = Vector3(
            direction.x / dir_norm,
            direction.y / dir_norm,
            direction.z / dir_norm
        )
        
        # Step along ray
        step_size = 0.1
        num_steps = int(max_distance / step_size)
        
        for i in range(num_steps):
            t = i * step_size
            point = Vector3(
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t
            )
            
            for obstacle in self._obstacles.values():
                if obstacle.contains_point(point):
                    return (point, obstacle.obstacle_id)
        
        return None
    
    def get_communication_quality(self, position: Vector3) -> float:
        """Get communication quality at position."""
        base_quality = self.config.base_comm_quality
        
        # Check communication zones
        for zone in self._comm_zones.values():
            zone_quality = zone.get_quality_at(position)
            if zone_quality > 0:
                base_quality = max(base_quality, zone_quality)
        
        # Check if blocked by obstacles
        # Simplified - would need proper LOS calculation
        
        return base_quality
    
    def get_wind_at(self, position: Vector3) -> Vector3:
        """Get wind velocity at position."""
        base_wind = self.config.wind_velocity
        
        # Add variability
        variability = self.config.wind_variability
        noise = Vector3(
            np.random.normal(0, variability),
            np.random.normal(0, variability),
            np.random.normal(0, variability * 0.5)
        )
        
        # Height-dependent wind (increases with altitude)
        height_factor = 1.0 + position.z * 0.02
        
        return Vector3(
            (base_wind.x + noise.x) * height_factor,
            (base_wind.y + noise.y) * height_factor,
            base_wind.z + noise.z
        )
    
    def update(self, dt: float) -> None:
        """Update environment state."""
        self._time += dt
        
        # Update dynamic obstacles
        for obstacle in self._obstacles.values():
            if obstacle.obstacle_type == ObstacleType.DYNAMIC:
                obstacle.update(dt)
                self._update_obstacle_grid(obstacle)
    
    def generate_random_obstacles(
        self,
        num_obstacles: int,
        min_size: float = 0.5,
        max_size: float = 3.0
    ) -> None:
        """Generate random obstacles in environment."""
        for i in range(num_obstacles):
            pos = Vector3(
                np.random.uniform(self.config.bounds_min.x + max_size,
                                 self.config.bounds_max.x - max_size),
                np.random.uniform(self.config.bounds_min.y + max_size,
                                 self.config.bounds_max.y - max_size),
                np.random.uniform(0, self.config.bounds_max.z * 0.5)
            )
            
            size = np.random.uniform(min_size, max_size)
            
            obstacle = Obstacle(
                obstacle_id=f"obstacle_{i}",
                obstacle_type=ObstacleType.STATIC,
                position=pos,
                radius=size,
                is_spherical=True
            )
            
            self.add_obstacle(obstacle)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "num_obstacles": len(self._obstacles),
            "num_comm_zones": len(self._comm_zones),
            "bounds": {
                "min": [self.config.bounds_min.x, self.config.bounds_min.y, self.config.bounds_min.z],
                "max": [self.config.bounds_max.x, self.config.bounds_max.y, self.config.bounds_max.z],
            },
            "wind": [self.config.wind_velocity.x, self.config.wind_velocity.y, self.config.wind_velocity.z],
            "time": self._time,
        }
