"""
Fleet-level Digital Twin for collective state management.

Maintains aggregate fleet state, formation tracking, and
collective behavior monitoring.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.digital_twin.agent_twin import AgentTwin, AgentTwinState, TwinStatus


@dataclass
class FleetMetrics:
    """Aggregate metrics for the fleet."""
    
    # Size metrics
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    
    # By platform type
    agents_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Energy metrics
    avg_battery_level: float = 0.0
    min_battery_level: float = 0.0
    total_remaining_energy_wh: float = 0.0
    
    # Position metrics
    centroid: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    spread_radius: float = 0.0
    bounding_box: Tuple[Vector3, Vector3] = field(
        default_factory=lambda: (Vector3(0, 0, 0), Vector3(0, 0, 0))
    )
    
    # Communication
    avg_communication_quality: float = 0.0
    disconnected_agents: int = 0
    
    # Performance
    avg_velocity: float = 0.0
    total_distance_traveled: float = 0.0
    
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "active_agents": self.active_agents,
            "avg_battery_level": self.avg_battery_level,
            "min_battery_level": self.min_battery_level,
            "centroid": [self.centroid.x, self.centroid.y, self.centroid.z],
            "spread_radius": self.spread_radius,
            "disconnected_agents": self.disconnected_agents,
            "timestamp": self.timestamp,
        }


@dataclass
class FormationState:
    """State of a formation."""
    
    formation_id: str = ""
    formation_type: str = "custom"  # line, v, circle, grid, custom
    
    # Members
    member_ids: List[str] = field(default_factory=list)
    leader_id: Optional[str] = None
    
    # Geometry
    target_positions: Dict[str, Vector3] = field(default_factory=dict)
    actual_positions: Dict[str, Vector3] = field(default_factory=dict)
    
    # Quality metrics
    formation_error: float = 0.0  # Average deviation from target
    max_error: float = 0.0
    is_formed: bool = False
    
    timestamp: float = field(default_factory=time.time)


class FormationTracker:
    """
    Tracks formation state and quality.
    """
    
    def __init__(self, formation_tolerance: float = 0.5):
        """
        Initialize formation tracker.
        
        Args:
            formation_tolerance: Maximum position error for "formed" status
        """
        self.formation_tolerance = formation_tolerance
        self._formations: Dict[str, FormationState] = {}
    
    def create_formation(
        self,
        formation_id: str,
        formation_type: str,
        member_ids: List[str],
        target_positions: Dict[str, Vector3],
        leader_id: str = None
    ) -> FormationState:
        """Create a new formation."""
        formation = FormationState(
            formation_id=formation_id,
            formation_type=formation_type,
            member_ids=member_ids,
            leader_id=leader_id or (member_ids[0] if member_ids else None),
            target_positions=target_positions,
        )
        self._formations[formation_id] = formation
        return formation
    
    def update_positions(
        self,
        formation_id: str,
        positions: Dict[str, Vector3]
    ) -> Optional[FormationState]:
        """Update actual positions for a formation."""
        formation = self._formations.get(formation_id)
        if formation is None:
            return None
        
        formation.actual_positions = positions
        formation.timestamp = time.time()
        
        # Compute errors
        errors = []
        for agent_id, target in formation.target_positions.items():
            if agent_id in positions:
                error = (positions[agent_id] - target).norm()
                errors.append(error)
        
        if errors:
            formation.formation_error = np.mean(errors)
            formation.max_error = max(errors)
            formation.is_formed = formation.max_error <= self.formation_tolerance
        
        return formation
    
    def get_formation(self, formation_id: str) -> Optional[FormationState]:
        """Get formation state."""
        return self._formations.get(formation_id)
    
    def remove_formation(self, formation_id: str) -> None:
        """Remove a formation."""
        self._formations.pop(formation_id, None)
    
    def generate_line_formation(
        self,
        member_ids: List[str],
        start_pos: Vector3,
        direction: Vector3,
        spacing: float
    ) -> Dict[str, Vector3]:
        """Generate target positions for line formation."""
        positions = {}
        dir_norm = direction.norm()
        if dir_norm < 1e-6:
            direction = Vector3(1, 0, 0)
        else:
            direction = Vector3(
                direction.x / dir_norm,
                direction.y / dir_norm,
                direction.z / dir_norm
            )
        
        for i, agent_id in enumerate(member_ids):
            positions[agent_id] = Vector3(
                start_pos.x + direction.x * spacing * i,
                start_pos.y + direction.y * spacing * i,
                start_pos.z + direction.z * spacing * i
            )
        
        return positions
    
    def generate_v_formation(
        self,
        member_ids: List[str],
        leader_pos: Vector3,
        direction: Vector3,
        spacing: float,
        angle: float = 45.0
    ) -> Dict[str, Vector3]:
        """Generate target positions for V formation."""
        positions = {}
        if not member_ids:
            return positions
        
        # Leader at front
        positions[member_ids[0]] = leader_pos
        
        # Wings
        angle_rad = np.radians(angle)
        for i, agent_id in enumerate(member_ids[1:], 1):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            
            # Offset from leader
            back_offset = row * spacing * np.cos(angle_rad)
            side_offset = row * spacing * np.sin(angle_rad) * side
            
            positions[agent_id] = Vector3(
                leader_pos.x - direction.x * back_offset + direction.y * side_offset,
                leader_pos.y - direction.y * back_offset - direction.x * side_offset,
                leader_pos.z
            )
        
        return positions
    
    def generate_circle_formation(
        self,
        member_ids: List[str],
        center: Vector3,
        radius: float
    ) -> Dict[str, Vector3]:
        """Generate target positions for circle formation."""
        positions = {}
        n = len(member_ids)
        
        for i, agent_id in enumerate(member_ids):
            angle = 2 * np.pi * i / n
            positions[agent_id] = Vector3(
                center.x + radius * np.cos(angle),
                center.y + radius * np.sin(angle),
                center.z
            )
        
        return positions


class FleetTwin:
    """
    Fleet-level digital twin.
    
    Aggregates agent twins and provides fleet-wide state,
    metrics, and formation tracking.
    """
    
    def __init__(self):
        """Initialize fleet twin."""
        self._agent_twins: Dict[str, AgentTwin] = {}
        self._metrics = FleetMetrics()
        self._formation_tracker = FormationTracker()
        
        # Metrics history
        self._metrics_history: List[FleetMetrics] = []
        self._history_size = 1000
        
        # Callbacks
        self._metrics_callbacks: List[Callable[[FleetMetrics], None]] = []
    
    def add_agent_twin(self, twin: AgentTwin) -> None:
        """Add an agent twin to the fleet."""
        self._agent_twins[twin.agent_id] = twin
        self._update_metrics()
    
    def remove_agent_twin(self, agent_id: str) -> Optional[AgentTwin]:
        """Remove an agent twin from the fleet."""
        twin = self._agent_twins.pop(agent_id, None)
        self._update_metrics()
        return twin
    
    def get_agent_twin(self, agent_id: str) -> Optional[AgentTwin]:
        """Get an agent twin by ID."""
        return self._agent_twins.get(agent_id)
    
    def get_all_twins(self) -> List[AgentTwin]:
        """Get all agent twins."""
        return list(self._agent_twins.values())
    
    def get_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return list(self._agent_twins.keys())
    
    @property
    def metrics(self) -> FleetMetrics:
        """Get current fleet metrics."""
        return self._metrics
    
    @property
    def formation_tracker(self) -> FormationTracker:
        """Get formation tracker."""
        return self._formation_tracker
    
    def update(self) -> FleetMetrics:
        """Update fleet metrics from all agent twins."""
        self._update_metrics()
        
        # Store in history
        self._metrics_history.append(self._metrics)
        if len(self._metrics_history) > self._history_size:
            self._metrics_history = self._metrics_history[-self._history_size:]
        
        # Trigger callbacks
        for callback in self._metrics_callbacks:
            callback(self._metrics)
        
        return self._metrics
    
    def _update_metrics(self) -> None:
        """Update fleet metrics."""
        twins = list(self._agent_twins.values())
        
        if not twins:
            self._metrics = FleetMetrics()
            return
        
        # Helper to get mode name
        def get_mode_name(mode):
            if hasattr(mode, 'name'):
                return mode.name
            return str(mode)
        
        # Count agents
        total = len(twins)
        active = sum(1 for t in twins if get_mode_name(t.state.mode) not in ["IDLE", "EMERGENCY", "1", "9"])
        idle = sum(1 for t in twins if get_mode_name(t.state.mode) in ["IDLE", "1"])
        disconnected = sum(1 for t in twins if t.status == TwinStatus.DISCONNECTED)
        
        # By platform type
        by_type: Dict[str, int] = {}
        for twin in twins:
            type_name = twin.platform_spec.platform_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        # Energy metrics
        battery_levels = [t.state.battery_level for t in twins]
        avg_battery = np.mean(battery_levels)
        min_battery = min(battery_levels)
        total_energy = sum(t.estimate_remaining_energy() for t in twins)
        
        # Position metrics
        positions = [t.state.position for t in twins]
        centroid = Vector3(
            np.mean([p.x for p in positions]),
            np.mean([p.y for p in positions]),
            np.mean([p.z for p in positions])
        )
        
        distances = [(p - centroid).norm() for p in positions]
        spread = max(distances) if distances else 0.0
        
        # Bounding box
        xs = [p.x for p in positions]
        ys = [p.y for p in positions]
        zs = [p.z for p in positions]
        bbox_min = Vector3(min(xs), min(ys), min(zs))
        bbox_max = Vector3(max(xs), max(ys), max(zs))
        
        # Communication
        comm_qualities = [t.state.communication_quality for t in twins]
        avg_comm = np.mean(comm_qualities)
        
        # Velocity
        velocities = [t.state.velocity.norm() for t in twins]
        avg_velocity = np.mean(velocities)
        
        self._metrics = FleetMetrics(
            total_agents=total,
            active_agents=active,
            idle_agents=idle,
            agents_by_type=by_type,
            avg_battery_level=avg_battery,
            min_battery_level=min_battery,
            total_remaining_energy_wh=total_energy,
            centroid=centroid,
            spread_radius=spread,
            bounding_box=(bbox_min, bbox_max),
            avg_communication_quality=avg_comm,
            disconnected_agents=disconnected,
            avg_velocity=avg_velocity,
            timestamp=time.time(),
        )
    
    def get_positions(self) -> Dict[str, Vector3]:
        """Get all agent positions."""
        return {aid: twin.state.position for aid, twin in self._agent_twins.items()}
    
    def get_velocities(self) -> Dict[str, Vector3]:
        """Get all agent velocities."""
        return {aid: twin.state.velocity for aid, twin in self._agent_twins.items()}
    
    def get_agents_in_radius(self, center: Vector3, radius: float) -> List[str]:
        """Get agents within radius of a point."""
        result = []
        for agent_id, twin in self._agent_twins.items():
            dist = (twin.state.position - center).norm()
            if dist <= radius:
                result.append(agent_id)
        return result
    
    def get_nearest_agent(self, position: Vector3, exclude: Set[str] = None) -> Optional[str]:
        """Get nearest agent to a position."""
        exclude = exclude or set()
        
        nearest = None
        min_dist = float('inf')
        
        for agent_id, twin in self._agent_twins.items():
            if agent_id in exclude:
                continue
            dist = (twin.state.position - position).norm()
            if dist < min_dist:
                min_dist = dist
                nearest = agent_id
        
        return nearest
    
    def get_agents_by_type(self, platform_type: PlatformType) -> List[str]:
        """Get agents of a specific platform type."""
        return [
            aid for aid, twin in self._agent_twins.items()
            if twin.platform_spec.platform_type == platform_type
        ]
    
    def get_available_agents(self) -> List[str]:
        """Get agents that are available for tasks."""
        def is_idle(mode):
            if hasattr(mode, 'name'):
                return mode.name == "IDLE"
            return str(mode) == "1"  # IDLE enum value
        
        return [
            aid for aid, twin in self._agent_twins.items()
            if is_idle(twin.state.mode) and twin.status == TwinStatus.SYNCHRONIZED
        ]
    
    def register_metrics_callback(self, callback: Callable[[FleetMetrics], None]) -> None:
        """Register callback for metrics updates."""
        self._metrics_callbacks.append(callback)
    
    def predict_fleet_state(self, dt: float) -> Dict[str, AgentTwinState]:
        """Predict fleet state at future time."""
        predictions = {}
        for agent_id, twin in self._agent_twins.items():
            predictions[agent_id] = twin.predict_state(dt)
        return predictions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fleet statistics."""
        return {
            "total_agents": len(self._agent_twins),
            "metrics": self._metrics.to_dict(),
            "formations": len(self._formation_tracker._formations),
            "history_size": len(self._metrics_history),
        }
