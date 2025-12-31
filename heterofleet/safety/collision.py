"""
Collision Checking and Avoidance for heterogeneous fleets.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3, CollisionEnvelope


class CollisionSeverity(Enum):
    """Severity of collision risk."""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class CollisionPrediction:
    """Prediction of potential collision."""
    agent_i: str
    agent_j: str
    time_to_collision: float
    collision_time: float
    closest_point_i: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    closest_point_j: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    min_distance: float = float('inf')
    severity: CollisionSeverity = CollisionSeverity.NONE
    relative_velocity: float = 0.0
    approach_angle: float = 0.0
    
    @property
    def is_collision_predicted(self) -> bool:
        return self.time_to_collision < float('inf')
    
    @property
    def is_critical(self) -> bool:
        return self.severity in (CollisionSeverity.HIGH, CollisionSeverity.CRITICAL)


@dataclass
class AvoidanceManeuver:
    """Recommended avoidance maneuver."""
    agent_id: str
    velocity_adjustment: Vector3
    start_time: float
    duration: float
    priority: float = 1.0
    reason: str = ""
    confidence: float = 1.0


class CollisionChecker:
    """Collision checker for heterogeneous agents."""
    
    def __init__(self, safety_margin: float = 0.1):
        self.safety_margin = safety_margin
    
    def check_collision(self, pos_i: Vector3, pos_j: Vector3,
                       envelope_i: CollisionEnvelope, envelope_j: CollisionEnvelope) -> Tuple[bool, float]:
        diff = pos_i - pos_j
        distance = diff.norm()
        r_i = max(envelope_i.semi_axes)
        r_j = max(envelope_j.semi_axes)
        min_dist = r_i + r_j + self.safety_margin
        return distance < min_dist, distance
    
    def compute_cpa(self, pos_i: Vector3, vel_i: Vector3,
                   pos_j: Vector3, vel_j: Vector3) -> Tuple[float, float, Vector3, Vector3]:
        rel_pos = np.array([pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z])
        rel_vel = np.array([vel_i.x - vel_j.x, vel_i.y - vel_j.y, vel_i.z - vel_j.z])
        
        vel_sq = np.dot(rel_vel, rel_vel)
        if vel_sq < 1e-10:
            t_cpa = 0.0
        else:
            t_cpa = max(0.0, -np.dot(rel_pos, rel_vel) / vel_sq)
        
        cpa_i = Vector3(pos_i.x + vel_i.x * t_cpa, pos_i.y + vel_i.y * t_cpa, pos_i.z + vel_i.z * t_cpa)
        cpa_j = Vector3(pos_j.x + vel_j.x * t_cpa, pos_j.y + vel_j.y * t_cpa, pos_j.z + vel_j.z * t_cpa)
        cpa_distance = (cpa_i - cpa_j).norm()
        
        return t_cpa, cpa_distance, cpa_i, cpa_j
    
    def predict_collision(self, agent_i: str, pos_i: Vector3, vel_i: Vector3, envelope_i: CollisionEnvelope,
                         agent_j: str, pos_j: Vector3, vel_j: Vector3, envelope_j: CollisionEnvelope,
                         horizon: float = 10.0, current_time: float = 0.0) -> CollisionPrediction:
        t_cpa, cpa_dist, cpa_i, cpa_j = self.compute_cpa(pos_i, vel_i, pos_j, vel_j)
        
        r_i = max(envelope_i.semi_axes)
        r_j = max(envelope_j.semi_axes)
        collision_dist = r_i + r_j + self.safety_margin
        
        if cpa_dist < collision_dist and t_cpa <= horizon:
            t_collision = self._find_collision_time(pos_i, vel_i, pos_j, vel_j, collision_dist)
        else:
            t_collision = float('inf')
        
        rel_vel = Vector3(vel_i.x - vel_j.x, vel_i.y - vel_j.y, vel_i.z - vel_j.z)
        rel_vel_mag = rel_vel.norm()
        
        severity = self._compute_severity(t_collision, cpa_dist, collision_dist, rel_vel_mag)
        
        return CollisionPrediction(
            agent_i=agent_i, agent_j=agent_j,
            time_to_collision=t_collision, collision_time=current_time + t_collision,
            closest_point_i=cpa_i, closest_point_j=cpa_j, min_distance=cpa_dist,
            severity=severity, relative_velocity=rel_vel_mag
        )
    
    def _find_collision_time(self, pos_i: Vector3, vel_i: Vector3,
                            pos_j: Vector3, vel_j: Vector3, collision_dist: float) -> float:
        rel_pos = np.array([pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z])
        rel_vel = np.array([vel_i.x - vel_j.x, vel_i.y - vel_j.y, vel_i.z - vel_j.z])
        
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - collision_dist ** 2
        
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0 or a < 1e-10:
            return float('inf')
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return float('inf')
    
    def _compute_severity(self, time_to_collision: float, cpa_dist: float,
                         collision_dist: float, rel_velocity: float) -> CollisionSeverity:
        if time_to_collision == float('inf'):
            return CollisionSeverity.LOW if cpa_dist < collision_dist * 1.5 else CollisionSeverity.NONE
        if time_to_collision < 0.5:
            return CollisionSeverity.CRITICAL
        elif time_to_collision < 2.0:
            return CollisionSeverity.HIGH
        elif time_to_collision < 5.0:
            return CollisionSeverity.MEDIUM
        return CollisionSeverity.LOW
    
    def check_fleet_collisions(self, positions: Dict[str, Vector3], velocities: Dict[str, Vector3],
                               envelopes: Dict[str, CollisionEnvelope], horizon: float = 10.0,
                               current_time: float = 0.0) -> List[CollisionPrediction]:
        predictions = []
        agent_ids = list(positions.keys())
        
        for i, id_i in enumerate(agent_ids):
            for j, id_j in enumerate(agent_ids):
                if i >= j:
                    continue
                pred = self.predict_collision(
                    id_i, positions[id_i], velocities[id_i], envelopes[id_i],
                    id_j, positions[id_j], velocities[id_j], envelopes[id_j],
                    horizon, current_time
                )
                if pred.severity != CollisionSeverity.NONE:
                    predictions.append(pred)
        
        severity_order = {CollisionSeverity.CRITICAL: 0, CollisionSeverity.HIGH: 1,
                         CollisionSeverity.MEDIUM: 2, CollisionSeverity.LOW: 3}
        predictions.sort(key=lambda p: (severity_order[p.severity], p.time_to_collision))
        return predictions


class CollisionAvoidance:
    """Collision avoidance algorithm."""
    
    def __init__(self, max_avoidance_velocity: float = 2.0, min_separation: float = 0.5):
        self.max_avoidance_velocity = max_avoidance_velocity
        self.min_separation = min_separation
        self.collision_checker = CollisionChecker(safety_margin=min_separation)
    
    def compute_avoidance_velocity(self, agent_id: str, agent_pos: Vector3,
                                   agent_vel: Vector3, predictions: List[CollisionPrediction]) -> Vector3:
        relevant = [p for p in predictions if p.agent_i == agent_id or p.agent_j == agent_id]
        if not relevant:
            return Vector3(0, 0, 0)
        
        avoidance = np.zeros(3)
        for pred in relevant:
            if not pred.is_collision_predicted:
                continue
            
            if pred.agent_i == agent_id:
                other_cpa, my_cpa = pred.closest_point_j, pred.closest_point_i
            else:
                other_cpa, my_cpa = pred.closest_point_i, pred.closest_point_j
            
            away = np.array([my_cpa.x - other_cpa.x, my_cpa.y - other_cpa.y, my_cpa.z - other_cpa.z])
            away_norm = np.linalg.norm(away)
            if away_norm < 1e-6:
                away = np.random.randn(3)
                away_norm = np.linalg.norm(away)
            away = away / away_norm
            
            urgency = 1.0 / max(0.1, pred.time_to_collision)
            severity_scale = {CollisionSeverity.LOW: 0.25, CollisionSeverity.MEDIUM: 0.5,
                            CollisionSeverity.HIGH: 0.75, CollisionSeverity.CRITICAL: 1.0}
            scale = severity_scale.get(pred.severity, 0.0)
            
            avoidance += away * urgency * scale * self.max_avoidance_velocity
        
        avoidance_norm = np.linalg.norm(avoidance)
        if avoidance_norm > self.max_avoidance_velocity:
            avoidance = avoidance / avoidance_norm * self.max_avoidance_velocity
        
        return Vector3(avoidance[0], avoidance[1], avoidance[2])
    
    def generate_maneuvers(self, predictions: List[CollisionPrediction], positions: Dict[str, Vector3],
                          velocities: Dict[str, Vector3], priorities: Dict[str, float] = None,
                          current_time: float = 0.0) -> List[AvoidanceManeuver]:
        maneuvers = []
        priorities = priorities or {}
        
        agent_predictions: Dict[str, List[CollisionPrediction]] = {}
        for pred in predictions:
            for agent_id in [pred.agent_i, pred.agent_j]:
                if agent_id not in agent_predictions:
                    agent_predictions[agent_id] = []
                agent_predictions[agent_id].append(pred)
        
        for agent_id, preds in agent_predictions.items():
            if agent_id not in positions:
                continue
            
            avoidance = self.compute_avoidance_velocity(
                agent_id, positions[agent_id],
                velocities.get(agent_id, Vector3(0, 0, 0)), preds
            )
            
            if avoidance.norm() > 0.01:
                min_ttc = min(p.time_to_collision for p in preds if p.is_collision_predicted)
                maneuvers.append(AvoidanceManeuver(
                    agent_id=agent_id, velocity_adjustment=avoidance,
                    start_time=current_time, duration=min(min_ttc, 2.0),
                    priority=priorities.get(agent_id, 0.5),
                    reason=f"Avoiding {len(preds)} potential collisions"
                ))
        
        return maneuvers


class SpatialIndex:
    """Spatial indexing for efficient collision queries."""
    
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self._grid: Dict[Tuple[int, int, int], List[str]] = {}
        self._positions: Dict[str, Vector3] = {}
    
    def _get_cell(self, pos: Vector3) -> Tuple[int, int, int]:
        return (int(np.floor(pos.x / self.cell_size)),
                int(np.floor(pos.y / self.cell_size)),
                int(np.floor(pos.z / self.cell_size)))
    
    def insert(self, agent_id: str, position: Vector3) -> None:
        if agent_id in self._positions:
            old_cell = self._get_cell(self._positions[agent_id])
            if old_cell in self._grid and agent_id in self._grid[old_cell]:
                self._grid[old_cell].remove(agent_id)
        
        cell = self._get_cell(position)
        if cell not in self._grid:
            self._grid[cell] = []
        self._grid[cell].append(agent_id)
        self._positions[agent_id] = position
    
    def remove(self, agent_id: str) -> None:
        if agent_id in self._positions:
            cell = self._get_cell(self._positions[agent_id])
            if cell in self._grid and agent_id in self._grid[cell]:
                self._grid[cell].remove(agent_id)
            del self._positions[agent_id]
    
    def query_radius(self, position: Vector3, radius: float) -> List[str]:
        results = []
        cells_to_check = int(np.ceil(radius / self.cell_size)) + 1
        center_cell = self._get_cell(position)
        
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                for dz in range(-cells_to_check, cells_to_check + 1):
                    cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    if cell in self._grid:
                        for agent_id in self._grid[cell]:
                            if agent_id in self._positions:
                                dist = (self._positions[agent_id] - position).norm()
                                if dist <= radius:
                                    results.append(agent_id)
        return results
    
    def clear(self) -> None:
        self._grid.clear()
        self._positions.clear()
