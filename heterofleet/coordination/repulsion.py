"""
Extended repulsion force calculations for heterogeneous agents.

Implements platform-pair specific repulsion gains as described in
the HAIM (Heterogeneous Agent Interaction Model).

Key features:
- Asymmetric repulsion based on platform types
- Downwash effects for UAV-UAV pairs
- Cross-domain safety margins (UAV-UGV)

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3, CollisionEnvelope


@dataclass
class RepulsionGainMatrix:
    """
    Platform-pair repulsion gain matrix P.
    
    P[τ_i][τ_j] gives the repulsion gain for platform type τ_i
    when interacting with platform type τ_j.
    
    The matrix is generally asymmetric: a large UAV repels a micro UAV
    more strongly than vice versa.
    """
    
    _gains: Dict[Tuple[PlatformType, PlatformType], float] = field(default_factory=dict)
    default_gain: float = 1.0
    
    def __post_init__(self):
        """Initialize default gain matrix."""
        if not self._gains:
            self._initialize_default_gains()
    
    def _initialize_default_gains(self) -> None:
        """Initialize the default platform-pair repulsion gains."""
        # UAV-UAV interactions
        uav_types = [
            PlatformType.MICRO_UAV,
            PlatformType.SMALL_UAV,
            PlatformType.MEDIUM_UAV,
            PlatformType.LARGE_UAV,
        ]
        
        uav_base_gains = {
            PlatformType.MICRO_UAV: 1.0,
            PlatformType.SMALL_UAV: 1.2,
            PlatformType.MEDIUM_UAV: 1.5,
            PlatformType.LARGE_UAV: 2.0,
        }
        
        for i, type_i in enumerate(uav_types):
            for j, type_j in enumerate(uav_types):
                if j > i:
                    gain = uav_base_gains[type_j] / uav_base_gains[type_i]
                elif i > j:
                    gain = 1.0 / (uav_base_gains[type_i] / uav_base_gains[type_j])
                else:
                    gain = 1.0
                self._gains[(type_i, type_j)] = gain
        
        # UGV-UGV interactions
        ugv_types = [
            PlatformType.SMALL_UGV,
            PlatformType.MEDIUM_UGV,
            PlatformType.LARGE_UGV,
        ]
        
        ugv_base_gains = {
            PlatformType.SMALL_UGV: 1.0,
            PlatformType.MEDIUM_UGV: 1.3,
            PlatformType.LARGE_UGV: 1.6,
        }
        
        for type_i in ugv_types:
            for type_j in ugv_types:
                gain = ugv_base_gains.get(type_j, 1.0) / ugv_base_gains.get(type_i, 1.0)
                self._gains[(type_i, type_j)] = gain
        
        # UAV-UGV cross-domain interactions
        for uav_type in uav_types:
            for ugv_type in ugv_types:
                self._gains[(uav_type, ugv_type)] = 0.8 * uav_base_gains[uav_type]
                self._gains[(ugv_type, uav_type)] = 1.2 * ugv_base_gains[ugv_type]
        
        # USV interactions
        usv_types = [PlatformType.SMALL_USV, PlatformType.MEDIUM_USV]
        for usv_type in usv_types:
            for other_type in list(PlatformType):
                if other_type not in usv_types:
                    self._gains[(usv_type, other_type)] = 0.5
                    self._gains[(other_type, usv_type)] = 0.5
                else:
                    self._gains[(usv_type, other_type)] = 1.0
    
    def get_gain(self, type_i: PlatformType, type_j: PlatformType) -> float:
        """Get repulsion gain for platform pair."""
        return self._gains.get((type_i, type_j), self.default_gain)
    
    def set_gain(self, type_i: PlatformType, type_j: PlatformType, gain: float) -> None:
        """Set repulsion gain for a platform pair."""
        self._gains[(type_i, type_j)] = gain
    
    def to_matrix(self, platform_order: List[PlatformType] = None) -> np.ndarray:
        """Convert to numpy matrix for analysis."""
        if platform_order is None:
            platform_order = list(PlatformType)
        
        n = len(platform_order)
        matrix = np.zeros((n, n))
        
        for i, type_i in enumerate(platform_order):
            for j, type_j in enumerate(platform_order):
                matrix[i, j] = self.get_gain(type_i, type_j)
        
        return matrix


@dataclass
class DownwashModel:
    """Model for UAV downwash effects on vertical separation."""
    
    _factors: Dict[PlatformType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default downwash factors."""
        if not self._factors:
            self._factors = {
                PlatformType.MICRO_UAV: 0.02,
                PlatformType.SMALL_UAV: 0.1,
                PlatformType.MEDIUM_UAV: 0.3,
                PlatformType.LARGE_UAV: 0.5,
            }
    
    def get_factor(self, platform_type: PlatformType) -> float:
        """Get downwash factor for a platform type."""
        return self._factors.get(platform_type, 0.0)
    
    def compute_combined_downwash(
        self,
        type_i: PlatformType,
        type_j: PlatformType
    ) -> float:
        """Compute combined downwash factor for two UAVs."""
        if not type_i.is_aerial or not type_j.is_aerial:
            return 0.0
        
        return self._factors.get(type_i, 0.0) + self._factors.get(type_j, 0.0)


@dataclass
class SafetyMarginModel:
    """Model for platform-pair specific safety margins."""
    
    base_margin_aerial_aerial: float = 0.2
    base_margin_ground_ground: float = 0.3
    base_margin_cross_domain: float = 0.5
    velocity_margin_factor: float = 0.1
    
    def get_safety_margin(
        self,
        type_i: PlatformType,
        type_j: PlatformType,
        relative_velocity: float = 0.0
    ) -> float:
        """Get safety margin for platform pair."""
        if type_i.is_aerial and type_j.is_aerial:
            base = self.base_margin_aerial_aerial
        elif type_i.is_ground and type_j.is_ground:
            base = self.base_margin_ground_ground
        else:
            base = self.base_margin_cross_domain
        
        velocity_margin = self.velocity_margin_factor * relative_velocity
        
        return base + velocity_margin


class RepulsionCalculator:
    """
    Calculator for extended repulsion forces in HAIM.
    
    Implements the repulsion force:
    v_rep_i = Σ_j P[τ_i][τ_j] · max(0, R₀(i,j) - d_ij) · û_ij^rep
    """
    
    def __init__(
        self,
        gain_matrix: RepulsionGainMatrix = None,
        downwash_model: DownwashModel = None,
        safety_model: SafetyMarginModel = None,
        max_repulsion_magnitude: float = 5.0,
        repulsion_gain: float = 2.0
    ):
        """Initialize repulsion calculator."""
        self.gain_matrix = gain_matrix or RepulsionGainMatrix()
        self.downwash_model = downwash_model or DownwashModel()
        self.safety_model = safety_model or SafetyMarginModel()
        self.max_repulsion_magnitude = max_repulsion_magnitude
        self.repulsion_gain = repulsion_gain
    
    def compute_combined_radius(
        self,
        type_i: PlatformType,
        type_j: PlatformType,
        envelope_i: CollisionEnvelope,
        envelope_j: CollisionEnvelope,
        relative_velocity: float = 0.0
    ) -> float:
        """
        Compute combined avoidance radius R₀(i,j).
        
        R₀(i,j) = r_env(i) + r_env(j) + r_safety(τ_i, τ_j) + d_downwash
        """
        r_i = max(envelope_i.semi_axes)
        r_j = max(envelope_j.semi_axes)
        r_safety = self.safety_model.get_safety_margin(type_i, type_j, relative_velocity)
        d_downwash = self.downwash_model.compute_combined_downwash(type_i, type_j)
        
        return r_i + r_j + r_safety + d_downwash
    
    def compute_repulsion_force(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        agent_envelope: CollisionEnvelope,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        neighbor_type: PlatformType,
        neighbor_envelope: CollisionEnvelope
    ) -> Vector3:
        """Compute repulsion force from a single neighbor."""
        diff = agent_pos - neighbor_pos
        distance = diff.norm()
        
        if distance < 1e-6:
            random_dir = Vector3(
                np.random.randn(),
                np.random.randn(),
                np.random.randn()
            ).normalized()
            return random_dir * self.max_repulsion_magnitude
        
        unit_repulsion = diff.normalized()
        rel_vel = (agent_vel - neighbor_vel).norm()
        
        R0 = self.compute_combined_radius(
            agent_type, neighbor_type,
            agent_envelope, neighbor_envelope,
            rel_vel
        )
        
        penetration = R0 - distance
        
        if penetration <= 0:
            return Vector3(0, 0, 0)
        
        P_ij = self.gain_matrix.get_gain(agent_type, neighbor_type)
        magnitude = self.repulsion_gain * P_ij * penetration
        magnitude = min(magnitude, self.max_repulsion_magnitude)
        
        return unit_repulsion * magnitude
    
    def compute_repulsion_3d(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        agent_envelope: CollisionEnvelope,
        neighbor_pos: Vector3,
        neighbor_vel: Vector3,
        neighbor_type: PlatformType,
        neighbor_envelope: CollisionEnvelope
    ) -> Vector3:
        """Compute 3D repulsion considering vertical separation for UAVs."""
        diff = agent_pos - neighbor_pos
        horizontal_diff = Vector3(diff.x, diff.y, 0)
        vertical_diff = diff.z
        
        horizontal_distance = horizontal_diff.norm()
        
        if agent_type.is_aerial and neighbor_type.is_aerial:
            r_h_i = max(agent_envelope.semi_axes[0], agent_envelope.semi_axes[1])
            r_h_j = max(neighbor_envelope.semi_axes[0], neighbor_envelope.semi_axes[1])
            R0_horizontal = r_h_i + r_h_j + self.safety_model.base_margin_aerial_aerial
            
            r_v_i = agent_envelope.semi_axes[2]
            r_v_j = neighbor_envelope.semi_axes[2]
            downwash = self.downwash_model.compute_combined_downwash(agent_type, neighbor_type)
            R0_vertical = r_v_i + r_v_j + downwash + 0.1
            
            h_penetration = max(0, R0_horizontal - horizontal_distance) / R0_horizontal
            v_penetration = max(0, R0_vertical - abs(vertical_diff)) / R0_vertical
            
            if h_penetration > 0 and v_penetration > 0:
                P_ij = self.gain_matrix.get_gain(agent_type, neighbor_type)
                
                if horizontal_distance > 1e-6:
                    h_dir = horizontal_diff.normalized()
                else:
                    h_dir = Vector3(np.random.randn(), np.random.randn(), 0).normalized()
                
                h_magnitude = self.repulsion_gain * P_ij * h_penetration * (R0_horizontal - horizontal_distance)
                h_repulsion = h_dir * min(h_magnitude, self.max_repulsion_magnitude)
                
                v_sign = 1.0 if vertical_diff >= 0 else -1.0
                v_magnitude = self.repulsion_gain * P_ij * v_penetration * (R0_vertical - abs(vertical_diff))
                v_repulsion = Vector3(0, 0, v_sign * min(v_magnitude, self.max_repulsion_magnitude))
                
                return h_repulsion + v_repulsion
            
            return Vector3(0, 0, 0)
        
        else:
            return self.compute_repulsion_force(
                agent_pos, agent_vel, agent_type, agent_envelope,
                neighbor_pos, neighbor_vel, neighbor_type, neighbor_envelope
            )
    
    def compute_total_repulsion(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        agent_type: PlatformType,
        agent_envelope: CollisionEnvelope,
        neighbors: List[Tuple[Vector3, Vector3, PlatformType, CollisionEnvelope]]
    ) -> Vector3:
        """Compute total repulsion from all neighbors."""
        total_repulsion = Vector3(0, 0, 0)
        
        for neighbor_pos, neighbor_vel, neighbor_type, neighbor_envelope in neighbors:
            repulsion = self.compute_repulsion_3d(
                agent_pos, agent_vel, agent_type, agent_envelope,
                neighbor_pos, neighbor_vel, neighbor_type, neighbor_envelope
            )
            total_repulsion = total_repulsion + repulsion
        
        magnitude = total_repulsion.norm()
        if magnitude > self.max_repulsion_magnitude:
            total_repulsion = total_repulsion * (self.max_repulsion_magnitude / magnitude)
        
        return total_repulsion
