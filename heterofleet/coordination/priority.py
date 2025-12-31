"""
Priority hierarchy management for heterogeneous agents.

Implements priority rules and resolution mechanisms for
coordinating between agents of different types.

Key features:
- Platform-type based priority hierarchy
- Task-based priority assignment
- Dynamic priority adjustment
- Conflict resolution

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.core.state import AgentMode


class PriorityRule(Enum):
    """Priority rule types."""
    SIZE_HIERARCHY = auto()      # Larger platforms have priority
    ENERGY_CRITICAL = auto()     # Low energy has priority (to return home)
    TASK_PRIORITY = auto()       # Higher task priority wins
    EMERGENCY = auto()           # Emergency mode has highest priority
    FORMATION_LEADER = auto()    # Formation leaders have priority
    RIGHT_OF_WAY = auto()        # Traffic-like right-of-way rules
    FIRST_COME = auto()          # First to claim has priority


@dataclass
class PriorityConfig:
    """Configuration for priority system."""
    
    # Weight for each rule in combined priority
    rule_weights: Dict[PriorityRule, float] = field(default_factory=dict)
    
    # Platform type base priorities (0-1)
    platform_priorities: Dict[PlatformType, float] = field(default_factory=dict)
    
    # Energy threshold for critical priority
    energy_critical_threshold: float = 0.15
    
    # Emergency priority boost
    emergency_priority_boost: float = 0.5
    
    # Formation leader priority boost
    formation_leader_boost: float = 0.2
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.rule_weights:
            self.rule_weights = {
                PriorityRule.EMERGENCY: 0.3,
                PriorityRule.ENERGY_CRITICAL: 0.2,
                PriorityRule.TASK_PRIORITY: 0.2,
                PriorityRule.SIZE_HIERARCHY: 0.15,
                PriorityRule.FORMATION_LEADER: 0.1,
                PriorityRule.RIGHT_OF_WAY: 0.05,
            }
        
        if not self.platform_priorities:
            self.platform_priorities = {
                # Larger/less agile platforms get higher base priority
                PlatformType.LARGE_UAV: 0.8,
                PlatformType.MEDIUM_UAV: 0.6,
                PlatformType.SMALL_UAV: 0.4,
                PlatformType.MICRO_UAV: 0.3,
                PlatformType.LARGE_UGV: 0.9,
                PlatformType.MEDIUM_UGV: 0.7,
                PlatformType.SMALL_UGV: 0.5,
                PlatformType.MEDIUM_USV: 0.7,
                PlatformType.SMALL_USV: 0.5,
            }


@dataclass
class AgentPriorityInfo:
    """Priority information for an agent."""
    
    agent_id: str
    platform_type: PlatformType
    position: Vector3
    velocity: Vector3
    
    # State
    mode: AgentMode = AgentMode.IDLE
    energy_level: float = 1.0
    task_priority: float = 0.5
    
    # Formation
    is_formation_leader: bool = False
    formation_id: Optional[str] = None
    
    # Computed priority
    computed_priority: float = 0.5
    
    # Priority breakdown
    priority_components: Dict[PriorityRule, float] = field(default_factory=dict)


@dataclass
class PriorityResolution:
    """Result of priority resolution between two agents."""
    
    agent_a_id: str
    agent_b_id: str
    
    # Which agent has priority
    winner_id: str
    yielder_id: str
    
    # Priority scores
    priority_a: float
    priority_b: float
    
    # Resolution details
    dominant_rule: PriorityRule
    resolution_confidence: float  # 0-1, higher = clearer winner
    
    # Recommended actions
    yielder_action: str  # "slow", "stop", "avoid", "wait"
    yield_factor: float  # How much to yield (0-1)


class PriorityManager:
    """
    Manager for computing and resolving agent priorities.
    
    Handles priority computation based on multiple rules and
    resolves conflicts between agents.
    """
    
    def __init__(self, config: PriorityConfig = None):
        """Initialize priority manager."""
        self.config = config or PriorityConfig()
        
        # Cache for priority computations
        self._priority_cache: Dict[str, AgentPriorityInfo] = {}
    
    def compute_platform_priority(self, platform_type: PlatformType) -> float:
        """Compute base priority from platform type."""
        return self.config.platform_priorities.get(platform_type, 0.5)
    
    def compute_energy_priority(self, energy_level: float) -> float:
        """
        Compute energy-based priority.
        
        Low energy agents get higher priority to return home.
        """
        if energy_level <= self.config.energy_critical_threshold:
            # Critical - high priority to reach safety
            return 1.0
        elif energy_level <= self.config.energy_critical_threshold * 2:
            # Low - elevated priority
            return 0.7
        else:
            # Normal - no energy-based priority
            return 0.0
    
    def compute_emergency_priority(self, mode: AgentMode) -> float:
        """Compute emergency-based priority."""
        if mode == AgentMode.EMERGENCY:
            return 1.0
        elif mode == AgentMode.RETURN:
            return 0.5
        else:
            return 0.0
    
    def compute_right_of_way(
        self,
        agent_pos: Vector3,
        agent_vel: Vector3,
        other_pos: Vector3,
        other_vel: Vector3
    ) -> float:
        """
        Compute right-of-way priority.
        
        Based on traffic rules - agent on the right has priority.
        """
        # Vector from agent to other
        to_other = other_pos - agent_pos
        
        # Cross product to determine relative position
        cross = agent_vel.cross(to_other)
        
        # Positive z means other is on the right (has priority)
        # Negative z means agent is on the right (has priority)
        if agent_vel.norm() < 1e-6:
            return 0.5  # Stationary - neutral
        
        # Scale by confidence (how perpendicular is the approach)
        approach_angle = abs(cross.z) / (agent_vel.norm() * to_other.norm() + 1e-6)
        
        if cross.z > 0:
            # Other is on right - agent should yield
            return 0.3 * approach_angle
        else:
            # Agent is on right - agent has priority
            return 0.7 + 0.3 * approach_angle
    
    def compute_priority(
        self,
        agent_info: AgentPriorityInfo,
        other_agents: List[AgentPriorityInfo] = None
    ) -> float:
        """
        Compute overall priority for an agent.
        
        Args:
            agent_info: Agent priority information
            other_agents: Optional list of other agents for context
            
        Returns:
            Computed priority (0-1)
        """
        components = {}
        
        # Platform-based priority
        components[PriorityRule.SIZE_HIERARCHY] = self.compute_platform_priority(
            agent_info.platform_type
        )
        
        # Energy-based priority
        components[PriorityRule.ENERGY_CRITICAL] = self.compute_energy_priority(
            agent_info.energy_level
        )
        
        # Emergency priority
        components[PriorityRule.EMERGENCY] = self.compute_emergency_priority(
            agent_info.mode
        )
        
        # Task priority
        components[PriorityRule.TASK_PRIORITY] = agent_info.task_priority
        
        # Formation leader
        if agent_info.is_formation_leader:
            components[PriorityRule.FORMATION_LEADER] = 1.0
        else:
            components[PriorityRule.FORMATION_LEADER] = 0.0
        
        # Right of way (average against all others)
        if other_agents:
            row_scores = []
            for other in other_agents:
                row = self.compute_right_of_way(
                    agent_info.position, agent_info.velocity,
                    other.position, other.velocity
                )
                row_scores.append(row)
            components[PriorityRule.RIGHT_OF_WAY] = np.mean(row_scores)
        else:
            components[PriorityRule.RIGHT_OF_WAY] = 0.5
        
        # Weighted combination
        total_weight = sum(self.config.rule_weights.values())
        priority = 0.0
        
        for rule, weight in self.config.rule_weights.items():
            component_value = components.get(rule, 0.0)
            priority += weight * component_value
        
        priority /= total_weight
        
        # Store components
        agent_info.priority_components = components
        agent_info.computed_priority = priority
        
        return priority
    
    def resolve_conflict(
        self,
        agent_a: AgentPriorityInfo,
        agent_b: AgentPriorityInfo
    ) -> PriorityResolution:
        """
        Resolve priority conflict between two agents.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            
        Returns:
            Priority resolution result
        """
        # Compute priorities
        priority_a = self.compute_priority(agent_a, [agent_b])
        priority_b = self.compute_priority(agent_b, [agent_a])
        
        # Determine winner
        if priority_a > priority_b:
            winner_id = agent_a.agent_id
            yielder_id = agent_b.agent_id
        else:
            winner_id = agent_b.agent_id
            yielder_id = agent_a.agent_id
        
        # Find dominant rule
        dominant_rule = PriorityRule.SIZE_HIERARCHY
        max_diff = 0.0
        
        for rule in PriorityRule:
            comp_a = agent_a.priority_components.get(rule, 0.5)
            comp_b = agent_b.priority_components.get(rule, 0.5)
            diff = abs(comp_a - comp_b) * self.config.rule_weights.get(rule, 0.0)
            
            if diff > max_diff:
                max_diff = diff
                dominant_rule = rule
        
        # Resolution confidence
        priority_diff = abs(priority_a - priority_b)
        confidence = min(1.0, priority_diff * 2)  # Scale to 0-1
        
        # Recommended action based on dominant rule
        if dominant_rule == PriorityRule.EMERGENCY:
            action = "stop"  # Clear path for emergency
        elif dominant_rule == PriorityRule.ENERGY_CRITICAL:
            action = "avoid"  # Let low-energy agent pass
        elif dominant_rule == PriorityRule.RIGHT_OF_WAY:
            action = "slow"  # Yield right of way
        else:
            action = "slow"  # Default
        
        # Yield factor based on priority difference
        yield_factor = min(1.0, priority_diff + 0.3)  # At least 30% yield
        
        return PriorityResolution(
            agent_a_id=agent_a.agent_id,
            agent_b_id=agent_b.agent_id,
            winner_id=winner_id,
            yielder_id=yielder_id,
            priority_a=priority_a,
            priority_b=priority_b,
            dominant_rule=dominant_rule,
            resolution_confidence=confidence,
            yielder_action=action,
            yield_factor=yield_factor
        )
    
    def compute_yield_velocity_adjustment(
        self,
        agent_vel: Vector3,
        resolution: PriorityResolution,
        agent_is_yielder: bool
    ) -> Vector3:
        """
        Compute velocity adjustment based on priority resolution.
        
        Args:
            agent_vel: Current agent velocity
            resolution: Priority resolution result
            agent_is_yielder: Whether this agent should yield
            
        Returns:
            Adjusted velocity
        """
        if not agent_is_yielder:
            return agent_vel
        
        action = resolution.yielder_action
        factor = resolution.yield_factor
        
        if action == "stop":
            return Vector3(0, 0, 0)
        elif action == "slow":
            # Reduce speed
            return agent_vel * (1.0 - 0.5 * factor)
        elif action == "avoid":
            # Reduce speed and prepare to turn
            return agent_vel * (1.0 - 0.3 * factor)
        elif action == "wait":
            return Vector3(0, 0, 0)
        else:
            return agent_vel * (1.0 - 0.3 * factor)
    
    def batch_compute_priorities(
        self,
        agents: List[AgentPriorityInfo]
    ) -> Dict[str, float]:
        """
        Compute priorities for multiple agents.
        
        Args:
            agents: List of agent priority information
            
        Returns:
            Dictionary mapping agent_id to priority
        """
        priorities = {}
        
        for agent in agents:
            others = [a for a in agents if a.agent_id != agent.agent_id]
            priority = self.compute_priority(agent, others)
            priorities[agent.agent_id] = priority
        
        return priorities
    
    def resolve_multi_agent_conflict(
        self,
        agents: List[AgentPriorityInfo]
    ) -> List[PriorityResolution]:
        """
        Resolve conflicts among multiple agents.
        
        Returns pairwise resolutions for all conflicting pairs.
        
        Args:
            agents: List of agent priority information
            
        Returns:
            List of priority resolutions
        """
        # First compute all priorities
        self.batch_compute_priorities(agents)
        
        # Generate pairwise resolutions
        resolutions = []
        
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents):
                if i >= j:
                    continue
                
                resolution = self.resolve_conflict(agent_a, agent_b)
                resolutions.append(resolution)
        
        return resolutions


class DynamicPriorityAdjuster:
    """
    Dynamic priority adjustment based on situation.
    
    Adjusts priorities in real-time based on:
    - Congestion levels
    - Deadlock detection
    - Fairness considerations
    """
    
    def __init__(
        self,
        priority_manager: PriorityManager,
        fairness_window: float = 60.0,  # seconds
        max_wait_time: float = 30.0
    ):
        """Initialize dynamic adjuster."""
        self.priority_manager = priority_manager
        self.fairness_window = fairness_window
        self.max_wait_time = max_wait_time
        
        # Track waiting times
        self._wait_times: Dict[str, float] = {}  # agent_id -> time waiting
        self._last_update: Dict[str, float] = {}  # agent_id -> last update time
        
        # Track yield history for fairness
        self._yield_counts: Dict[str, int] = {}  # agent_id -> yield count
    
    def update_wait_time(self, agent_id: str, is_waiting: bool, current_time: float) -> None:
        """Update wait time tracking for an agent."""
        last_time = self._last_update.get(agent_id, current_time)
        dt = current_time - last_time
        
        if is_waiting:
            current_wait = self._wait_times.get(agent_id, 0.0)
            self._wait_times[agent_id] = current_wait + dt
        else:
            self._wait_times[agent_id] = 0.0
        
        self._last_update[agent_id] = current_time
    
    def record_yield(self, agent_id: str) -> None:
        """Record that an agent yielded."""
        self._yield_counts[agent_id] = self._yield_counts.get(agent_id, 0) + 1
    
    def compute_fairness_boost(self, agent_id: str) -> float:
        """
        Compute priority boost for fairness.
        
        Agents that have yielded more get a boost.
        """
        yield_count = self._yield_counts.get(agent_id, 0)
        
        # Logarithmic boost to prevent runaway
        return min(0.2, 0.05 * np.log1p(yield_count))
    
    def compute_wait_time_boost(self, agent_id: str) -> float:
        """
        Compute priority boost based on wait time.
        
        Long-waiting agents get priority to prevent starvation.
        """
        wait_time = self._wait_times.get(agent_id, 0.0)
        
        if wait_time < 5.0:
            return 0.0
        elif wait_time < self.max_wait_time:
            return 0.1 * (wait_time - 5.0) / (self.max_wait_time - 5.0)
        else:
            return 0.2  # Max boost
    
    def detect_deadlock(
        self,
        agents: List[AgentPriorityInfo],
        positions_history: Dict[str, List[Vector3]],
        velocity_threshold: float = 0.05
    ) -> List[str]:
        """
        Detect agents potentially in deadlock.
        
        Args:
            agents: Agent information
            positions_history: Recent position history per agent
            velocity_threshold: Velocity below which agent is considered stuck
            
        Returns:
            List of agent IDs potentially in deadlock
        """
        stuck_agents = []
        
        for agent in agents:
            # Check if agent is barely moving
            if agent.velocity.norm() > velocity_threshold:
                continue
            
            # Check if agent has been stuck
            history = positions_history.get(agent.agent_id, [])
            if len(history) < 5:
                continue
            
            # Check position variation
            positions = np.array([[p.x, p.y, p.z] for p in history[-5:]])
            variation = np.std(positions, axis=0).sum()
            
            if variation < 0.1:  # Less than 10cm variation
                stuck_agents.append(agent.agent_id)
        
        return stuck_agents
    
    def adjust_priorities(
        self,
        agents: List[AgentPriorityInfo],
        current_time: float
    ) -> Dict[str, float]:
        """
        Compute adjusted priorities with dynamic factors.
        
        Args:
            agents: Agent information
            current_time: Current simulation time
            
        Returns:
            Dictionary of agent_id -> adjusted priority
        """
        # Base priorities
        base_priorities = self.priority_manager.batch_compute_priorities(agents)
        
        adjusted = {}
        
        for agent in agents:
            base = base_priorities[agent.agent_id]
            
            # Fairness boost
            fairness = self.compute_fairness_boost(agent.agent_id)
            
            # Wait time boost
            wait_boost = self.compute_wait_time_boost(agent.agent_id)
            
            # Combined
            adjusted[agent.agent_id] = min(1.0, base + fairness + wait_boost)
        
        return adjusted
    
    def reset_fairness(self) -> None:
        """Reset fairness tracking (e.g., at new mission)."""
        self._yield_counts.clear()
        self._wait_times.clear()
        self._last_update.clear()
