"""
Simulation Engine for HeteroFleet.

Main simulation loop and orchestration for multi-agent simulation.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.core.state import AgentState
from heterofleet.simulation.environment import SimulationEnvironment, EnvironmentConfig
from heterofleet.simulation.agent_sim import SimulatedAgent
from heterofleet.digital_twin.fleet_twin import FleetTwin
from heterofleet.digital_twin.agent_twin import AgentTwin
from heterofleet.coordination.haim import HAIMCoordinator


class SimulationStatus(Enum):
    """Status of simulation."""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    
    # Timing
    time_step: float = 0.01  # seconds
    real_time_factor: float = 1.0  # 1.0 = real-time, >1 = faster
    max_duration: float = 300.0  # seconds
    
    # Agents
    num_uavs: int = 5
    num_ugvs: int = 2
    
    # Environment
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Features
    enable_coordination: bool = True
    enable_collision_avoidance: bool = True
    enable_communication_sim: bool = True
    
    # Logging
    log_interval: float = 1.0  # seconds
    record_trajectory: bool = True


@dataclass
class SimulationState:
    """Current state of simulation."""
    
    status: SimulationStatus = SimulationStatus.STOPPED
    sim_time: float = 0.0
    wall_time: float = 0.0
    step_count: int = 0
    
    # Agents
    num_agents: int = 0
    active_agents: int = 0
    
    # Performance
    avg_step_time_ms: float = 0.0
    real_time_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "step_count": self.step_count,
            "num_agents": self.num_agents,
            "avg_step_time_ms": self.avg_step_time_ms,
            "real_time_ratio": self.real_time_ratio,
        }


class SimulationEngine:
    """
    Main simulation engine for HeteroFleet.
    
    Orchestrates multi-agent simulation including:
    - Agent dynamics
    - Coordination algorithms
    - Environment updates
    - Digital twin synchronization
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize simulation engine.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        
        # Environment
        self.environment = SimulationEnvironment(self.config.environment_config)
        
        # Agents
        self._agents: Dict[str, SimulatedAgent] = {}
        
        # Digital twin
        self.fleet_twin = FleetTwin()
        
        # Coordination
        self._coordinator: Optional[HAIMCoordinator] = None
        if self.config.enable_coordination:
            self._coordinator = HAIMCoordinator()
        
        # State
        self._state = SimulationState()
        self._start_wall_time = 0.0
        
        # History
        self._trajectory_history: Dict[str, List[Vector3]] = {}
        self._state_history: List[Dict[str, AgentState]] = []
        
        # Callbacks
        self._step_callbacks: List[Callable[[float, Dict[str, AgentState]], None]] = []
        self._event_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    @property
    def state(self) -> SimulationState:
        """Get current simulation state."""
        return self._state
    
    @property
    def sim_time(self) -> float:
        """Get current simulation time."""
        return self._state.sim_time
    
    def add_agent(
        self,
        agent_id: str,
        platform_spec: PlatformSpecification,
        initial_position: Vector3 = None
    ) -> SimulatedAgent:
        """Add an agent to the simulation."""
        agent = SimulatedAgent(
            agent_id=agent_id,
            platform_spec=platform_spec,
            environment=self.environment,
            initial_position=initial_position
        )
        
        self._agents[agent_id] = agent
        
        # Create digital twin
        twin = AgentTwin(agent_id, platform_spec)
        self.fleet_twin.add_agent_twin(twin)
        
        # Register with coordinator
        if self._coordinator:
            self._coordinator.update_agent(
                agent_id,
                platform_spec.platform_type,
                platform_spec.collision_envelope
            )
        
        # Initialize trajectory tracking
        if self.config.record_trajectory:
            self._trajectory_history[agent_id] = []
        
        self._state.num_agents = len(self._agents)
        
        logger.info(f"Added agent {agent_id} at {initial_position}")
        
        return agent
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the simulation."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.fleet_twin.remove_agent_twin(agent_id)
            
            if agent_id in self._trajectory_history:
                del self._trajectory_history[agent_id]
            
            self._state.num_agents = len(self._agents)
            
            logger.info(f"Removed agent {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[SimulatedAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def setup_default_scenario(self) -> None:
        """Set up default scenario with multiple agents."""
        # Create UAVs
        for i in range(self.config.num_uavs):
            platform_type = PlatformType.SMALL_UAV
            spec = PlatformSpecification.from_platform_type(platform_type)
            
            # Spread out initial positions
            angle = 2 * np.pi * i / self.config.num_uavs
            radius = 5.0
            pos = Vector3(
                radius * np.cos(angle),
                radius * np.sin(angle),
                1.0
            )
            
            self.add_agent(f"uav_{i}", spec, pos)
        
        # Create UGVs
        for i in range(self.config.num_ugvs):
            platform_type = PlatformType.SMALL_UGV
            spec = PlatformSpecification.from_platform_type(platform_type)
            
            pos = Vector3(
                -5.0 + i * 2.0,
                -5.0,
                0.0
            )
            
            self.add_agent(f"ugv_{i}", spec, pos)
        
        # Add some obstacles
        self.environment.generate_random_obstacles(10)
        
        logger.info(f"Set up default scenario with {len(self._agents)} agents")
    
    def start(self) -> None:
        """Start the simulation."""
        if self._state.status == SimulationStatus.RUNNING:
            return
        
        self._state.status = SimulationStatus.RUNNING
        self._start_wall_time = time.time()
        
        logger.info("Simulation started")
        self._emit_event("simulation_started", {})
    
    def pause(self) -> None:
        """Pause the simulation."""
        if self._state.status == SimulationStatus.RUNNING:
            self._state.status = SimulationStatus.PAUSED
            logger.info("Simulation paused")
            self._emit_event("simulation_paused", {})
    
    def resume(self) -> None:
        """Resume the simulation."""
        if self._state.status == SimulationStatus.PAUSED:
            self._state.status = SimulationStatus.RUNNING
            logger.info("Simulation resumed")
            self._emit_event("simulation_resumed", {})
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._state.status = SimulationStatus.STOPPED
        logger.info("Simulation stopped")
        self._emit_event("simulation_stopped", {})
    
    def step(self) -> Dict[str, AgentState]:
        """
        Execute one simulation step.
        
        Returns:
            Dictionary of agent states
        """
        if self._state.status != SimulationStatus.RUNNING:
            return {}
        
        step_start = time.time()
        dt = self.config.time_step
        
        # Get current states for coordination
        states = {}
        positions = {}
        velocities = {}
        
        for agent_id, agent in self._agents.items():
            states[agent_id] = agent.get_state()
            positions[agent_id] = agent.position
            velocities[agent_id] = agent.velocity
        
        # Compute coordination velocities
        if self._coordinator and self.config.enable_coordination:
            # Update coordinator with current states
            for agent_id, agent in self._agents.items():
                self._coordinator.neighbor_tracker.update_agent(
                    agent_id,
                    states[agent_id],
                    agent.platform_spec
                )
            self._coordinator.neighbor_tracker.compute_neighbors()
            
            for agent_id, agent in self._agents.items():
                # Get target position for coordination
                target_pos = agent._target_position or agent.position
                
                # Compute coordinated velocity
                interaction = self._coordinator.compute_interaction(
                    agent_id,
                    target_pos,
                    dt=dt
                )
                
                if interaction and interaction.desired_velocity:
                    # Apply coordinated velocity
                    agent.set_target_velocity(interaction.desired_velocity)
        
        # Update all agents
        for agent_id, agent in self._agents.items():
            agent.update(dt)
            
            # Record trajectory
            if self.config.record_trajectory:
                self._trajectory_history[agent_id].append(
                    Vector3(agent.position.x, agent.position.y, agent.position.z)
                )
        
        # Update environment
        self.environment.update(dt)
        
        # Update digital twins
        for agent_id, agent in self._agents.items():
            twin = self.fleet_twin.get_agent_twin(agent_id)
            if twin:
                twin.update_from_agent_state(agent.get_state())
        
        self.fleet_twin.update()
        
        # Get final states
        final_states = {aid: agent.get_state() for aid, agent in self._agents.items()}
        
        # Record state history
        if self.config.record_trajectory:
            self._state_history.append(final_states)
        
        # Update simulation state
        self._state.sim_time += dt
        self._state.step_count += 1
        self._state.wall_time = time.time() - self._start_wall_time
        
        step_time = (time.time() - step_start) * 1000
        self._state.avg_step_time_ms = (
            self._state.avg_step_time_ms * 0.95 + step_time * 0.05
        )
        
        if self._state.wall_time > 0:
            self._state.real_time_ratio = self._state.sim_time / self._state.wall_time
        
        # Trigger callbacks
        for callback in self._step_callbacks:
            callback(self._state.sim_time, final_states)
        
        # Check termination
        if self._state.sim_time >= self.config.max_duration:
            self._state.status = SimulationStatus.FINISHED
            self._emit_event("simulation_finished", {"reason": "max_duration"})
        
        return final_states
    
    def run(self, duration: float = None) -> None:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Duration to run (None = until max_duration)
        """
        self.start()
        
        target_duration = duration or self.config.max_duration
        start_time = self._state.sim_time
        
        while (self._state.status == SimulationStatus.RUNNING and
               self._state.sim_time - start_time < target_duration):
            
            self.step()
            
            # Real-time pacing
            if self.config.real_time_factor < float('inf'):
                expected_wall_time = (self._state.sim_time - start_time) / self.config.real_time_factor
                actual_wall_time = time.time() - self._start_wall_time
                
                if actual_wall_time < expected_wall_time:
                    time.sleep(expected_wall_time - actual_wall_time)
        
        if self._state.status == SimulationStatus.RUNNING:
            self._state.status = SimulationStatus.FINISHED
    
    def set_agent_target(self, agent_id: str, target: Vector3) -> bool:
        """Set target position for an agent."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        
        agent.set_target_position(target)
        return True
    
    def set_agent_velocity(self, agent_id: str, velocity: Vector3) -> bool:
        """Set target velocity for an agent."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        
        agent.set_target_velocity(velocity)
        return True
    
    def register_step_callback(
        self,
        callback: Callable[[float, Dict[str, AgentState]], None]
    ) -> None:
        """Register callback for each simulation step."""
        self._step_callbacks.append(callback)
    
    def register_event_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Register callback for simulation events."""
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit simulation event."""
        for callback in self._event_callbacks:
            callback(event_type, data)
    
    def get_trajectory(self, agent_id: str) -> List[Vector3]:
        """Get recorded trajectory for an agent."""
        return self._trajectory_history.get(agent_id, [])
    
    def get_all_trajectories(self) -> Dict[str, List[Vector3]]:
        """Get all recorded trajectories."""
        return self._trajectory_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            "state": self._state.to_dict(),
            "environment": self.environment.get_statistics(),
            "fleet": self.fleet_twin.get_statistics(),
            "num_trajectory_points": sum(
                len(t) for t in self._trajectory_history.values()
            ),
        }
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._state = SimulationState()
        self._trajectory_history = {aid: [] for aid in self._agents}
        self._state_history = []
        
        # Reset agent positions (would need to store initial positions)
        for agent in self._agents.values():
            agent._time = 0.0
            agent._battery_level = 1.0
        
        logger.info("Simulation reset")
