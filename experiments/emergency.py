"""
Emergency Response Experiment for HeteroFleet.

Tests system response to emergency situations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.core.state import OperationalMode
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from experiments.base import ExperimentBase, ExperimentConfig


@dataclass
class EmergencyConfig(ExperimentConfig):
    """Configuration for emergency experiment."""
    
    name: str = "emergency"
    description: str = "Test emergency response capabilities"
    
    # Fleet
    num_uavs: int = 5
    num_ugvs: int = 2
    
    # Emergency scenarios
    scenario: str = "agent_failure"  # agent_failure, low_battery, collision_risk
    failure_time: float = 10.0
    failed_agent_idx: int = 0


class EmergencyExperiment(ExperimentBase):
    """
    Emergency response experiment.
    
    Measures:
    - Response time to emergency
    - Task reallocation efficiency
    - Fleet recovery time
    - Safety maintenance
    """
    
    def __init__(self, config: EmergencyConfig = None):
        super().__init__(config or EmergencyConfig())
        self.config: EmergencyConfig = self.config
        self._emergency_triggered = False
        self._emergency_time = 0.0
        self._recovery_time = 0.0
        self._tasks_reallocated = 0
    
    def setup(self, run_id: int) -> SimulationEngine:
        """Set up simulation for this run."""
        sim_config = SimulationConfig(
            time_step=0.02,
            real_time_factor=float('inf'),
            max_duration=self.config.duration,
            num_uavs=self.config.num_uavs,
            num_ugvs=self.config.num_ugvs,
            enable_coordination=True,
            enable_collision_avoidance=True,
        )
        
        engine = SimulationEngine(sim_config)
        engine.setup_default_scenario()
        
        # Assign tasks to all agents
        for i, agent_id in enumerate(engine._agents.keys()):
            target = Vector3(
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15),
                1.0 if "uav" in agent_id else 0.0
            )
            engine.set_agent_target(agent_id, target)
        
        # Reset tracking
        self._emergency_triggered = False
        self._emergency_time = 0.0
        self._recovery_time = 0.0
        self._tasks_reallocated = 0
        
        # Set up emergency scenario
        def on_step(sim_time: float, states: Dict):
            if not self._emergency_triggered and sim_time >= self.config.failure_time:
                self._trigger_emergency(engine, sim_time)
            
            # Check for recovery
            if self._emergency_triggered and self._recovery_time == 0:
                self._check_recovery(engine, states, sim_time)
        
        engine.register_step_callback(on_step)
        
        return engine
    
    def _trigger_emergency(self, engine: SimulationEngine, sim_time: float) -> None:
        """Trigger emergency scenario."""
        self._emergency_triggered = True
        self._emergency_time = sim_time
        
        if self.config.scenario == "agent_failure":
            # Fail an agent
            agent_ids = list(engine._agents.keys())
            if self.config.failed_agent_idx < len(agent_ids):
                failed_id = agent_ids[self.config.failed_agent_idx]
                agent = engine.get_agent(failed_id)
                if agent:
                    agent._mode = OperationalMode.EMERGENCY
                    agent._battery_level = 0.0
                    logger.info(f"Emergency: Agent {failed_id} failed at t={sim_time:.2f}s")
                    
                    # Reallocate task
                    self._reallocate_task(engine, failed_id)
        
        elif self.config.scenario == "low_battery":
            # Force low battery on multiple agents
            for agent_id in list(engine._agents.keys())[:2]:
                agent = engine.get_agent(agent_id)
                if agent:
                    agent._battery_level = 0.05
                    logger.info(f"Emergency: Agent {agent_id} low battery at t={sim_time:.2f}s")
        
        elif self.config.scenario == "collision_risk":
            # Move agents toward each other
            agents = list(engine._agents.values())
            if len(agents) >= 2:
                center = Vector3(0, 0, 1.0)
                for agent in agents[:3]:
                    agent.set_target_position(center)
                logger.info(f"Emergency: Collision risk created at t={sim_time:.2f}s")
    
    def _reallocate_task(self, engine: SimulationEngine, failed_id: str) -> None:
        """Reallocate task from failed agent."""
        # Find nearest available agent
        failed_agent = engine.get_agent(failed_id)
        if failed_agent is None or failed_agent._target_position is None:
            return
        
        target = failed_agent._target_position
        nearest = None
        min_dist = float('inf')
        
        for agent_id, agent in engine._agents.items():
            if agent_id == failed_id:
                continue
            if agent.mode == OperationalMode.EMERGENCY:
                continue
            
            dist = (agent.position - target).norm()
            if dist < min_dist:
                min_dist = dist
                nearest = agent_id
        
        if nearest:
            engine.set_agent_target(nearest, target)
            self._tasks_reallocated += 1
            logger.info(f"Task reallocated to {nearest}")
    
    def _check_recovery(self, engine: SimulationEngine, states: Dict, sim_time: float) -> None:
        """Check if fleet has recovered from emergency."""
        # Check if all non-failed agents are operational
        operational = 0
        total = 0
        
        for agent_id, state in states.items():
            if state.mode == OperationalMode.EMERGENCY:
                continue
            total += 1
            if state.mode in [OperationalMode.IDLE, OperationalMode.NAVIGATING]:
                operational += 1
        
        # Consider recovered if most agents are operational
        if total > 0 and operational / total >= 0.8:
            self._recovery_time = sim_time - self._emergency_time
            logger.info(f"Fleet recovered in {self._recovery_time:.2f}s")
    
    def compute_metrics(self, engine: SimulationEngine) -> Dict[str, float]:
        """Compute emergency response metrics."""
        # Count final status
        emergency_count = sum(
            1 for a in engine._agents.values() if a.mode == OperationalMode.EMERGENCY
        )
        
        return {
            "scenario": hash(self.config.scenario) % 1000,  # Encode as number
            "emergency_triggered": 1.0 if self._emergency_triggered else 0.0,
            "emergency_time": self._emergency_time,
            "recovery_time": self._recovery_time if self._recovery_time > 0 else self.config.duration,
            "tasks_reallocated": self._tasks_reallocated,
            "agents_in_emergency": emergency_count,
            "fleet_operational_rate": 1.0 - emergency_count / len(engine._agents),
        }


def run_emergency_experiment():
    """Run the emergency experiment."""
    config = EmergencyConfig(
        num_runs=10,
        duration=30.0,
        scenario="agent_failure",
        failure_time=10.0,
        output_dir="./results/emergency",
    )
    
    experiment = EmergencyExperiment(config)
    summary = experiment.run_all()
    
    print("\nEmergency Response Experiment Results:")
    print(f"Successful runs: {summary.successful_runs}/{summary.num_runs}")
    print(f"Mean recovery time: {summary.metrics_mean.get('recovery_time', 0):.2f}s")
    print(f"Mean tasks reallocated: {summary.metrics_mean.get('tasks_reallocated', 0):.1f}")
    print(f"Mean fleet operational: {summary.metrics_mean.get('fleet_operational_rate', 0)*100:.1f}%")
    
    return summary


if __name__ == "__main__":
    run_emergency_experiment()
