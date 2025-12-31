"""
Formation Control Experiment for HeteroFleet.

Tests formation acquisition and maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.digital_twin.fleet_twin import FormationTracker
from experiments.base import ExperimentBase, ExperimentConfig


@dataclass
class FormationConfig(ExperimentConfig):
    """Configuration for formation experiment."""
    
    name: str = "formation"
    description: str = "Test formation control performance"
    
    # Formation
    formation_type: str = "v"  # line, v, circle
    num_agents: int = 5
    formation_spacing: float = 2.0
    
    # Performance
    formation_tolerance: float = 0.5


class FormationExperiment(ExperimentBase):
    """
    Formation control experiment.
    
    Measures:
    - Time to achieve formation
    - Formation error over time
    - Robustness to disturbances
    """
    
    def __init__(self, config: FormationConfig = None):
        super().__init__(config or FormationConfig())
        self.config: FormationConfig = self.config
        self._formation_tracker: FormationTracker = None
        self._target_positions: Dict[str, Vector3] = {}
        self._formation_achieved_time: float = 0.0
    
    def setup(self, run_id: int) -> SimulationEngine:
        """Set up simulation for this run."""
        sim_config = SimulationConfig(
            time_step=0.02,
            real_time_factor=float('inf'),
            max_duration=self.config.duration,
            num_uavs=self.config.num_agents,
            num_ugvs=0,
            enable_coordination=True,
        )
        
        engine = SimulationEngine(sim_config)
        
        # Create formation tracker
        self._formation_tracker = FormationTracker(
            formation_tolerance=self.config.formation_tolerance
        )
        
        # Add agents at random positions
        agent_ids = []
        for i in range(self.config.num_agents):
            pos = Vector3(
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                1.0
            )
            
            spec = PlatformSpecification.from_platform_type(PlatformType.SMALL_UAV)
            engine.add_agent(f"uav_{i}", spec, pos)
            agent_ids.append(f"uav_{i}")
        
        # Generate target formation
        leader_pos = Vector3(0, 0, 1.5)
        direction = Vector3(1, 0, 0)
        
        if self.config.formation_type == "line":
            self._target_positions = self._formation_tracker.generate_line_formation(
                agent_ids, leader_pos, direction, self.config.formation_spacing
            )
        elif self.config.formation_type == "v":
            self._target_positions = self._formation_tracker.generate_v_formation(
                agent_ids, leader_pos, direction, self.config.formation_spacing
            )
        elif self.config.formation_type == "circle":
            self._target_positions = self._formation_tracker.generate_circle_formation(
                agent_ids, leader_pos, self.config.formation_spacing
            )
        
        # Create formation in tracker
        self._formation_tracker.create_formation(
            "main_formation",
            self.config.formation_type,
            agent_ids,
            self._target_positions,
            agent_ids[0]
        )
        
        # Set targets for agents
        for agent_id, target in self._target_positions.items():
            engine.set_agent_target(agent_id, target)
        
        # Track formation achievement
        self._formation_achieved_time = 0.0
        
        def on_step(sim_time: float, states: Dict):
            positions = {aid: states[aid].position for aid in states}
            formation = self._formation_tracker.update_positions("main_formation", positions)
            
            if formation and formation.is_formed and self._formation_achieved_time == 0:
                self._formation_achieved_time = sim_time
        
        engine.register_step_callback(on_step)
        
        return engine
    
    def compute_metrics(self, engine: SimulationEngine) -> Dict[str, float]:
        """Compute formation metrics."""
        formation = self._formation_tracker.get_formation("main_formation")
        
        return {
            "formation_type": self.config.formation_type,
            "num_agents": self.config.num_agents,
            "formation_achieved": 1.0 if formation and formation.is_formed else 0.0,
            "time_to_formation": self._formation_achieved_time if self._formation_achieved_time > 0 else self.config.duration,
            "final_error": formation.formation_error if formation else float('inf'),
            "max_error": formation.max_error if formation else float('inf'),
        }


def run_formation_experiment():
    """Run the formation experiment."""
    config = FormationConfig(
        num_runs=10,
        duration=30.0,
        num_agents=5,
        formation_type="v",
        output_dir="./results/formation",
    )
    
    experiment = FormationExperiment(config)
    summary = experiment.run_all()
    
    print("\nFormation Experiment Results:")
    print(f"Successful runs: {summary.successful_runs}/{summary.num_runs}")
    print(f"Formation success rate: {summary.metrics_mean.get('formation_achieved', 0)*100:.1f}%")
    print(f"Mean time to formation: {summary.metrics_mean.get('time_to_formation', 0):.2f}s")
    print(f"Mean final error: {summary.metrics_mean.get('final_error', 0):.3f}m")
    
    return summary


if __name__ == "__main__":
    run_formation_experiment()
