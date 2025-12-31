"""
Scalability Experiment for HeteroFleet.

Tests system performance as fleet size increases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from experiments.base import ExperimentBase, ExperimentConfig


@dataclass
class ScalabilityConfig(ExperimentConfig):
    """Configuration for scalability experiment."""
    
    name: str = "scalability"
    description: str = "Test system performance with varying fleet sizes"
    
    # Fleet sizes to test
    fleet_sizes: List[int] = None
    
    # Composition
    uav_ratio: float = 0.7  # Ratio of UAVs
    
    # Environment
    arena_size: float = 50.0
    
    def __post_init__(self):
        if self.fleet_sizes is None:
            self.fleet_sizes = [5, 10, 20, 50, 100]


class ScalabilityExperiment(ExperimentBase):
    """
    Scalability analysis experiment.
    
    Measures:
    - Coordination overhead vs fleet size
    - Update rate as agents increase
    - Memory usage scaling
    - Communication bandwidth
    """
    
    def __init__(self, config: ScalabilityConfig = None):
        super().__init__(config or ScalabilityConfig())
        self.config: ScalabilityConfig = self.config
        self._current_fleet_size = 0
    
    def setup(self, run_id: int) -> SimulationEngine:
        """Set up simulation for this run."""
        # Determine fleet size for this run
        size_idx = run_id % len(self.config.fleet_sizes)
        self._current_fleet_size = self.config.fleet_sizes[size_idx]
        
        num_uavs = int(self._current_fleet_size * self.config.uav_ratio)
        num_ugvs = self._current_fleet_size - num_uavs
        
        # Create simulation
        sim_config = SimulationConfig(
            time_step=0.02,
            real_time_factor=float('inf'),  # As fast as possible
            max_duration=self.config.duration,
            num_uavs=num_uavs,
            num_ugvs=num_ugvs,
            enable_coordination=True,
        )
        
        engine = SimulationEngine(sim_config)
        
        # Add agents in grid pattern
        grid_size = int(np.ceil(np.sqrt(self._current_fleet_size)))
        spacing = self.config.arena_size / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= self._current_fleet_size:
                    break
                
                pos = Vector3(
                    -self.config.arena_size/2 + (i+1) * spacing,
                    -self.config.arena_size/2 + (j+1) * spacing,
                    1.0 if idx < num_uavs else 0.0
                )
                
                if idx < num_uavs:
                    platform_type = PlatformType.SMALL_UAV
                else:
                    platform_type = PlatformType.SMALL_UGV
                
                spec = PlatformSpecification.from_platform_type(platform_type)
                engine.add_agent(f"agent_{idx}", spec, pos)
                
                idx += 1
        
        # Set random targets
        for agent_id in engine._agents.keys():
            target = Vector3(
                np.random.uniform(-self.config.arena_size/2, self.config.arena_size/2),
                np.random.uniform(-self.config.arena_size/2, self.config.arena_size/2),
                1.0 if "uav" in agent_id or int(agent_id.split("_")[1]) < num_uavs else 0.0
            )
            engine.set_agent_target(agent_id, target)
        
        return engine
    
    def compute_metrics(self, engine: SimulationEngine) -> Dict[str, float]:
        """Compute scalability metrics."""
        state = engine.state
        
        return {
            "fleet_size": self._current_fleet_size,
            "sim_time": state.sim_time,
            "wall_time": state.wall_time,
            "step_count": state.step_count,
            "avg_step_time_ms": state.avg_step_time_ms,
            "real_time_ratio": state.real_time_ratio,
            "steps_per_second": state.step_count / max(0.001, state.wall_time),
        }


def run_scalability_experiment():
    """Run the scalability experiment."""
    config = ScalabilityConfig(
        num_runs=15,  # 3 runs per fleet size
        duration=30.0,
        fleet_sizes=[5, 10, 20, 50, 100],
        output_dir="./results/scalability",
    )
    
    experiment = ScalabilityExperiment(config)
    summary = experiment.run_all()
    
    print("\nScalability Experiment Results:")
    print(f"Successful runs: {summary.successful_runs}/{summary.num_runs}")
    print(f"Mean step time: {summary.metrics_mean.get('avg_step_time_ms', 0):.2f} ms")
    print(f"Mean real-time ratio: {summary.metrics_mean.get('real_time_ratio', 0):.2f}")
    
    return summary


if __name__ == "__main__":
    run_scalability_experiment()
