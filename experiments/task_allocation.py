"""
Task Allocation Experiment for HeteroFleet.

Tests multi-objective task allocation performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.planning.task import Task, TaskType, TaskStatus, TaskPriority
from heterofleet.planning.mopota import MOPOTAAllocator, AgentCapabilities, ObjectiveWeights
from experiments.base import ExperimentBase, ExperimentConfig


@dataclass
class TaskAllocationConfig(ExperimentConfig):
    """Configuration for task allocation experiment."""
    
    name: str = "task_allocation"
    description: str = "Test multi-objective task allocation"
    
    # Fleet
    num_uavs: int = 5
    num_ugvs: int = 2
    
    # Tasks
    num_tasks: int = 15
    task_area_size: float = 30.0
    
    # MOPOTA settings
    nsga_generations: int = 50
    population_size: int = 50


class TaskAllocationExperiment(ExperimentBase):
    """
    Task allocation experiment.
    
    Measures:
    - Task completion rate
    - Mission time optimization
    - Energy consumption
    - Workload balance
    """
    
    def __init__(self, config: TaskAllocationConfig = None):
        super().__init__(config or TaskAllocationConfig())
        self.config: TaskAllocationConfig = self.config
        self._tasks: List[Task] = []
        self._allocation_result = None
    
    def setup(self, run_id: int) -> SimulationEngine:
        """Set up simulation for this run."""
        sim_config = SimulationConfig(
            time_step=0.05,
            real_time_factor=float('inf'),
            max_duration=self.config.duration,
            num_uavs=self.config.num_uavs,
            num_ugvs=self.config.num_ugvs,
            enable_coordination=True,
        )
        
        engine = SimulationEngine(sim_config)
        engine.setup_default_scenario()
        
        # Generate random tasks
        self._tasks = []
        for i in range(self.config.num_tasks):
            # Random position
            pos = Vector3(
                np.random.uniform(-self.config.task_area_size/2, self.config.task_area_size/2),
                np.random.uniform(-self.config.task_area_size/2, self.config.task_area_size/2),
                np.random.uniform(0, 3) if np.random.random() > 0.5 else 0
            )
            
            # Random task type
            if pos.z > 0:
                task_type = np.random.choice([TaskType.SURVEILLANCE, TaskType.INSPECTION])
            else:
                task_type = np.random.choice([TaskType.DELIVERY, TaskType.SURVEILLANCE])
            
            task = Task(
                task_id=f"task_{i}",
                task_type=task_type,
                location=pos,
                duration=np.random.uniform(5, 20),
                priority=TaskPriority.MEDIUM,
                required_capabilities=[task_type.name.lower()],
            )
            self._tasks.append(task)
        
        # Create agent capabilities
        capabilities = {}
        
        for i in range(self.config.num_uavs):
            agent_id = f"uav_{i}"
            agent = engine.get_agent(agent_id)
            if agent:
                capabilities[agent_id] = AgentCapabilities(
                    agent_id=agent_id,
                    platform_type=PlatformType.SMALL_UAV,
                    position=agent.position,
                    max_speed=agent.platform_spec.max_velocity,
                    payload_capacity=0.1,
                    sensor_range=10.0,
                    energy_level=1.0,
                    available_sensors=["camera", "surveillance", "inspection"],
                )
        
        for i in range(self.config.num_ugvs):
            agent_id = f"ugv_{i}"
            agent = engine.get_agent(agent_id)
            if agent:
                capabilities[agent_id] = AgentCapabilities(
                    agent_id=agent_id,
                    platform_type=PlatformType.SMALL_UGV,
                    position=agent.position,
                    max_speed=agent.platform_spec.max_velocity,
                    payload_capacity=1.0,
                    sensor_range=5.0,
                    energy_level=1.0,
                    available_sensors=["camera", "delivery", "surveillance"],
                )
        
        # Run MOPOTA allocation
        allocator = MOPOTAAllocator(
            num_generations=self.config.nsga_generations,
            population_size=self.config.population_size,
        )
        
        self._allocation_result = allocator.allocate(
            self._tasks,
            capabilities,
            ObjectiveWeights()
        )
        
        # Assign tasks to agents
        for task_id, agent_id in self._allocation_result.assignments.items():
            task = next((t for t in self._tasks if t.task_id == task_id), None)
            if task and agent_id:
                engine.set_agent_target(agent_id, task.location)
        
        return engine
    
    def compute_metrics(self, engine: SimulationEngine) -> Dict[str, float]:
        """Compute task allocation metrics."""
        if self._allocation_result is None:
            return {}
        
        # Count assigned tasks
        assigned = sum(1 for a in self._allocation_result.assignments.values() if a is not None)
        
        return {
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_uavs + self.config.num_ugvs,
            "tasks_assigned": assigned,
            "completion_rate": self._allocation_result.objectives["completion_rate"],
            "mission_time": self._allocation_result.objectives["mission_time"],
            "energy_consumption": self._allocation_result.objectives["energy_consumption"],
            "workload_balance": self._allocation_result.objectives["workload_balance"],
            "pareto_solutions": len(self._allocation_result.pareto_solutions) if self._allocation_result.pareto_solutions else 0,
        }


def run_task_allocation_experiment():
    """Run the task allocation experiment."""
    config = TaskAllocationConfig(
        num_runs=10,
        duration=60.0,
        num_tasks=15,
        output_dir="./results/task_allocation",
    )
    
    experiment = TaskAllocationExperiment(config)
    summary = experiment.run_all()
    
    print("\nTask Allocation Experiment Results:")
    print(f"Successful runs: {summary.successful_runs}/{summary.num_runs}")
    print(f"Mean completion rate: {summary.metrics_mean.get('completion_rate', 0)*100:.1f}%")
    print(f"Mean mission time: {summary.metrics_mean.get('mission_time', 0):.2f}s")
    print(f"Mean energy: {summary.metrics_mean.get('energy_consumption', 0):.2f}")
    print(f"Mean workload balance: {summary.metrics_mean.get('workload_balance', 0):.3f}")
    
    return summary


if __name__ == "__main__":
    run_task_allocation_experiment()
