"""
Multi-Objective Pareto-Optimal Task Allocation (MOPOTA).

Implements task allocation optimization for heterogeneous fleets
using NSGA-III multi-objective optimization.

Objectives:
1. Minimize total mission time
2. Minimize energy consumption
3. Maximize task completion rate
4. Balance workload across agents

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, PlatformSpecification, Vector3
from heterofleet.core.state import AgentState
from heterofleet.planning.task import Task, TaskType, TaskStatus, TaskConstraints
from heterofleet.planning.nsga3 import NSGA3Optimizer, Individual, ParetoFront


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization."""
    
    mission_time: float = 0.3
    energy_consumption: float = 0.25
    completion_rate: float = 0.25
    workload_balance: float = 0.2
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.mission_time,
            self.energy_consumption,
            self.completion_rate,
            self.workload_balance
        ])
    
    def normalize(self) -> ObjectiveWeights:
        """Normalize weights to sum to 1."""
        total = (self.mission_time + self.energy_consumption + 
                self.completion_rate + self.workload_balance)
        if total > 0:
            return ObjectiveWeights(
                mission_time=self.mission_time / total,
                energy_consumption=self.energy_consumption / total,
                completion_rate=self.completion_rate / total,
                workload_balance=self.workload_balance / total
            )
        return self


@dataclass
class AgentCapabilities:
    """Capabilities of an agent for task allocation."""
    
    agent_id: str
    platform_type: PlatformType
    position: Vector3
    max_speed: float
    payload_capacity: float
    sensor_range: float
    energy_level: float
    energy_capacity: float
    available: bool = True
    current_task: Optional[str] = None
    
    # Capabilities list
    capabilities: List[str] = field(default_factory=list)
    
    def can_perform_task(self, task: Task) -> bool:
        """Check if agent can perform a task."""
        constraints = task.constraints
        
        # Check platform type
        if not constraints.is_platform_allowed(self.platform_type):
            return False
        
        # Check payload
        if self.payload_capacity < constraints.min_payload_capacity:
            return False
        
        # Check sensor range
        if self.sensor_range < constraints.min_sensor_range:
            return False
        
        # Check capabilities
        for req_cap in constraints.required_capabilities:
            if req_cap not in self.capabilities:
                return False
        
        # Check energy
        if self.energy_level < constraints.estimated_energy / self.energy_capacity:
            return False
        
        return True
    
    def estimate_travel_time(self, destination: Vector3) -> float:
        """Estimate travel time to destination."""
        distance = (destination - self.position).norm()
        return distance / self.max_speed if self.max_speed > 0 else float('inf')
    
    def estimate_energy_cost(self, task: Task, cruise_power: float) -> float:
        """Estimate energy cost for a task."""
        distance = (task.location.position - self.position).norm()
        travel_time = distance / self.max_speed if self.max_speed > 0 else 0
        task_duration = task.estimate_duration(self.max_speed)
        
        total_time = travel_time + task_duration
        energy_wh = cruise_power * total_time / 3600
        
        return energy_wh


@dataclass
class AllocationResult:
    """Result of task allocation optimization."""
    
    # Task assignments: task_id -> agent_id
    assignments: Dict[str, str] = field(default_factory=dict)
    
    # Objective values
    mission_time: float = 0.0
    energy_consumption: float = 0.0
    completion_rate: float = 0.0
    workload_balance: float = 0.0
    
    # Statistics
    num_tasks_assigned: int = 0
    num_tasks_unassigned: int = 0
    
    # Per-agent info
    agent_workloads: Dict[str, float] = field(default_factory=dict)
    agent_task_counts: Dict[str, int] = field(default_factory=dict)
    
    # Pareto front info
    pareto_front_size: int = 0
    is_pareto_optimal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": self.assignments,
            "objectives": {
                "mission_time": self.mission_time,
                "energy_consumption": self.energy_consumption,
                "completion_rate": self.completion_rate,
                "workload_balance": self.workload_balance,
            },
            "statistics": {
                "num_tasks_assigned": self.num_tasks_assigned,
                "num_tasks_unassigned": self.num_tasks_unassigned,
                "pareto_front_size": self.pareto_front_size,
            },
            "agent_workloads": self.agent_workloads,
        }


class MOPOTAAllocator:
    """
    Multi-Objective Pareto-Optimal Task Allocator.
    
    Uses NSGA-III to find Pareto-optimal task allocations
    optimizing multiple objectives simultaneously.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2
    ):
        """Initialize MOPOTA allocator."""
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Data
        self._tasks: List[Task] = []
        self._agents: List[AgentCapabilities] = []
        self._feasibility_matrix: Optional[np.ndarray] = None
        
        # Results
        self._pareto_front: Optional[ParetoFront] = None
        self._best_allocation: Optional[AllocationResult] = None
    
    def set_tasks(self, tasks: List[Task]) -> None:
        """Set tasks to allocate."""
        self._tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        self._feasibility_matrix = None
    
    def set_agents(self, agents: List[AgentCapabilities]) -> None:
        """Set available agents."""
        self._agents = [a for a in agents if a.available]
        self._feasibility_matrix = None
    
    def _compute_feasibility_matrix(self) -> np.ndarray:
        """Compute task-agent feasibility matrix."""
        num_tasks = len(self._tasks)
        num_agents = len(self._agents)
        
        # Add 1 for "unassigned" option
        matrix = np.zeros((num_tasks, num_agents + 1), dtype=bool)
        
        for i, task in enumerate(self._tasks):
            # Unassigned is always feasible
            matrix[i, -1] = True
            
            for j, agent in enumerate(self._agents):
                matrix[i, j] = agent.can_perform_task(task)
        
        return matrix
    
    def _chromosome_to_assignment(self, chromosome: np.ndarray) -> Dict[str, str]:
        """Convert chromosome to task-agent assignment."""
        assignments = {}
        
        for i, gene in enumerate(chromosome):
            if gene < len(self._agents):  # Not unassigned
                task_id = self._tasks[i].task_id
                agent_id = self._agents[gene].agent_id
                assignments[task_id] = agent_id
        
        return assignments
    
    def _compute_mission_time(self, chromosome: np.ndarray) -> float:
        """Compute total mission time for allocation."""
        agent_times: Dict[int, float] = {}
        
        for task_idx, agent_idx in enumerate(chromosome):
            if agent_idx >= len(self._agents):
                continue
            
            task = self._tasks[task_idx]
            agent = self._agents[agent_idx]
            
            # Travel time
            travel_time = agent.estimate_travel_time(task.location.position)
            
            # Task duration
            task_duration = task.estimate_duration(agent.max_speed)
            
            # Add to agent's total time
            current = agent_times.get(agent_idx, 0.0)
            agent_times[agent_idx] = current + travel_time + task_duration
        
        # Mission time is max agent time (parallel execution)
        return max(agent_times.values()) if agent_times else 0.0
    
    def _compute_energy_consumption(self, chromosome: np.ndarray) -> float:
        """Compute total energy consumption."""
        total_energy = 0.0
        
        for task_idx, agent_idx in enumerate(chromosome):
            if agent_idx >= len(self._agents):
                continue
            
            task = self._tasks[task_idx]
            agent = self._agents[agent_idx]
            
            # Estimate energy (assume average cruise power)
            cruise_power = 5.0  # Watts (will be refined with platform specs)
            energy = agent.estimate_energy_cost(task, cruise_power)
            total_energy += energy
        
        return total_energy
    
    def _compute_completion_rate(self, chromosome: np.ndarray) -> float:
        """Compute task completion rate (to maximize, return negative)."""
        assigned = sum(1 for g in chromosome if g < len(self._agents))
        rate = assigned / len(self._tasks) if self._tasks else 0.0
        return -rate  # Negative because we minimize
    
    def _compute_workload_balance(self, chromosome: np.ndarray) -> float:
        """Compute workload imbalance (to minimize)."""
        if not self._agents:
            return 0.0
        
        agent_loads = np.zeros(len(self._agents))
        
        for task_idx, agent_idx in enumerate(chromosome):
            if agent_idx < len(self._agents):
                task = self._tasks[task_idx]
                agent = self._agents[agent_idx]
                
                # Load = estimated time
                load = (agent.estimate_travel_time(task.location.position) +
                       task.estimate_duration(agent.max_speed))
                agent_loads[agent_idx] += load
        
        # Standard deviation of loads
        if np.sum(agent_loads) > 0:
            return float(np.std(agent_loads))
        return 0.0
    
    def _evaluate_chromosome(self, chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a chromosome and return objectives and constraints."""
        # Compute objectives
        objectives = np.array([
            self._compute_mission_time(chromosome),
            self._compute_energy_consumption(chromosome),
            self._compute_completion_rate(chromosome),
            self._compute_workload_balance(chromosome),
        ])
        
        # Compute constraints
        constraints = []
        
        # Feasibility constraint
        for task_idx, agent_idx in enumerate(chromosome):
            if not self._feasibility_matrix[task_idx, agent_idx]:
                constraints.append(1.0)  # Violation
        
        return objectives, np.array(constraints)
    
    def _repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """Repair infeasible chromosome."""
        repaired = chromosome.copy()
        
        for task_idx in range(len(self._tasks)):
            if not self._feasibility_matrix[task_idx, repaired[task_idx]]:
                # Find feasible agents
                feasible = np.where(self._feasibility_matrix[task_idx])[0]
                if len(feasible) > 0:
                    repaired[task_idx] = np.random.choice(feasible)
        
        return repaired
    
    def allocate(
        self,
        weights: ObjectiveWeights = None,
        callback: Callable[[int, ParetoFront], bool] = None
    ) -> AllocationResult:
        """
        Perform task allocation optimization.
        
        Args:
            weights: Objective weights for compromise selection
            callback: Optional callback(generation, front) -> continue
            
        Returns:
            Best allocation result
        """
        if not self._tasks:
            logger.warning("No tasks to allocate")
            return AllocationResult()
        
        if not self._agents:
            logger.warning("No agents available")
            return AllocationResult(num_tasks_unassigned=len(self._tasks))
        
        # Compute feasibility matrix
        self._feasibility_matrix = self._compute_feasibility_matrix()
        
        # Check if any allocation is possible
        if not np.any(self._feasibility_matrix[:, :-1]):
            logger.warning("No feasible task-agent assignments")
            return AllocationResult(num_tasks_unassigned=len(self._tasks))
        
        # Setup NSGA-III
        num_tasks = len(self._tasks)
        num_agents = len(self._agents)
        
        # Variable bounds: 0 to num_agents (inclusive, last = unassigned)
        variable_bounds = [(0, num_agents)] * num_tasks
        
        optimizer = NSGA3Optimizer(
            num_objectives=4,
            num_variables=num_tasks,
            variable_bounds=variable_bounds,
            population_size=self.population_size,
            max_generations=self.max_generations,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
        )
        
        optimizer.set_objective_function(self._evaluate_chromosome)
        
        # Run optimization
        def internal_callback(gen, pop, front):
            if callback:
                return callback(gen, front)
            return True
        
        self._pareto_front = optimizer.optimize(internal_callback)
        
        # Select best compromise solution
        weights = weights or ObjectiveWeights()
        weights = weights.normalize()
        
        best_individual = optimizer.get_best_compromise(weights.to_array())
        
        # Convert to result
        result = self._individual_to_result(best_individual)
        result.pareto_front_size = self._pareto_front.size
        result.is_pareto_optimal = True
        
        self._best_allocation = result
        
        logger.info(f"Allocation complete: {result.num_tasks_assigned}/{num_tasks} tasks assigned, "
                   f"Pareto front size: {result.pareto_front_size}")
        
        return result
    
    def _individual_to_result(self, individual: Individual) -> AllocationResult:
        """Convert individual to allocation result."""
        assignments = self._chromosome_to_assignment(individual.chromosome)
        
        # Compute per-agent stats
        agent_workloads = {}
        agent_task_counts = {}
        
        for agent in self._agents:
            agent_workloads[agent.agent_id] = 0.0
            agent_task_counts[agent.agent_id] = 0
        
        for task_idx, agent_idx in enumerate(individual.chromosome):
            if agent_idx < len(self._agents):
                task = self._tasks[task_idx]
                agent = self._agents[agent_idx]
                
                load = (agent.estimate_travel_time(task.location.position) +
                       task.estimate_duration(agent.max_speed))
                
                agent_workloads[agent.agent_id] += load
                agent_task_counts[agent.agent_id] += 1
        
        num_assigned = sum(1 for g in individual.chromosome if g < len(self._agents))
        
        return AllocationResult(
            assignments=assignments,
            mission_time=individual.objectives[0],
            energy_consumption=individual.objectives[1],
            completion_rate=-individual.objectives[2],  # Was negated
            workload_balance=individual.objectives[3],
            num_tasks_assigned=num_assigned,
            num_tasks_unassigned=len(self._tasks) - num_assigned,
            agent_workloads=agent_workloads,
            agent_task_counts=agent_task_counts,
        )
    
    def get_pareto_front(self) -> Optional[ParetoFront]:
        """Get the Pareto front from last optimization."""
        return self._pareto_front
    
    def get_all_pareto_allocations(self) -> List[AllocationResult]:
        """Get all Pareto-optimal allocations."""
        if self._pareto_front is None:
            return []
        
        return [self._individual_to_result(ind) 
                for ind in self._pareto_front.individuals]
    
    def allocate_greedy(self) -> AllocationResult:
        """
        Perform greedy task allocation (fast baseline).
        
        Assigns tasks to nearest capable agent.
        """
        if not self._tasks or not self._agents:
            return AllocationResult(num_tasks_unassigned=len(self._tasks))
        
        if self._feasibility_matrix is None:
            self._feasibility_matrix = self._compute_feasibility_matrix()
        
        assignments = {}
        agent_positions = {a.agent_id: a.position for a in self._agents}
        agent_loads = {a.agent_id: 0.0 for a in self._agents}
        
        # Sort tasks by priority
        sorted_tasks = sorted(self._tasks, 
                             key=lambda t: -t.priority_value)
        
        for task in sorted_tasks:
            task_idx = self._tasks.index(task)
            best_agent = None
            best_score = float('inf')
            
            for agent_idx, agent in enumerate(self._agents):
                if not self._feasibility_matrix[task_idx, agent_idx]:
                    continue
                
                # Score = distance + current load
                distance = (task.location.position - agent_positions[agent.agent_id]).norm()
                load = agent_loads[agent.agent_id]
                score = distance + load * 0.5
                
                if score < best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                assignments[task.task_id] = best_agent.agent_id
                
                # Update agent position and load
                travel_time = best_agent.estimate_travel_time(task.location.position)
                task_duration = task.estimate_duration(best_agent.max_speed)
                agent_loads[best_agent.agent_id] += travel_time + task_duration
                agent_positions[best_agent.agent_id] = task.location.position
        
        # Build result
        agent_workloads = agent_loads.copy()
        agent_task_counts = {a.agent_id: 0 for a in self._agents}
        for task_id, agent_id in assignments.items():
            agent_task_counts[agent_id] += 1
        
        mission_time = max(agent_loads.values()) if agent_loads else 0.0
        
        return AllocationResult(
            assignments=assignments,
            mission_time=mission_time,
            num_tasks_assigned=len(assignments),
            num_tasks_unassigned=len(self._tasks) - len(assignments),
            agent_workloads=agent_workloads,
            agent_task_counts=agent_task_counts,
        )
    
    def reallocate_failed_task(
        self,
        task: Task,
        failed_agent_id: str
    ) -> Optional[str]:
        """
        Reallocate a failed task to another agent.
        
        Args:
            task: Task that failed
            failed_agent_id: Agent that failed
            
        Returns:
            New agent ID or None if no agent available
        """
        # Find alternative agents
        alternatives = []
        
        for agent in self._agents:
            if agent.agent_id == failed_agent_id:
                continue
            if not agent.available:
                continue
            if not agent.can_perform_task(task):
                continue
            
            # Score by distance and current load
            distance = (task.location.position - agent.position).norm()
            alternatives.append((agent.agent_id, distance))
        
        if not alternatives:
            return None
        
        # Select best alternative
        alternatives.sort(key=lambda x: x[1])
        return alternatives[0][0]


class AuctionBasedAllocator:
    """
    Auction-based distributed task allocation.
    
    Alternative to centralized MOPOTA for real-time scenarios.
    """
    
    def __init__(self):
        """Initialize auction allocator."""
        self._tasks: Dict[str, Task] = {}
        self._bids: Dict[str, Dict[str, float]] = {}  # task_id -> {agent_id -> bid}
    
    def announce_task(self, task: Task) -> None:
        """Announce a new task for bidding."""
        self._tasks[task.task_id] = task
        self._bids[task.task_id] = {}
    
    def submit_bid(self, task_id: str, agent_id: str, bid_value: float) -> bool:
        """Submit a bid for a task."""
        if task_id not in self._tasks:
            return False
        
        self._bids[task_id][agent_id] = bid_value
        return True
    
    def resolve_auction(self, task_id: str) -> Optional[str]:
        """Resolve auction and return winning agent."""
        if task_id not in self._bids:
            return None
        
        bids = self._bids[task_id]
        if not bids:
            return None
        
        # Lower bid wins (bid represents cost)
        winner = min(bids.items(), key=lambda x: x[1])[0]
        
        # Clear auction
        del self._tasks[task_id]
        del self._bids[task_id]
        
        return winner
    
    def compute_bid(
        self,
        agent: AgentCapabilities,
        task: Task,
        current_load: float = 0.0
    ) -> float:
        """Compute bid value for a task."""
        if not agent.can_perform_task(task):
            return float('inf')
        
        # Bid = travel cost + opportunity cost
        travel_time = agent.estimate_travel_time(task.location.position)
        task_duration = task.estimate_duration(agent.max_speed)
        
        # Opportunity cost based on current load
        opportunity_cost = current_load * 0.1
        
        # Energy cost
        energy_cost = agent.estimate_energy_cost(task, 5.0) * 0.5
        
        return travel_time + task_duration + opportunity_cost + energy_cost
