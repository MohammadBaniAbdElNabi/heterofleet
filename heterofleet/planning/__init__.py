"""
Planning module for HeteroFleet.

Implements Multi-Objective Pareto-Optimal Task Allocation (MOPOTA):
- NSGA-III based multi-objective optimization
- Task representation and management
- Platform capability matching
- Temporal constraint scheduling

Based on HeteroFleet Architecture v1.0
"""

from heterofleet.planning.mopota import (
    MOPOTAAllocator,
    AllocationResult,
    ObjectiveWeights,
)
from heterofleet.planning.nsga3 import (
    NSGA3Optimizer,
    Individual,
    ParetoFront,
)
from heterofleet.planning.task import (
    Task,
    TaskType,
    TaskStatus,
    TaskConstraints,
    TaskManager,
)
from heterofleet.planning.scheduler import (
    TemporalScheduler,
    ScheduleEntry,
    Schedule,
)

__all__ = [
    # MOPOTA
    "MOPOTAAllocator",
    "AllocationResult",
    "ObjectiveWeights",
    # NSGA-III
    "NSGA3Optimizer",
    "Individual",
    "ParetoFront",
    # Task
    "Task",
    "TaskType",
    "TaskStatus",
    "TaskConstraints",
    "TaskManager",
    # Scheduler
    "TemporalScheduler",
    "ScheduleEntry",
    "Schedule",
]
