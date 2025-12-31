"""
Mission-level Digital Twin for operational objectives.

Maintains mission state, progress tracking, and
performance monitoring.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np
from loguru import logger

from heterofleet.core.platform import Vector3
from heterofleet.planning.task import Task, TaskStatus, TaskType


class MissionStatus(Enum):
    """Status of mission execution."""
    PLANNING = auto()
    EXECUTING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class MissionObjective:
    """A mission objective."""
    
    objective_id: str = ""
    name: str = ""
    description: str = ""
    
    # Progress
    target_value: float = 1.0
    current_value: float = 0.0
    
    # Timing
    deadline: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Priority
    priority: float = 1.0
    is_critical: bool = False
    
    @property
    def progress(self) -> float:
        """Get progress percentage (0-1)."""
        if self.target_value <= 0:
            return 1.0
        return min(1.0, self.current_value / self.target_value)
    
    @property
    def is_complete(self) -> bool:
        return self.current_value >= self.target_value


@dataclass
class MissionMetrics:
    """Metrics for mission performance."""
    
    # Progress
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    pending_tasks: int = 0
    
    # Objectives
    total_objectives: int = 0
    completed_objectives: int = 0
    
    # Performance
    completion_rate: float = 0.0
    avg_task_duration: float = 0.0
    total_duration: float = 0.0
    
    # Resources
    agents_assigned: int = 0
    total_energy_consumed_wh: float = 0.0
    total_distance_traveled_m: float = 0.0
    
    # Quality
    avg_task_quality: float = 0.0
    on_time_rate: float = 0.0
    
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "completion_rate": self.completion_rate,
            "total_objectives": self.total_objectives,
            "completed_objectives": self.completed_objectives,
            "avg_task_duration": self.avg_task_duration,
            "total_energy_consumed_wh": self.total_energy_consumed_wh,
            "timestamp": self.timestamp,
        }


@dataclass
class MissionState:
    """Complete mission state."""
    
    mission_id: str = ""
    name: str = ""
    description: str = ""
    
    # Status
    status: MissionStatus = MissionStatus.PLANNING
    
    # Timing
    planned_start: Optional[float] = None
    actual_start: Optional[float] = None
    planned_end: Optional[float] = None
    actual_end: Optional[float] = None
    
    # Tasks
    task_ids: List[str] = field(default_factory=list)
    task_statuses: Dict[str, TaskStatus] = field(default_factory=dict)
    
    # Objectives
    objectives: List[MissionObjective] = field(default_factory=list)
    
    # Assignments
    assigned_agents: Set[str] = field(default_factory=set)
    
    # Metrics
    metrics: MissionMetrics = field(default_factory=MissionMetrics)
    
    # Area of operation
    operation_area: Optional[Tuple[Vector3, Vector3]] = None  # bounding box
    
    @property
    def progress(self) -> float:
        """Get overall mission progress."""
        if not self.task_ids:
            return 0.0
        
        completed = sum(1 for s in self.task_statuses.values() 
                       if s == TaskStatus.COMPLETED)
        return completed / len(self.task_ids)
    
    @property
    def is_active(self) -> bool:
        return self.status in (MissionStatus.EXECUTING, MissionStatus.PAUSED)
    
    @property
    def elapsed_time(self) -> float:
        if self.actual_start is None:
            return 0.0
        end = self.actual_end or time.time()
        return end - self.actual_start


class MissionTwin:
    """
    Mission-level digital twin.
    
    Tracks mission state, objectives, and performance.
    """
    
    def __init__(self, mission_id: str, name: str = ""):
        """
        Initialize mission twin.
        
        Args:
            mission_id: Mission identifier
            name: Mission name
        """
        self.mission_id = mission_id
        
        self._state = MissionState(
            mission_id=mission_id,
            name=name or mission_id,
        )
        
        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._task_start_times: Dict[str, float] = {}
        self._task_end_times: Dict[str, float] = {}
        
        # History
        self._metrics_history: List[MissionMetrics] = []
        self._event_log: List[Dict[str, Any]] = []
        
        # Callbacks
        self._status_callbacks: List[Callable[[MissionStatus], None]] = []
        self._progress_callbacks: List[Callable[[float], None]] = []
    
    @property
    def state(self) -> MissionState:
        """Get current mission state."""
        return self._state
    
    @property
    def status(self) -> MissionStatus:
        """Get mission status."""
        return self._state.status
    
    @property
    def progress(self) -> float:
        """Get mission progress."""
        return self._state.progress
    
    def add_task(self, task: Task) -> None:
        """Add a task to the mission."""
        self._tasks[task.task_id] = task
        self._state.task_ids.append(task.task_id)
        self._state.task_statuses[task.task_id] = task.status
        self._update_metrics()
        self._log_event("task_added", {"task_id": task.task_id})
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from the mission."""
        self._tasks.pop(task_id, None)
        if task_id in self._state.task_ids:
            self._state.task_ids.remove(task_id)
        self._state.task_statuses.pop(task_id, None)
        self._update_metrics()
        self._log_event("task_removed", {"task_id": task_id})
    
    def add_objective(self, objective: MissionObjective) -> None:
        """Add a mission objective."""
        self._state.objectives.append(objective)
        self._update_metrics()
    
    def update_objective(self, objective_id: str, current_value: float) -> None:
        """Update objective progress."""
        for obj in self._state.objectives:
            if obj.objective_id == objective_id:
                obj.current_value = current_value
                if obj.is_complete and obj.completed_at is None:
                    obj.completed_at = time.time()
                    self._log_event("objective_completed", {"objective_id": objective_id})
                break
        self._update_metrics()
    
    def assign_agent(self, agent_id: str) -> None:
        """Assign an agent to the mission."""
        self._state.assigned_agents.add(agent_id)
        self._log_event("agent_assigned", {"agent_id": agent_id})
    
    def unassign_agent(self, agent_id: str) -> None:
        """Unassign an agent from the mission."""
        self._state.assigned_agents.discard(agent_id)
        self._log_event("agent_unassigned", {"agent_id": agent_id})
    
    def start(self) -> bool:
        """Start the mission."""
        if self._state.status != MissionStatus.PLANNING:
            return False
        
        self._state.status = MissionStatus.EXECUTING
        self._state.actual_start = time.time()
        self._log_event("mission_started", {})
        self._trigger_status_callback()
        
        return True
    
    def pause(self) -> bool:
        """Pause the mission."""
        if self._state.status != MissionStatus.EXECUTING:
            return False
        
        self._state.status = MissionStatus.PAUSED
        self._log_event("mission_paused", {})
        self._trigger_status_callback()
        
        return True
    
    def resume(self) -> bool:
        """Resume the mission."""
        if self._state.status != MissionStatus.PAUSED:
            return False
        
        self._state.status = MissionStatus.EXECUTING
        self._log_event("mission_resumed", {})
        self._trigger_status_callback()
        
        return True
    
    def complete(self) -> bool:
        """Mark mission as completed."""
        if self._state.status not in (MissionStatus.EXECUTING, MissionStatus.PAUSED):
            return False
        
        self._state.status = MissionStatus.COMPLETED
        self._state.actual_end = time.time()
        self._log_event("mission_completed", {})
        self._trigger_status_callback()
        
        return True
    
    def fail(self, reason: str = "") -> bool:
        """Mark mission as failed."""
        if self._state.status not in (MissionStatus.EXECUTING, MissionStatus.PAUSED):
            return False
        
        self._state.status = MissionStatus.FAILED
        self._state.actual_end = time.time()
        self._log_event("mission_failed", {"reason": reason})
        self._trigger_status_callback()
        
        return True
    
    def abort(self, reason: str = "") -> bool:
        """Abort the mission."""
        if self._state.status == MissionStatus.COMPLETED:
            return False
        
        self._state.status = MissionStatus.ABORTED
        self._state.actual_end = time.time()
        self._log_event("mission_aborted", {"reason": reason})
        self._trigger_status_callback()
        
        return True
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        if task_id not in self._tasks:
            return
        
        old_status = self._state.task_statuses.get(task_id)
        self._state.task_statuses[task_id] = status
        
        # Track timing
        if status == TaskStatus.IN_PROGRESS and task_id not in self._task_start_times:
            self._task_start_times[task_id] = time.time()
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            self._task_end_times[task_id] = time.time()
        
        self._log_event("task_status_changed", {
            "task_id": task_id,
            "old_status": old_status.name if old_status else None,
            "new_status": status.name,
        })
        
        self._update_metrics()
        self._trigger_progress_callback()
        
        # Check for mission completion
        self._check_completion()
    
    def _check_completion(self) -> None:
        """Check if mission should be marked complete."""
        if self._state.status != MissionStatus.EXECUTING:
            return
        
        # All tasks completed or failed
        active_statuses = [TaskStatus.PENDING, TaskStatus.ASSIGNED, 
                         TaskStatus.IN_PROGRESS, TaskStatus.PAUSED]
        
        has_active = any(s in active_statuses 
                        for s in self._state.task_statuses.values())
        
        if not has_active and self._state.task_ids:
            # Check if too many failures
            failed = sum(1 for s in self._state.task_statuses.values() 
                        if s == TaskStatus.FAILED)
            
            if failed > len(self._state.task_ids) * 0.5:
                self.fail("Too many task failures")
            else:
                self.complete()
    
    def _update_metrics(self) -> None:
        """Update mission metrics."""
        statuses = list(self._state.task_statuses.values())
        
        completed = sum(1 for s in statuses if s == TaskStatus.COMPLETED)
        failed = sum(1 for s in statuses if s == TaskStatus.FAILED)
        active = sum(1 for s in statuses if s == TaskStatus.IN_PROGRESS)
        pending = sum(1 for s in statuses if s in (TaskStatus.PENDING, TaskStatus.ASSIGNED))
        
        # Task durations
        durations = []
        for task_id in self._task_end_times:
            if task_id in self._task_start_times:
                duration = self._task_end_times[task_id] - self._task_start_times[task_id]
                durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Objectives
        obj_completed = sum(1 for o in self._state.objectives if o.is_complete)
        
        completion_rate = completed / len(statuses) if statuses else 0.0
        
        self._state.metrics = MissionMetrics(
            total_tasks=len(statuses),
            completed_tasks=completed,
            failed_tasks=failed,
            active_tasks=active,
            pending_tasks=pending,
            total_objectives=len(self._state.objectives),
            completed_objectives=obj_completed,
            completion_rate=completion_rate,
            avg_task_duration=avg_duration,
            total_duration=self._state.elapsed_time,
            agents_assigned=len(self._state.assigned_agents),
            timestamp=time.time(),
        )
        
        # Store history
        self._metrics_history.append(self._state.metrics)
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a mission event."""
        self._event_log.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data,
        })
    
    def _trigger_status_callback(self) -> None:
        """Trigger status change callbacks."""
        for callback in self._status_callbacks:
            callback(self._state.status)
    
    def _trigger_progress_callback(self) -> None:
        """Trigger progress change callbacks."""
        for callback in self._progress_callbacks:
            callback(self.progress)
    
    def register_status_callback(self, callback: Callable[[MissionStatus], None]) -> None:
        """Register callback for status changes."""
        self._status_callbacks.append(callback)
    
    def register_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for progress changes."""
        self._progress_callbacks.append(callback)
    
    def get_event_log(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get mission event log."""
        if limit:
            return self._event_log[-limit:]
        return self._event_log.copy()
    
    def get_metrics_history(self) -> List[MissionMetrics]:
        """Get metrics history."""
        return self._metrics_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mission statistics."""
        return {
            "mission_id": self.mission_id,
            "status": self._state.status.name,
            "progress": self.progress,
            "elapsed_time": self._state.elapsed_time,
            "metrics": self._state.metrics.to_dict(),
            "num_events": len(self._event_log),
        }
