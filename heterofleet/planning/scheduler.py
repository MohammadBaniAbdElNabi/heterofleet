"""
Temporal Scheduler for heterogeneous fleet operations.

Handles temporal constraint scheduling, dependency resolution,
and execution timeline management.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import Vector3
from heterofleet.planning.task import Task, TaskStatus, TaskType


class ScheduleEventType(Enum):
    """Types of schedule events."""
    TASK_START = auto()
    TASK_END = auto()
    AGENT_AVAILABLE = auto()
    DEADLINE_WARNING = auto()
    DEPENDENCY_MET = auto()


@dataclass(order=True)
class ScheduleEvent:
    """Event in the schedule timeline."""
    
    time: float
    event_type: ScheduleEventType = field(compare=False)
    task_id: Optional[str] = field(default=None, compare=False)
    agent_id: Optional[str] = field(default=None, compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class ScheduleEntry:
    """Entry in the schedule for a task."""
    
    task_id: str
    agent_id: str
    
    # Timing
    scheduled_start: float
    scheduled_end: float
    actual_start: Optional[float] = None
    actual_end: Optional[float] = None
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Travel info
    travel_time: float = 0.0
    execution_time: float = 0.0
    
    # Position info
    start_position: Optional[Vector3] = None
    end_position: Optional[Vector3] = None
    
    @property
    def duration(self) -> float:
        """Get scheduled duration."""
        return self.scheduled_end - self.scheduled_start
    
    @property
    def is_active(self) -> bool:
        """Check if entry is currently active."""
        return self.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    
    @property
    def delay(self) -> float:
        """Get delay from scheduled start."""
        if self.actual_start is None:
            return 0.0
        return max(0.0, self.actual_start - self.scheduled_start)


@dataclass
class Schedule:
    """Complete schedule for all agents."""
    
    entries: Dict[str, ScheduleEntry] = field(default_factory=dict)  # task_id -> entry
    agent_schedules: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> [task_ids]
    
    # Timeline
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    version: int = 1
    
    def add_entry(self, entry: ScheduleEntry) -> None:
        """Add an entry to the schedule."""
        self.entries[entry.task_id] = entry
        
        if entry.agent_id not in self.agent_schedules:
            self.agent_schedules[entry.agent_id] = []
        
        # Insert in sorted order
        agent_tasks = self.agent_schedules[entry.agent_id]
        insert_idx = 0
        for i, tid in enumerate(agent_tasks):
            if self.entries[tid].scheduled_start > entry.scheduled_start:
                insert_idx = i
                break
            insert_idx = i + 1
        
        agent_tasks.insert(insert_idx, entry.task_id)
        
        # Update timeline
        self.end_time = max(self.end_time, entry.scheduled_end)
        if self.start_time == 0 or entry.scheduled_start < self.start_time:
            self.start_time = entry.scheduled_start
    
    def remove_entry(self, task_id: str) -> Optional[ScheduleEntry]:
        """Remove an entry from the schedule."""
        entry = self.entries.pop(task_id, None)
        
        if entry and entry.agent_id in self.agent_schedules:
            if task_id in self.agent_schedules[entry.agent_id]:
                self.agent_schedules[entry.agent_id].remove(task_id)
        
        return entry
    
    def get_agent_tasks(self, agent_id: str) -> List[ScheduleEntry]:
        """Get scheduled tasks for an agent."""
        task_ids = self.agent_schedules.get(agent_id, [])
        return [self.entries[tid] for tid in task_ids if tid in self.entries]
    
    def get_next_task(self, agent_id: str, current_time: float) -> Optional[ScheduleEntry]:
        """Get next scheduled task for an agent."""
        for entry in self.get_agent_tasks(agent_id):
            if entry.scheduled_start >= current_time and entry.status == TaskStatus.PENDING:
                return entry
        return None
    
    def get_current_task(self, agent_id: str, current_time: float) -> Optional[ScheduleEntry]:
        """Get currently executing task for an agent."""
        for entry in self.get_agent_tasks(agent_id):
            if entry.status == TaskStatus.IN_PROGRESS:
                return entry
            if (entry.scheduled_start <= current_time <= entry.scheduled_end and
                entry.status in (TaskStatus.ASSIGNED, TaskStatus.PENDING)):
                return entry
        return None
    
    def get_conflicts(self) -> List[Tuple[str, str]]:
        """Find conflicting schedule entries."""
        conflicts = []
        
        for agent_id, task_ids in self.agent_schedules.items():
            entries = [self.entries[tid] for tid in task_ids]
            entries.sort(key=lambda e: e.scheduled_start)
            
            for i in range(len(entries) - 1):
                if entries[i].scheduled_end > entries[i + 1].scheduled_start:
                    conflicts.append((entries[i].task_id, entries[i + 1].task_id))
        
        return conflicts
    
    def get_makespan(self) -> float:
        """Get total schedule makespan."""
        return self.end_time - self.start_time
    
    def get_agent_utilization(self, agent_id: str) -> float:
        """Get utilization percentage for an agent."""
        entries = self.get_agent_tasks(agent_id)
        if not entries:
            return 0.0
        
        total_time = sum(e.duration for e in entries)
        makespan = self.get_makespan()
        
        return total_time / makespan if makespan > 0 else 0.0
    
    def to_timeline(self) -> List[ScheduleEvent]:
        """Convert schedule to event timeline."""
        events = []
        
        for entry in self.entries.values():
            events.append(ScheduleEvent(
                time=entry.scheduled_start,
                event_type=ScheduleEventType.TASK_START,
                task_id=entry.task_id,
                agent_id=entry.agent_id,
            ))
            events.append(ScheduleEvent(
                time=entry.scheduled_end,
                event_type=ScheduleEventType.TASK_END,
                task_id=entry.task_id,
                agent_id=entry.agent_id,
            ))
        
        events.sort(key=lambda e: e.time)
        return events


class TemporalScheduler:
    """
    Temporal scheduler for task execution planning.
    
    Handles:
    - Temporal constraint satisfaction
    - Dependency resolution
    - Schedule optimization
    - Real-time rescheduling
    """
    
    def __init__(self):
        """Initialize temporal scheduler."""
        self._tasks: Dict[str, Task] = {}
        self._schedule = Schedule()
        
        # Agent availability tracking
        self._agent_positions: Dict[str, Vector3] = {}
        self._agent_available_times: Dict[str, float] = {}
        self._agent_speeds: Dict[str, float] = {}
        
        # Event queue
        self._event_queue: List[ScheduleEvent] = []
        
        # Callbacks
        self._event_callbacks: List[Callable[[ScheduleEvent], None]] = []
    
    def set_agent_info(
        self,
        agent_id: str,
        position: Vector3,
        speed: float,
        available_time: float = 0.0
    ) -> None:
        """Set agent information for scheduling."""
        self._agent_positions[agent_id] = position
        self._agent_speeds[agent_id] = speed
        self._agent_available_times[agent_id] = available_time
    
    def add_task(self, task: Task) -> None:
        """Add a task to be scheduled."""
        self._tasks[task.task_id] = task
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from scheduling."""
        self._tasks.pop(task_id, None)
        self._schedule.remove_entry(task_id)
    
    def schedule_task(
        self,
        task: Task,
        agent_id: str,
        start_time: float = None
    ) -> Optional[ScheduleEntry]:
        """
        Schedule a single task for an agent.
        
        Args:
            task: Task to schedule
            agent_id: Agent to assign
            start_time: Start time (default: earliest feasible)
            
        Returns:
            Schedule entry if successful, None otherwise
        """
        if agent_id not in self._agent_positions:
            logger.warning(f"Unknown agent: {agent_id}")
            return None
        
        # Check dependencies
        if not self._check_dependencies_scheduled(task):
            logger.warning(f"Task {task.task_id} has unscheduled dependencies")
            return None
        
        # Compute earliest start time
        earliest_start = self._compute_earliest_start(task, agent_id)
        
        if start_time is not None:
            if start_time < earliest_start:
                logger.warning(f"Requested start time {start_time} is before "
                             f"earliest feasible {earliest_start}")
                return None
            earliest_start = start_time
        
        # Compute travel and execution time
        agent_pos = self._agent_positions[agent_id]
        agent_speed = self._agent_speeds.get(agent_id, 1.0)
        
        travel_time = (task.location.position - agent_pos).norm() / agent_speed
        execution_time = task.estimate_duration(agent_speed)
        
        # Check deadline
        end_time = earliest_start + travel_time + execution_time
        
        if task.constraints.deadline is not None:
            if end_time > task.constraints.deadline:
                logger.warning(f"Task {task.task_id} would miss deadline")
                return None
        
        # Create entry
        entry = ScheduleEntry(
            task_id=task.task_id,
            agent_id=agent_id,
            scheduled_start=earliest_start,
            scheduled_end=end_time,
            travel_time=travel_time,
            execution_time=execution_time,
            start_position=agent_pos,
            end_position=task.location.position,
            status=TaskStatus.ASSIGNED,
        )
        
        # Add to schedule
        self._schedule.add_entry(entry)
        
        # Update agent availability
        self._agent_available_times[agent_id] = end_time
        self._agent_positions[agent_id] = task.location.position
        
        # Add events
        self._add_schedule_events(entry)
        
        return entry
    
    def _check_dependencies_scheduled(self, task: Task) -> bool:
        """Check if all task dependencies are scheduled."""
        for dep_id in task.constraints.depends_on:
            if dep_id not in self._schedule.entries:
                return False
        return True
    
    def _compute_earliest_start(self, task: Task, agent_id: str) -> float:
        """Compute earliest feasible start time for a task."""
        earliest = self._agent_available_times.get(agent_id, 0.0)
        
        # Check earliest_start constraint
        if task.constraints.earliest_start is not None:
            earliest = max(earliest, task.constraints.earliest_start)
        
        # Check dependency completion times
        for dep_id in task.constraints.depends_on:
            dep_entry = self._schedule.entries.get(dep_id)
            if dep_entry:
                earliest = max(earliest, dep_entry.scheduled_end)
        
        return earliest
    
    def _add_schedule_events(self, entry: ScheduleEntry) -> None:
        """Add events for a schedule entry."""
        heapq.heappush(self._event_queue, ScheduleEvent(
            time=entry.scheduled_start,
            event_type=ScheduleEventType.TASK_START,
            task_id=entry.task_id,
            agent_id=entry.agent_id,
        ))
        
        heapq.heappush(self._event_queue, ScheduleEvent(
            time=entry.scheduled_end,
            event_type=ScheduleEventType.TASK_END,
            task_id=entry.task_id,
            agent_id=entry.agent_id,
        ))
        
        # Add deadline warning if applicable
        task = self._tasks.get(entry.task_id)
        if task and task.constraints.deadline is not None:
            warning_time = task.constraints.deadline - 60.0  # 1 minute warning
            if warning_time > entry.scheduled_start:
                heapq.heappush(self._event_queue, ScheduleEvent(
                    time=warning_time,
                    event_type=ScheduleEventType.DEADLINE_WARNING,
                    task_id=entry.task_id,
                    agent_id=entry.agent_id,
                ))
    
    def schedule_all(
        self,
        assignments: Dict[str, str],
        current_time: float = 0.0
    ) -> Schedule:
        """
        Schedule all assigned tasks.
        
        Args:
            assignments: task_id -> agent_id mapping
            current_time: Current simulation time
            
        Returns:
            Complete schedule
        """
        # Reset schedule
        self._schedule = Schedule()
        self._event_queue = []
        
        # Reset agent availability
        for agent_id in self._agent_available_times:
            self._agent_available_times[agent_id] = current_time
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(list(self._tasks.values()))
        
        # Schedule each task
        for task in sorted_tasks:
            if task.task_id not in assignments:
                continue
            
            agent_id = assignments[task.task_id]
            entry = self.schedule_task(task, agent_id)
            
            if entry is None:
                logger.warning(f"Failed to schedule task {task.task_id}")
        
        return self._schedule
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks respecting dependencies and priority."""
        # Build dependency graph
        in_degree = {t.task_id: 0 for t in tasks}
        dependents = {t.task_id: [] for t in tasks}
        
        task_map = {t.task_id: t for t in tasks}
        
        for task in tasks:
            for dep_id in task.constraints.depends_on:
                if dep_id in task_map:
                    in_degree[task.task_id] += 1
                    dependents[dep_id].append(task.task_id)
        
        # Priority queue (negative priority for max-heap behavior)
        queue = [(-task.priority_value, task.task_id) 
                for task in tasks if in_degree[task.task_id] == 0]
        heapq.heapify(queue)
        
        sorted_tasks = []
        
        while queue:
            _, task_id = heapq.heappop(queue)
            sorted_tasks.append(task_map[task_id])
            
            for dep_id in dependents[task_id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    heapq.heappush(queue, (-task_map[dep_id].priority_value, dep_id))
        
        return sorted_tasks
    
    def get_schedule(self) -> Schedule:
        """Get current schedule."""
        return self._schedule
    
    def get_next_event(self, current_time: float) -> Optional[ScheduleEvent]:
        """Get next event after current time."""
        while self._event_queue:
            event = self._event_queue[0]
            if event.time >= current_time:
                return event
            heapq.heappop(self._event_queue)
        return None
    
    def process_events(self, current_time: float) -> List[ScheduleEvent]:
        """Process all events up to current time."""
        processed = []
        
        while self._event_queue and self._event_queue[0].time <= current_time:
            event = heapq.heappop(self._event_queue)
            processed.append(event)
            
            # Trigger callbacks
            for callback in self._event_callbacks:
                callback(event)
        
        return processed
    
    def register_event_callback(self, callback: Callable[[ScheduleEvent], None]) -> None:
        """Register callback for schedule events."""
        self._event_callbacks.append(callback)
    
    def reschedule_task(
        self,
        task_id: str,
        new_agent_id: str = None,
        new_start_time: float = None
    ) -> Optional[ScheduleEntry]:
        """
        Reschedule a task.
        
        Args:
            task_id: Task to reschedule
            new_agent_id: New agent (default: keep current)
            new_start_time: New start time (default: earliest feasible)
            
        Returns:
            New schedule entry if successful
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        
        # Get current entry
        current_entry = self._schedule.entries.get(task_id)
        if current_entry is None:
            return None
        
        agent_id = new_agent_id or current_entry.agent_id
        
        # Remove current entry
        self._schedule.remove_entry(task_id)
        
        # Remove old events (simplified - would need more sophisticated handling)
        self._event_queue = [e for e in self._event_queue if e.task_id != task_id]
        heapq.heapify(self._event_queue)
        
        # Schedule with new parameters
        return self.schedule_task(task, agent_id, new_start_time)
    
    def mark_task_started(self, task_id: str, actual_start: float) -> None:
        """Mark a task as started."""
        entry = self._schedule.entries.get(task_id)
        if entry:
            entry.status = TaskStatus.IN_PROGRESS
            entry.actual_start = actual_start
    
    def mark_task_completed(self, task_id: str, actual_end: float) -> None:
        """Mark a task as completed."""
        entry = self._schedule.entries.get(task_id)
        if entry:
            entry.status = TaskStatus.COMPLETED
            entry.actual_end = actual_end
            
            # Trigger dependency events
            for task in self._tasks.values():
                if task_id in task.constraints.depends_on:
                    if self._check_dependencies_scheduled(task):
                        heapq.heappush(self._event_queue, ScheduleEvent(
                            time=actual_end,
                            event_type=ScheduleEventType.DEPENDENCY_MET,
                            task_id=task.task_id,
                        ))
    
    def mark_task_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        entry = self._schedule.entries.get(task_id)
        if entry:
            entry.status = TaskStatus.FAILED
    
    def get_schedule_statistics(self) -> Dict[str, Any]:
        """Get schedule statistics."""
        entries = list(self._schedule.entries.values())
        
        if not entries:
            return {"empty": True}
        
        total_duration = sum(e.duration for e in entries)
        total_travel = sum(e.travel_time for e in entries)
        total_execution = sum(e.execution_time for e in entries)
        
        delays = [e.delay for e in entries if e.actual_start is not None]
        
        return {
            "num_tasks": len(entries),
            "makespan": self._schedule.get_makespan(),
            "total_duration": total_duration,
            "total_travel_time": total_travel,
            "total_execution_time": total_execution,
            "travel_fraction": total_travel / total_duration if total_duration > 0 else 0,
            "num_agents": len(self._schedule.agent_schedules),
            "avg_delay": np.mean(delays) if delays else 0.0,
            "max_delay": max(delays) if delays else 0.0,
            "conflicts": len(self._schedule.get_conflicts()),
        }
    
    def optimize_schedule(self) -> None:
        """
        Optimize current schedule to minimize makespan.
        
        Uses simple local search to improve schedule.
        """
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try swapping adjacent tasks for each agent
            for agent_id, task_ids in self._schedule.agent_schedules.items():
                for i in range(len(task_ids) - 1):
                    # Check if swap is feasible
                    task1 = self._tasks.get(task_ids[i])
                    task2 = self._tasks.get(task_ids[i + 1])
                    
                    if task1 is None or task2 is None:
                        continue
                    
                    # Check dependencies
                    if task_ids[i] in task2.constraints.depends_on:
                        continue
                    
                    # Try swap
                    old_makespan = self._schedule.get_makespan()
                    
                    # Swap in list
                    task_ids[i], task_ids[i + 1] = task_ids[i + 1], task_ids[i]
                    
                    # Recompute times
                    self._recompute_agent_schedule(agent_id)
                    
                    new_makespan = self._schedule.get_makespan()
                    
                    if new_makespan < old_makespan:
                        improved = True
                    else:
                        # Revert
                        task_ids[i], task_ids[i + 1] = task_ids[i + 1], task_ids[i]
                        self._recompute_agent_schedule(agent_id)
    
    def _recompute_agent_schedule(self, agent_id: str) -> None:
        """Recompute schedule times for an agent."""
        task_ids = self._schedule.agent_schedules.get(agent_id, [])
        if not task_ids:
            return
        
        current_time = 0.0
        current_pos = self._agent_positions.get(agent_id, Vector3(0, 0, 0))
        speed = self._agent_speeds.get(agent_id, 1.0)
        
        for task_id in task_ids:
            task = self._tasks.get(task_id)
            entry = self._schedule.entries.get(task_id)
            
            if task is None or entry is None:
                continue
            
            # Travel time
            travel_time = (task.location.position - current_pos).norm() / speed
            
            # Execution time
            exec_time = task.estimate_duration(speed)
            
            # Update entry
            entry.scheduled_start = current_time
            entry.scheduled_end = current_time + travel_time + exec_time
            entry.travel_time = travel_time
            entry.execution_time = exec_time
            entry.start_position = current_pos
            entry.end_position = task.location.position
            
            # Update state
            current_time = entry.scheduled_end
            current_pos = task.location.position
        
        # Update schedule end time
        self._schedule.end_time = max(
            e.scheduled_end for e in self._schedule.entries.values()
        ) if self._schedule.entries else 0.0
