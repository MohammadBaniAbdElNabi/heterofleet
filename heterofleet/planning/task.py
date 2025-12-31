"""
Task representation and management for heterogeneous fleet operations.

Defines task types, constraints, and lifecycle management.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3


class TaskType(Enum):
    """Types of tasks in the system."""
    DELIVERY = auto()        # Deliver payload from A to B
    SURVEILLANCE = auto()    # Monitor an area
    INSPECTION = auto()      # Inspect a target
    SEARCH = auto()          # Search an area
    RESCUE = auto()          # Rescue operation
    MAPPING = auto()         # Map an area
    RELAY = auto()           # Communication relay
    ESCORT = auto()          # Escort another agent
    FORMATION = auto()       # Maintain formation
    CUSTOM = auto()          # Custom task


class TaskStatus(Enum):
    """Status of a task in its lifecycle."""
    PENDING = auto()         # Not yet assigned
    ASSIGNED = auto()        # Assigned but not started
    IN_PROGRESS = auto()     # Currently executing
    PAUSED = auto()          # Temporarily paused
    COMPLETED = auto()       # Successfully completed
    FAILED = auto()          # Failed to complete
    CANCELLED = auto()       # Cancelled before completion


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class TaskConstraints:
    """Constraints on task execution."""
    
    # Temporal constraints
    earliest_start: Optional[float] = None  # Unix timestamp
    deadline: Optional[float] = None        # Unix timestamp
    max_duration: Optional[float] = None    # seconds
    
    # Resource constraints
    required_capabilities: List[str] = field(default_factory=list)
    min_payload_capacity: float = 0.0       # kg
    min_sensor_range: float = 0.0           # meters
    
    # Platform constraints
    allowed_platforms: List[PlatformType] = field(default_factory=list)
    excluded_platforms: List[PlatformType] = field(default_factory=list)
    min_agents: int = 1
    max_agents: int = 1
    
    # Energy constraints
    estimated_energy: float = 0.0           # Wh
    
    # Network constraints
    min_network_quality: float = 0.0        # 0-1
    requires_continuous_comm: bool = False
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    blocks: List[str] = field(default_factory=list)      # Task IDs
    
    def is_platform_allowed(self, platform_type: PlatformType) -> bool:
        """Check if a platform type is allowed for this task."""
        # Check exclusions first
        if platform_type in self.excluded_platforms:
            return False
        
        # If allowed list is specified, must be in it
        if self.allowed_platforms:
            return platform_type in self.allowed_platforms
        
        return True
    
    def check_temporal_feasibility(self, current_time: float, expected_duration: float) -> bool:
        """Check if task can be completed within temporal constraints."""
        # Check earliest start
        if self.earliest_start is not None and current_time < self.earliest_start:
            return False
        
        # Check deadline
        if self.deadline is not None:
            completion_time = current_time + expected_duration
            if completion_time > self.deadline:
                return False
        
        # Check max duration
        if self.max_duration is not None and expected_duration > self.max_duration:
            return False
        
        return True


@dataclass
class TaskLocation:
    """Location information for a task."""
    
    # Primary location
    position: Vector3 = field(default_factory=Vector3)
    
    # Optional area bounds (for area tasks)
    area_min: Optional[Vector3] = None
    area_max: Optional[Vector3] = None
    
    # Waypoints (for path-based tasks)
    waypoints: List[Vector3] = field(default_factory=list)
    
    # Altitude constraints
    min_altitude: float = 0.0
    max_altitude: float = 10.0
    
    # Approach direction (optional)
    approach_direction: Optional[Vector3] = None
    
    @property
    def is_area_task(self) -> bool:
        """Check if this is an area-based task."""
        return self.area_min is not None and self.area_max is not None
    
    @property
    def area_size(self) -> float:
        """Get area size in square meters."""
        if not self.is_area_task:
            return 0.0
        
        dx = self.area_max.x - self.area_min.x
        dy = self.area_max.y - self.area_min.y
        return dx * dy
    
    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is within the task area."""
        if not self.is_area_task:
            # Point task - check distance
            return (point - self.position).norm() < 0.5
        
        return (self.area_min.x <= point.x <= self.area_max.x and
                self.area_min.y <= point.y <= self.area_max.y and
                self.min_altitude <= point.z <= self.max_altitude)


@dataclass
class TaskResult:
    """Result of task execution."""
    
    status: TaskStatus = TaskStatus.PENDING
    completion_percentage: float = 0.0
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Outcome data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@dataclass
class Task:
    """
    Complete task representation.
    
    A task is a unit of work that can be assigned to one or more agents.
    """
    
    # Identity
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = TaskType.CUSTOM
    name: str = ""
    description: str = ""
    
    # Priority
    priority: TaskPriority = TaskPriority.NORMAL
    priority_value: float = 0.5  # Numeric priority 0-1
    
    # Location
    location: TaskLocation = field(default_factory=TaskLocation)
    
    # Constraints
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    
    # Assignment
    assigned_agents: List[str] = field(default_factory=list)
    
    # Status and result
    status: TaskStatus = TaskStatus.PENDING
    result: TaskResult = field(default_factory=TaskResult)
    
    # Timing
    created_time: float = field(default_factory=time.time)
    
    # Callbacks
    _on_complete: Optional[Callable[[Task], None]] = None
    _on_fail: Optional[Callable[[Task], None]] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if not self.name:
            self.name = f"{self.task_type.name}_{self.task_id}"
    
    def assign(self, agent_id: str) -> bool:
        """
        Assign an agent to this task.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if assignment successful
        """
        if len(self.assigned_agents) >= self.constraints.max_agents:
            return False
        
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)
            
            if self.status == TaskStatus.PENDING:
                self.status = TaskStatus.ASSIGNED
            
            return True
        
        return False
    
    def unassign(self, agent_id: str) -> bool:
        """Remove agent assignment."""
        if agent_id in self.assigned_agents:
            self.assigned_agents.remove(agent_id)
            
            if not self.assigned_agents:
                self.status = TaskStatus.PENDING
            
            return True
        
        return False
    
    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.result.status = TaskStatus.IN_PROGRESS
        self.result.start_time = time.time()
    
    def complete(self, data: Dict[str, Any] = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result.status = TaskStatus.COMPLETED
        self.result.end_time = time.time()
        self.result.completion_percentage = 100.0
        
        if data:
            self.result.data.update(data)
        
        if self._on_complete:
            self._on_complete(self)
    
    def fail(self, error_message: str = None) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.result.status = TaskStatus.FAILED
        self.result.end_time = time.time()
        self.result.error_message = error_message
        
        if self._on_fail:
            self._on_fail(self)
    
    def pause(self) -> None:
        """Pause task execution."""
        if self.status == TaskStatus.IN_PROGRESS:
            self.status = TaskStatus.PAUSED
            self.result.status = TaskStatus.PAUSED
    
    def resume(self) -> None:
        """Resume paused task."""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.IN_PROGRESS
            self.result.status = TaskStatus.IN_PROGRESS
    
    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.result.status = TaskStatus.CANCELLED
        self.result.end_time = time.time()
    
    def update_progress(self, percentage: float) -> None:
        """Update task progress."""
        self.result.completion_percentage = min(100.0, max(0.0, percentage))
    
    def estimate_duration(self, speed: float) -> float:
        """
        Estimate task duration based on travel distance.
        
        Args:
            speed: Agent speed in m/s
            
        Returns:
            Estimated duration in seconds
        """
        if speed <= 0:
            return float('inf')
        
        if self.location.waypoints:
            # Sum of waypoint distances
            total_dist = 0.0
            for i in range(len(self.location.waypoints) - 1):
                total_dist += (
                    self.location.waypoints[i + 1] - 
                    self.location.waypoints[i]
                ).norm()
            
            return total_dist / speed
        
        elif self.location.is_area_task:
            # Estimate coverage time
            area = self.location.area_size
            sweep_width = 5.0  # Assumed sensor width
            path_length = area / sweep_width
            
            return path_length / speed + 30.0  # Add setup time
        
        else:
            # Simple point task
            return 30.0  # Default duration
    
    @property
    def is_active(self) -> bool:
        """Check if task is currently active."""
        return self.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.PAUSED)
    
    @property
    def is_complete(self) -> bool:
        """Check if task is finished (success or failure)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.name,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "priority_value": self.priority_value,
            "position": (self.location.position.x, 
                        self.location.position.y, 
                        self.location.position.z),
            "status": self.status.name,
            "assigned_agents": self.assigned_agents,
            "progress": self.result.completion_percentage,
            "created_time": self.created_time,
        }
    
    @classmethod
    def create_delivery_task(
        cls,
        pickup_pos: Vector3,
        dropoff_pos: Vector3,
        payload_weight: float,
        deadline: float = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Task:
        """Create a delivery task."""
        location = TaskLocation(
            position=pickup_pos,
            waypoints=[pickup_pos, dropoff_pos]
        )
        
        constraints = TaskConstraints(
            deadline=deadline,
            min_payload_capacity=payload_weight,
            required_capabilities=["delivery"]
        )
        
        return cls(
            task_type=TaskType.DELIVERY,
            name=f"Delivery to ({dropoff_pos.x:.1f}, {dropoff_pos.y:.1f})",
            priority=priority,
            location=location,
            constraints=constraints
        )
    
    @classmethod
    def create_surveillance_task(
        cls,
        area_min: Vector3,
        area_max: Vector3,
        duration: float,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Task:
        """Create a surveillance task."""
        center = Vector3(
            (area_min.x + area_max.x) / 2,
            (area_min.y + area_max.y) / 2,
            (area_min.z + area_max.z) / 2
        )
        
        location = TaskLocation(
            position=center,
            area_min=area_min,
            area_max=area_max
        )
        
        constraints = TaskConstraints(
            max_duration=duration,
            required_capabilities=["surveillance", "camera"]
        )
        
        return cls(
            task_type=TaskType.SURVEILLANCE,
            name=f"Surveillance of area",
            priority=priority,
            location=location,
            constraints=constraints
        )


class TaskManager:
    """
    Manager for task lifecycle and tracking.
    
    Handles task creation, assignment, status updates, and queries.
    """
    
    def __init__(self):
        """Initialize task manager."""
        self._tasks: Dict[str, Task] = {}
        self._agent_tasks: Dict[str, List[str]] = {}  # agent_id -> task_ids
        
        # Callbacks
        self._on_task_complete: List[Callable[[Task], None]] = []
        self._on_task_fail: List[Callable[[Task], None]] = []
    
    def add_task(self, task: Task) -> None:
        """Add a task to the manager."""
        self._tasks[task.task_id] = task
        
        # Set up callbacks
        task._on_complete = self._handle_task_complete
        task._on_fail = self._handle_task_fail
        
        logger.info(f"Added task: {task.task_id} ({task.task_type.name})")
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """Remove a task from the manager."""
        task = self._tasks.pop(task_id, None)
        
        if task:
            # Remove from agent assignments
            for agent_id in task.assigned_agents:
                if agent_id in self._agent_tasks:
                    if task_id in self._agent_tasks[agent_id]:
                        self._agent_tasks[agent_id].remove(task_id)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return self.get_tasks_by_status(TaskStatus.PENDING)
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks."""
        return [t for t in self._tasks.values() if t.is_active]
    
    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get tasks assigned to an agent."""
        task_ids = self._agent_tasks.get(agent_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        
        if task.assign(agent_id):
            if agent_id not in self._agent_tasks:
                self._agent_tasks[agent_id] = []
            self._agent_tasks[agent_id].append(task_id)
            
            logger.info(f"Assigned task {task_id} to agent {agent_id}")
            return True
        
        return False
    
    def unassign_task(self, task_id: str, agent_id: str) -> bool:
        """Unassign a task from an agent."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        
        if task.unassign(agent_id):
            if agent_id in self._agent_tasks:
                if task_id in self._agent_tasks[agent_id]:
                    self._agent_tasks[agent_id].remove(task_id)
            return True
        
        return False
    
    def start_task(self, task_id: str) -> bool:
        """Start a task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.ASSIGNED:
            task.start()
            return True
        return False
    
    def complete_task(self, task_id: str, data: Dict[str, Any] = None) -> bool:
        """Mark a task as complete."""
        task = self._tasks.get(task_id)
        if task and task.is_active:
            task.complete(data)
            return True
        return False
    
    def fail_task(self, task_id: str, error: str = None) -> bool:
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if task and task.is_active:
            task.fail(error)
            return True
        return False
    
    def _handle_task_complete(self, task: Task) -> None:
        """Handle task completion."""
        for callback in self._on_task_complete:
            callback(task)
    
    def _handle_task_fail(self, task: Task) -> None:
        """Handle task failure."""
        for callback in self._on_task_fail:
            callback(task)
    
    def register_completion_callback(
        self,
        callback: Callable[[Task], None]
    ) -> None:
        """Register callback for task completion."""
        self._on_task_complete.append(callback)
    
    def register_failure_callback(
        self,
        callback: Callable[[Task], None]
    ) -> None:
        """Register callback for task failure."""
        self._on_task_fail.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.name] = len(self.get_tasks_by_status(status))
        
        type_counts = {}
        for task in self._tasks.values():
            type_name = task.task_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_tasks": len(self._tasks),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "agents_with_tasks": len([a for a in self._agent_tasks if self._agent_tasks[a]]),
        }
    
    def check_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.constraints.depends_on:
            dep_task = self._tasks.get(dep_id)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to be assigned (dependencies met)."""
        ready = []
        for task in self.get_pending_tasks():
            if self.check_dependencies_satisfied(task):
                ready.append(task)
        return ready
