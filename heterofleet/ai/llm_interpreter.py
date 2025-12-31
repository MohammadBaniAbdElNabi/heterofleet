"""
LLM-based Mission Interpreter for HeteroFleet.

Translates natural language commands into structured mission plans.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.planning.task import Task, TaskType, TaskPriority


class MissionIntentType(Enum):
    """Types of mission intents."""
    SURVEY = auto()
    SEARCH = auto()
    DELIVERY = auto()
    ESCORT = auto()
    PATROL = auto()
    INSPECTION = auto()
    MAPPING = auto()
    MONITORING = auto()
    CUSTOM = auto()


@dataclass
class MissionIntent:
    """Parsed mission intent from natural language."""
    
    intent_type: MissionIntentType = MissionIntentType.CUSTOM
    confidence: float = 0.0
    
    # Target area/location
    target_area: Optional[Tuple[Vector3, Vector3]] = None  # bounding box
    target_points: List[Vector3] = field(default_factory=list)
    
    # Constraints
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Platform requirements
    required_platforms: List[PlatformType] = field(default_factory=list)
    min_agents: int = 1
    max_agents: Optional[int] = None
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Original text
    original_text: str = ""


@dataclass
class InterpretationResult:
    """Result of LLM interpretation."""
    
    success: bool = False
    intent: Optional[MissionIntent] = None
    tasks: List[Task] = field(default_factory=list)
    
    # Clarification needed
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    
    # Explanation
    explanation: str = ""
    raw_response: str = ""


class LLMInterpreter:
    """
    LLM-based interpreter for natural language mission commands.
    
    Uses pattern matching and structured parsing for offline operation,
    with optional LLM API integration for complex commands.
    """
    
    def __init__(self, use_api: bool = False, api_key: str = None):
        """
        Initialize LLM interpreter.
        
        Args:
            use_api: Whether to use external LLM API
            api_key: API key for LLM service
        """
        self.use_api = use_api
        self.api_key = api_key
        
        # Intent patterns
        self._intent_patterns = {
            MissionIntentType.SURVEY: [
                r"survey\s+(?:the\s+)?(\w+)",
                r"scan\s+(?:the\s+)?(\w+)",
                r"cover\s+(?:the\s+)?area",
            ],
            MissionIntentType.SEARCH: [
                r"search\s+for\s+(\w+)",
                r"find\s+(?:the\s+)?(\w+)",
                r"locate\s+(\w+)",
                r"look\s+for\s+(\w+)",
            ],
            MissionIntentType.DELIVERY: [
                r"deliver\s+(\w+)\s+to",
                r"transport\s+(\w+)",
                r"carry\s+(\w+)",
                r"bring\s+(\w+)",
            ],
            MissionIntentType.PATROL: [
                r"patrol\s+(?:the\s+)?(\w+)",
                r"guard\s+(?:the\s+)?(\w+)",
                r"monitor\s+(?:the\s+)?perimeter",
            ],
            MissionIntentType.INSPECTION: [
                r"inspect\s+(?:the\s+)?(\w+)",
                r"check\s+(?:the\s+)?(\w+)",
                r"examine\s+(?:the\s+)?(\w+)",
            ],
            MissionIntentType.MAPPING: [
                r"map\s+(?:the\s+)?(\w+)",
                r"create\s+(?:a\s+)?map",
                r"build\s+(?:a\s+)?map",
            ],
        }
        
        # Location patterns
        self._location_patterns = [
            r"at\s+\(?([-\d.]+)[,\s]+([-\d.]+)[,\s]*([-\d.]*)\)?",
            r"position\s+\(?([-\d.]+)[,\s]+([-\d.]+)[,\s]*([-\d.]*)\)?",
            r"coordinates?\s+\(?([-\d.]+)[,\s]+([-\d.]+)[,\s]*([-\d.]*)\)?",
        ]
        
        # Area patterns
        self._area_patterns = [
            r"area\s+from\s+\(?([-\d.]+)[,\s]+([-\d.]+)\)?\s+to\s+\(?([-\d.]+)[,\s]+([-\d.]+)\)?",
            r"region\s+\(?([-\d.]+)[,\s]+([-\d.]+)\)?\s+to\s+\(?([-\d.]+)[,\s]+([-\d.]+)\)?",
            r"(\d+)\s*(?:m|meter)s?\s+by\s+(\d+)\s*(?:m|meter)s?",
        ]
        
        # Time patterns
        self._time_patterns = [
            r"within\s+(\d+)\s*(second|minute|hour)s?",
            r"in\s+(\d+)\s*(second|minute|hour)s?",
            r"before\s+(\d+)\s*(second|minute|hour)s?",
        ]
        
        # Platform patterns
        self._platform_keywords = {
            "drone": PlatformType.SMALL_UAV,
            "drones": PlatformType.SMALL_UAV,
            "uav": PlatformType.SMALL_UAV,
            "uavs": PlatformType.SMALL_UAV,
            "quadcopter": PlatformType.SMALL_UAV,
            "crazyflie": PlatformType.MICRO_UAV,
            "ugv": PlatformType.SMALL_UGV,
            "rover": PlatformType.SMALL_UGV,
            "robot": PlatformType.SMALL_UGV,
            "ground robot": PlatformType.SMALL_UGV,
            "mentorpi": PlatformType.SMALL_UGV,
        }
    
    def interpret(self, command: str) -> InterpretationResult:
        """
        Interpret a natural language command.
        
        Args:
            command: Natural language command
            
        Returns:
            Interpretation result
        """
        command = command.strip().lower()
        
        if not command:
            return InterpretationResult(
                success=False,
                explanation="Empty command provided"
            )
        
        # Parse intent
        intent = self._parse_intent(command)
        
        if intent.confidence < 0.3:
            return InterpretationResult(
                success=False,
                needs_clarification=True,
                clarification_questions=[
                    "What type of mission would you like to execute?",
                    "Could you specify the target area or location?",
                ],
                explanation="Could not understand the mission intent"
            )
        
        # Generate tasks
        tasks = self._generate_tasks(intent)
        
        if not tasks:
            return InterpretationResult(
                success=False,
                intent=intent,
                needs_clarification=True,
                clarification_questions=[
                    "Please specify the target location or area",
                ],
                explanation="Could not generate tasks without location information"
            )
        
        return InterpretationResult(
            success=True,
            intent=intent,
            tasks=tasks,
            explanation=self._generate_explanation(intent, tasks)
        )
    
    def _parse_intent(self, command: str) -> MissionIntent:
        """Parse mission intent from command."""
        intent = MissionIntent(original_text=command)
        
        # Match intent type
        best_match = None
        best_confidence = 0.0
        
        for intent_type, patterns in self._intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command)
                if match:
                    confidence = 0.8  # Base confidence for pattern match
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
        
        if best_match:
            intent.intent_type = best_match
            intent.confidence = best_confidence
        else:
            intent.intent_type = MissionIntentType.CUSTOM
            intent.confidence = 0.2
        
        # Parse locations
        intent.target_points = self._parse_locations(command)
        
        # Parse area
        intent.target_area = self._parse_area(command)
        
        # Parse deadline
        intent.deadline = self._parse_deadline(command)
        
        # Parse priority
        intent.priority = self._parse_priority(command)
        
        # Parse platform requirements
        intent.required_platforms = self._parse_platforms(command)
        
        # Parse agent count
        intent.min_agents, intent.max_agents = self._parse_agent_count(command)
        
        # Adjust confidence based on completeness
        if intent.target_points or intent.target_area:
            intent.confidence = min(1.0, intent.confidence + 0.1)
        else:
            intent.confidence = max(0.1, intent.confidence - 0.2)
        
        return intent
    
    def _parse_locations(self, command: str) -> List[Vector3]:
        """Parse location coordinates from command."""
        locations = []
        
        for pattern in self._location_patterns:
            matches = re.finditer(pattern, command)
            for match in matches:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3)) if match.group(3) else 0.0
                    locations.append(Vector3(x, y, z))
                except (ValueError, IndexError):
                    pass
        
        return locations
    
    def _parse_area(self, command: str) -> Optional[Tuple[Vector3, Vector3]]:
        """Parse target area from command."""
        for pattern in self._area_patterns:
            match = re.search(pattern, command)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 4:
                        # from-to format
                        min_point = Vector3(float(groups[0]), float(groups[1]), 0)
                        max_point = Vector3(float(groups[2]), float(groups[3]), 10)
                        return (min_point, max_point)
                    elif len(groups) == 2:
                        # dimension format (e.g., "10m by 20m")
                        width = float(groups[0])
                        height = float(groups[1])
                        min_point = Vector3(0, 0, 0)
                        max_point = Vector3(width, height, 10)
                        return (min_point, max_point)
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def _parse_deadline(self, command: str) -> Optional[float]:
        """Parse deadline from command."""
        for pattern in self._time_patterns:
            match = re.search(pattern, command)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    if "minute" in unit:
                        return value * 60
                    elif "hour" in unit:
                        return value * 3600
                    else:
                        return value
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def _parse_priority(self, command: str) -> TaskPriority:
        """Parse priority from command."""
        if any(word in command for word in ["urgent", "emergency", "critical", "asap"]):
            return TaskPriority.CRITICAL
        elif any(word in command for word in ["important", "high priority"]):
            return TaskPriority.HIGH
        elif any(word in command for word in ["low priority", "when possible"]):
            return TaskPriority.LOW
        return TaskPriority.MEDIUM
    
    def _parse_platforms(self, command: str) -> List[PlatformType]:
        """Parse platform requirements from command."""
        platforms = []
        
        for keyword, platform_type in self._platform_keywords.items():
            if keyword in command:
                if platform_type not in platforms:
                    platforms.append(platform_type)
        
        return platforms
    
    def _parse_agent_count(self, command: str) -> Tuple[int, Optional[int]]:
        """Parse agent count requirements."""
        min_agents = 1
        max_agents = None
        
        # Pattern: "use 3 drones" or "with 5 robots"
        count_pattern = r"(?:use|with|using)\s+(\d+)\s+(?:drone|uav|robot|agent)"
        match = re.search(count_pattern, command)
        if match:
            count = int(match.group(1))
            min_agents = count
            max_agents = count
        
        # Pattern: "at least 3" or "minimum 3"
        min_pattern = r"(?:at\s+least|minimum|min)\s+(\d+)"
        match = re.search(min_pattern, command)
        if match:
            min_agents = int(match.group(1))
        
        # Pattern: "at most 5" or "maximum 5"
        max_pattern = r"(?:at\s+most|maximum|max)\s+(\d+)"
        match = re.search(max_pattern, command)
        if match:
            max_agents = int(match.group(1))
        
        return min_agents, max_agents
    
    def _generate_tasks(self, intent: MissionIntent) -> List[Task]:
        """Generate tasks from parsed intent."""
        tasks = []
        
        # Determine task type based on intent
        task_type_map = {
            MissionIntentType.SURVEY: TaskType.SURVEY,
            MissionIntentType.SEARCH: TaskType.SEARCH,
            MissionIntentType.DELIVERY: TaskType.DELIVERY,
            MissionIntentType.PATROL: TaskType.PATROL,
            MissionIntentType.INSPECTION: TaskType.INSPECT,
            MissionIntentType.MAPPING: TaskType.SURVEY,
            MissionIntentType.MONITORING: TaskType.MONITOR,
        }
        
        task_type = task_type_map.get(intent.intent_type, TaskType.NAVIGATE)
        
        # Generate tasks for target points
        for i, point in enumerate(intent.target_points):
            task = Task(
                task_id=f"task_{intent.intent_type.name.lower()}_{i}",
                task_type=task_type,
                priority=intent.priority,
                location=point,
                deadline=intent.deadline,
            )
            tasks.append(task)
        
        # Generate area coverage tasks
        if intent.target_area and not intent.target_points:
            min_p, max_p = intent.target_area
            
            # Create grid of waypoints
            grid_spacing = 5.0
            x_points = int((max_p.x - min_p.x) / grid_spacing) + 1
            y_points = int((max_p.y - min_p.y) / grid_spacing) + 1
            
            for i in range(x_points):
                for j in range(y_points):
                    x = min_p.x + i * grid_spacing
                    y = min_p.y + j * grid_spacing
                    z = (min_p.z + max_p.z) / 2
                    
                    task = Task(
                        task_id=f"task_{intent.intent_type.name.lower()}_{i}_{j}",
                        task_type=task_type,
                        priority=intent.priority,
                        location=Vector3(x, y, z),
                        deadline=intent.deadline,
                    )
                    tasks.append(task)
        
        return tasks
    
    def _generate_explanation(self, intent: MissionIntent, tasks: List[Task]) -> str:
        """Generate human-readable explanation."""
        parts = [f"Interpreted as {intent.intent_type.name} mission"]
        
        if tasks:
            parts.append(f"with {len(tasks)} tasks")
        
        if intent.target_area:
            min_p, max_p = intent.target_area
            parts.append(f"covering area ({min_p.x}, {min_p.y}) to ({max_p.x}, {max_p.y})")
        
        if intent.deadline:
            parts.append(f"deadline: {intent.deadline}s")
        
        if intent.required_platforms:
            platforms = [p.name for p in intent.required_platforms]
            parts.append(f"using: {', '.join(platforms)}")
        
        return ", ".join(parts)
