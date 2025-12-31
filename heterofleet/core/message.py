"""
Message definitions and routing for heterogeneous fleet communication.

This module defines:
- Message types for inter-agent communication
- Compact binary encoding for efficient transmission
- Message routing and priority handling

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import struct
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3


class MessageType(IntEnum):
    """Message type identifiers for protocol."""
    STATE_BROADCAST = 0x01
    TASK_ANNOUNCE = 0x02
    TASK_ACCEPT = 0x03
    TASK_COMPLETE = 0x04
    SAFETY_ALERT = 0x10
    COLLISION_WARNING = 0x11
    EMERGENCY_OVERRIDE = 0x20
    EMERGENCY_STOP = 0x21
    HEARTBEAT = 0x30
    SYNC_REQUEST = 0x31
    SYNC_RESPONSE = 0x32
    NETWORK_QUALITY = 0x40
    FORMATION_UPDATE = 0x50
    TRAJECTORY_SHARE = 0x60


class MessagePriority(IntEnum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class MessageHeader:
    """
    Common header for all messages.
    
    Binary format: 16 bytes total
    - message_type: uint8 (1 byte)
    - priority: uint8 (1 byte)
    - sender_id: uint16 (2 bytes)
    - sequence_number: uint32 (4 bytes)
    - timestamp_ms: uint64 (8 bytes)
    """
    message_type: MessageType
    priority: MessagePriority
    sender_id: int
    sequence_number: int
    timestamp_ms: int
    
    HEADER_FORMAT = "!BBHIq"  # Network byte order
    HEADER_SIZE = 16
    
    def pack(self) -> bytes:
        """Pack header into binary format."""
        return struct.pack(
            self.HEADER_FORMAT,
            self.message_type,
            self.priority,
            self.sender_id,
            self.sequence_number,
            self.timestamp_ms
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> MessageHeader:
        """Unpack header from binary format."""
        values = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])
        return cls(
            message_type=MessageType(values[0]),
            priority=MessagePriority(values[1]),
            sender_id=values[2],
            sequence_number=values[3],
            timestamp_ms=values[4]
        )
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        sender_id: int,
        sequence_number: int,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> MessageHeader:
        """Create a new header with current timestamp."""
        return cls(
            message_type=message_type,
            priority=priority,
            sender_id=sender_id,
            sequence_number=sequence_number,
            timestamp_ms=int(time.time() * 1000)
        )


@dataclass
class Message:
    """
    Generic message for routing and communication.
    
    Simple message format for higher-level protocols.
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time-to-live (hops)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.name,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "payload": self.payload,
            "priority": self.priority.name,
            "timestamp": self.timestamp,
        }


# Alias for compatibility
FleetMessage = Message


@dataclass
class StateBroadcastMessage:
    """
    Message format for inter-agent state broadcast.
    
    Compact binary encoding for efficiency (56 bytes total).
    
    Format:
    - Header: 16 bytes
    - Platform type: uint8 (1 byte)
    - Position (x,y,z): int32 x 3 (12 bytes) - millimeter precision
    - Velocity (vx,vy,vz): int16 x 3 (6 bytes) - cm/s precision
    - Target position: int32 x 3 (12 bytes) - millimeter precision
    - Priority: uint8 (1 byte)
    - Battery percent: uint8 (1 byte)
    - Network quality: uint8 (1 byte)
    - Status flags: uint8 (1 byte)
    - Safety envelope (a,b,c): uint16 x 3 (6 bytes) - cm precision
    """
    
    header: MessageHeader
    platform_type: PlatformType
    position: Vector3  # meters
    velocity: Vector3  # m/s
    target_position: Vector3  # meters
    priority: float  # 0-1
    battery_percent: float  # 0-1
    network_quality: float  # 0-1
    status_flags: int  # Bitfield
    safety_envelope: Tuple[float, float, float]  # meters
    
    PAYLOAD_FORMAT = "!B3i3h3iBBBB3H"  # After header
    PAYLOAD_SIZE = 40
    TOTAL_SIZE = MessageHeader.HEADER_SIZE + PAYLOAD_SIZE
    
    # Status flag bits
    FLAG_ACTIVE = 0x01
    FLAG_EMERGENCY = 0x02
    FLAG_LOW_BATTERY = 0x04
    FLAG_COMM_DEGRADED = 0x08
    FLAG_TASK_ACTIVE = 0x10
    FLAG_FORMATION = 0x20
    
    def pack(self) -> bytes:
        """Pack message into binary format."""
        # Convert to packed units
        pos_mm = (
            int(self.position.x * 1000),
            int(self.position.y * 1000),
            int(self.position.z * 1000)
        )
        vel_cms = (
            int(self.velocity.x * 100),
            int(self.velocity.y * 100),
            int(self.velocity.z * 100)
        )
        target_mm = (
            int(self.target_position.x * 1000),
            int(self.target_position.y * 1000),
            int(self.target_position.z * 1000)
        )
        envelope_cm = (
            int(self.safety_envelope[0] * 100),
            int(self.safety_envelope[1] * 100),
            int(self.safety_envelope[2] * 100)
        )
        
        # Get platform type value
        ptype = self.platform_type.value if hasattr(self.platform_type, 'value') else 0
        ptype_int = list(PlatformType).index(self.platform_type) if isinstance(self.platform_type, PlatformType) else 0
        
        payload = struct.pack(
            self.PAYLOAD_FORMAT,
            ptype_int,
            *pos_mm,
            *vel_cms,
            *target_mm,
            int(self.priority * 255),
            int(self.battery_percent * 100),
            int(self.network_quality * 100),
            self.status_flags,
            *envelope_cm
        )
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> StateBroadcastMessage:
        """Unpack message from binary format."""
        header = MessageHeader.unpack(data)
        
        values = struct.unpack(
            cls.PAYLOAD_FORMAT,
            data[MessageHeader.HEADER_SIZE:cls.TOTAL_SIZE]
        )
        
        ptype_list = list(PlatformType)
        ptype_idx = values[0]
        platform_type = ptype_list[ptype_idx] if ptype_idx < len(ptype_list) else PlatformType.MICRO_UAV
        
        return cls(
            header=header,
            platform_type=platform_type,
            position=Vector3(values[1]/1000, values[2]/1000, values[3]/1000),
            velocity=Vector3(values[4]/100, values[5]/100, values[6]/100),
            target_position=Vector3(values[7]/1000, values[8]/1000, values[9]/1000),
            priority=values[10] / 255,
            battery_percent=values[11] / 100,
            network_quality=values[12] / 100,
            status_flags=values[13],
            safety_envelope=(values[14]/100, values[15]/100, values[16]/100)
        )
    
    @classmethod
    def create(
        cls,
        sender_id: int,
        sequence_number: int,
        platform_type: PlatformType,
        position: Vector3,
        velocity: Vector3,
        target_position: Vector3,
        priority: float = 0.5,
        battery_percent: float = 1.0,
        network_quality: float = 1.0,
        safety_envelope: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        is_active: bool = True,
        is_emergency: bool = False,
        has_task: bool = False
    ) -> StateBroadcastMessage:
        """Create a new state broadcast message."""
        
        # Build status flags
        flags = 0
        if is_active:
            flags |= cls.FLAG_ACTIVE
        if is_emergency:
            flags |= cls.FLAG_EMERGENCY
        if battery_percent < 0.2:
            flags |= cls.FLAG_LOW_BATTERY
        if network_quality < 0.5:
            flags |= cls.FLAG_COMM_DEGRADED
        if has_task:
            flags |= cls.FLAG_TASK_ACTIVE
        
        header = MessageHeader.create(
            message_type=MessageType.STATE_BROADCAST,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.NORMAL
        )
        
        return cls(
            header=header,
            platform_type=platform_type,
            position=position,
            velocity=velocity,
            target_position=target_position,
            priority=priority,
            battery_percent=battery_percent,
            network_quality=network_quality,
            status_flags=flags,
            safety_envelope=safety_envelope
        )
    
    def is_flag_set(self, flag: int) -> bool:
        """Check if a status flag is set."""
        return bool(self.status_flags & flag)
    
    @property
    def is_active(self) -> bool:
        return self.is_flag_set(self.FLAG_ACTIVE)
    
    @property
    def is_emergency(self) -> bool:
        return self.is_flag_set(self.FLAG_EMERGENCY)
    
    @property
    def has_low_battery(self) -> bool:
        return self.is_flag_set(self.FLAG_LOW_BATTERY)


@dataclass
class TaskAnnounceMessage:
    """
    Message for announcing a new task.
    
    Used by the task allocator to announce tasks to the fleet.
    """
    
    header: MessageHeader
    task_id: str
    task_type: str
    location: Vector3
    deadline: float  # Unix timestamp
    priority: float  # 0-1
    required_capabilities: List[str]
    payload_requirement: float  # kg
    estimated_duration: float  # seconds
    
    def pack(self) -> bytes:
        """Pack to binary format (simplified JSON-based for variable fields)."""
        import json
        payload = json.dumps({
            "task_id": self.task_id,
            "task_type": self.task_type,
            "location": [self.location.x, self.location.y, self.location.z],
            "deadline": self.deadline,
            "priority": self.priority,
            "required_capabilities": self.required_capabilities,
            "payload_requirement": self.payload_requirement,
            "estimated_duration": self.estimated_duration
        }).encode('utf-8')
        
        # Include payload length
        length_bytes = struct.pack("!I", len(payload))
        return self.header.pack() + length_bytes + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> TaskAnnounceMessage:
        """Unpack from binary format."""
        import json
        
        header = MessageHeader.unpack(data)
        length = struct.unpack("!I", data[MessageHeader.HEADER_SIZE:MessageHeader.HEADER_SIZE+4])[0]
        payload_data = data[MessageHeader.HEADER_SIZE+4:MessageHeader.HEADER_SIZE+4+length]
        payload = json.loads(payload_data.decode('utf-8'))
        
        return cls(
            header=header,
            task_id=payload["task_id"],
            task_type=payload["task_type"],
            location=Vector3(*payload["location"]),
            deadline=payload["deadline"],
            priority=payload["priority"],
            required_capabilities=payload["required_capabilities"],
            payload_requirement=payload["payload_requirement"],
            estimated_duration=payload["estimated_duration"]
        )
    
    @classmethod
    def create(
        cls,
        sender_id: int,
        sequence_number: int,
        task_id: str,
        task_type: str,
        location: Vector3,
        deadline: float,
        priority: float = 0.5,
        required_capabilities: List[str] = None,
        payload_requirement: float = 0.0,
        estimated_duration: float = 60.0
    ) -> TaskAnnounceMessage:
        """Create a new task announcement."""
        header = MessageHeader.create(
            message_type=MessageType.TASK_ANNOUNCE,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.HIGH
        )
        
        return cls(
            header=header,
            task_id=task_id,
            task_type=task_type,
            location=location,
            deadline=deadline,
            priority=priority,
            required_capabilities=required_capabilities or [],
            payload_requirement=payload_requirement,
            estimated_duration=estimated_duration
        )


@dataclass
class SafetyAlertMessage:
    """
    Message for safety-related alerts.
    
    Used for collision warnings, boundary violations, etc.
    """
    
    header: MessageHeader
    alert_type: str  # "collision", "boundary", "energy", etc.
    severity: float  # 0-1
    involved_agents: List[int]  # Agent IDs involved
    location: Vector3
    description: str
    recommended_action: str
    time_to_impact: Optional[float] = None  # seconds, for collision warnings
    
    def pack(self) -> bytes:
        """Pack to binary format."""
        import json
        payload = json.dumps({
            "alert_type": self.alert_type,
            "severity": self.severity,
            "involved_agents": self.involved_agents,
            "location": [self.location.x, self.location.y, self.location.z],
            "description": self.description,
            "recommended_action": self.recommended_action,
            "time_to_impact": self.time_to_impact
        }).encode('utf-8')
        
        length_bytes = struct.pack("!I", len(payload))
        return self.header.pack() + length_bytes + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> SafetyAlertMessage:
        """Unpack from binary format."""
        import json
        
        header = MessageHeader.unpack(data)
        length = struct.unpack("!I", data[MessageHeader.HEADER_SIZE:MessageHeader.HEADER_SIZE+4])[0]
        payload_data = data[MessageHeader.HEADER_SIZE+4:MessageHeader.HEADER_SIZE+4+length]
        payload = json.loads(payload_data.decode('utf-8'))
        
        return cls(
            header=header,
            alert_type=payload["alert_type"],
            severity=payload["severity"],
            involved_agents=payload["involved_agents"],
            location=Vector3(*payload["location"]),
            description=payload["description"],
            recommended_action=payload["recommended_action"],
            time_to_impact=payload.get("time_to_impact")
        )
    
    @classmethod
    def create_collision_warning(
        cls,
        sender_id: int,
        sequence_number: int,
        agent1_id: int,
        agent2_id: int,
        collision_point: Vector3,
        time_to_impact: float,
        severity: float
    ) -> SafetyAlertMessage:
        """Create a collision warning alert."""
        header = MessageHeader.create(
            message_type=MessageType.COLLISION_WARNING,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.CRITICAL
        )
        
        return cls(
            header=header,
            alert_type="collision",
            severity=severity,
            involved_agents=[agent1_id, agent2_id],
            location=collision_point,
            description=f"Collision predicted between agents {agent1_id} and {agent2_id}",
            recommended_action="immediate_avoidance",
            time_to_impact=time_to_impact
        )


@dataclass
class EmergencyOverrideMessage:
    """
    Message for emergency override commands.
    
    Highest priority message that can override normal operation.
    """
    
    header: MessageHeader
    override_type: str  # "stop_all", "return_home", "land", "hover"
    target_agents: List[int]  # Empty list = all agents
    override_params: Dict[str, Any]
    reason: str
    duration: float  # How long override should last (0 = until cancelled)
    
    def pack(self) -> bytes:
        """Pack to binary format."""
        import json
        payload = json.dumps({
            "override_type": self.override_type,
            "target_agents": self.target_agents,
            "override_params": self.override_params,
            "reason": self.reason,
            "duration": self.duration
        }).encode('utf-8')
        
        length_bytes = struct.pack("!I", len(payload))
        return self.header.pack() + length_bytes + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> EmergencyOverrideMessage:
        """Unpack from binary format."""
        import json
        
        header = MessageHeader.unpack(data)
        length = struct.unpack("!I", data[MessageHeader.HEADER_SIZE:MessageHeader.HEADER_SIZE+4])[0]
        payload_data = data[MessageHeader.HEADER_SIZE+4:MessageHeader.HEADER_SIZE+4+length]
        payload = json.loads(payload_data.decode('utf-8'))
        
        return cls(
            header=header,
            override_type=payload["override_type"],
            target_agents=payload["target_agents"],
            override_params=payload["override_params"],
            reason=payload["reason"],
            duration=payload["duration"]
        )
    
    @classmethod
    def create_emergency_stop(
        cls,
        sender_id: int,
        sequence_number: int,
        reason: str,
        target_agents: List[int] = None
    ) -> EmergencyOverrideMessage:
        """Create an emergency stop command."""
        header = MessageHeader.create(
            message_type=MessageType.EMERGENCY_STOP,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.EMERGENCY
        )
        
        return cls(
            header=header,
            override_type="emergency_stop",
            target_agents=target_agents or [],
            override_params={"stop_immediately": True},
            reason=reason,
            duration=0  # Until cancelled
        )
    
    @classmethod
    def create_return_home(
        cls,
        sender_id: int,
        sequence_number: int,
        reason: str,
        target_agents: List[int] = None
    ) -> EmergencyOverrideMessage:
        """Create a return-to-home command."""
        header = MessageHeader.create(
            message_type=MessageType.EMERGENCY_OVERRIDE,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.CRITICAL
        )
        
        return cls(
            header=header,
            override_type="return_home",
            target_agents=target_agents or [],
            override_params={},
            reason=reason,
            duration=0
        )


@dataclass
class TrajectoryShareMessage:
    """
    Message for sharing planned trajectory with neighbors.
    
    Used for coordination and collision avoidance.
    """
    
    header: MessageHeader
    horizon: float  # seconds
    num_points: int
    timestamps: List[float]
    positions: List[Tuple[float, float, float]]
    velocities: List[Tuple[float, float, float]]
    
    def pack(self) -> bytes:
        """Pack to binary format."""
        # Fixed format: header + horizon + num_points + trajectory data
        base = struct.pack("!fI", self.horizon, self.num_points)
        
        traj_data = b""
        for t, pos, vel in zip(self.timestamps, self.positions, self.velocities):
            traj_data += struct.pack("!f3f3f", t, *pos, *vel)
        
        return self.header.pack() + base + traj_data
    
    @classmethod
    def unpack(cls, data: bytes) -> TrajectoryShareMessage:
        """Unpack from binary format."""
        header = MessageHeader.unpack(data)
        offset = MessageHeader.HEADER_SIZE
        
        horizon, num_points = struct.unpack("!fI", data[offset:offset+8])
        offset += 8
        
        timestamps = []
        positions = []
        velocities = []
        
        for _ in range(num_points):
            values = struct.unpack("!f3f3f", data[offset:offset+28])
            timestamps.append(values[0])
            positions.append((values[1], values[2], values[3]))
            velocities.append((values[4], values[5], values[6]))
            offset += 28
        
        return cls(
            header=header,
            horizon=horizon,
            num_points=num_points,
            timestamps=timestamps,
            positions=positions,
            velocities=velocities
        )
    
    @classmethod
    def create(
        cls,
        sender_id: int,
        sequence_number: int,
        trajectory: List[Tuple[float, Vector3, Vector3]]  # List of (time, pos, vel)
    ) -> TrajectoryShareMessage:
        """Create from trajectory list."""
        header = MessageHeader.create(
            message_type=MessageType.TRAJECTORY_SHARE,
            sender_id=sender_id,
            sequence_number=sequence_number,
            priority=MessagePriority.HIGH
        )
        
        if not trajectory:
            return cls(header=header, horizon=0, num_points=0, timestamps=[], positions=[], velocities=[])
        
        timestamps = [t[0] for t in trajectory]
        positions = [(t[1].x, t[1].y, t[1].z) for t in trajectory]
        velocities = [(t[2].x, t[2].y, t[2].z) for t in trajectory]
        
        return cls(
            header=header,
            horizon=timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            num_points=len(trajectory),
            timestamps=timestamps,
            positions=positions,
            velocities=velocities
        )


class MessageRouter:
    """
    Message routing and handling for the fleet communication system.
    
    Handles message prioritization, delivery, and callback management.
    """
    
    def __init__(self, agent_id: int):
        """
        Initialize router for an agent.
        
        Args:
            agent_id: ID of the owning agent
        """
        self.agent_id = agent_id
        self.sequence_number = 0
        
        # Message handlers by type
        self._handlers: Dict[MessageType, List[Callable]] = {
            mtype: [] for mtype in MessageType
        }
        
        # Priority queues for outgoing messages
        self._outgoing_queues: Dict[MessagePriority, List[bytes]] = {
            priority: [] for priority in MessagePriority
        }
        
        # Recent message cache (for deduplication)
        self._message_cache: Dict[str, float] = {}  # message_hash -> timestamp
        self._cache_ttl = 5.0  # seconds
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "duplicates_filtered": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
    
    def get_next_sequence(self) -> int:
        """Get next sequence number."""
        seq = self.sequence_number
        self.sequence_number = (self.sequence_number + 1) % (2**32)
        return seq
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable
    ) -> None:
        """
        Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function(message) to handle the message
        """
        self._handlers[message_type].append(handler)
    
    def unregister_handler(
        self,
        message_type: MessageType,
        handler: Callable
    ) -> bool:
        """
        Unregister a handler.
        
        Returns:
            True if handler was found and removed
        """
        try:
            self._handlers[message_type].remove(handler)
            return True
        except ValueError:
            return False
    
    def _compute_message_hash(self, data: bytes) -> str:
        """Compute hash for message deduplication."""
        return hashlib.md5(data).hexdigest()
    
    def _is_duplicate(self, data: bytes) -> bool:
        """Check if message is a duplicate."""
        msg_hash = self._compute_message_hash(data)
        current_time = time.time()
        
        # Clean old entries
        self._message_cache = {
            h: t for h, t in self._message_cache.items()
            if current_time - t < self._cache_ttl
        }
        
        if msg_hash in self._message_cache:
            return True
        
        self._message_cache[msg_hash] = current_time
        return False
    
    def receive_message(self, data: bytes) -> Optional[Any]:
        """
        Process a received message.
        
        Args:
            data: Raw message bytes
            
        Returns:
            Parsed message object if handled, None otherwise
        """
        # Check for duplicates
        if self._is_duplicate(data):
            self._stats["duplicates_filtered"] += 1
            return None
        
        self._stats["messages_received"] += 1
        self._stats["bytes_received"] += len(data)
        
        # Parse header
        try:
            header = MessageHeader.unpack(data)
        except Exception as e:
            logger.warning(f"Failed to parse message header: {e}")
            return None
        
        # Parse and handle based on type
        message = None
        try:
            if header.message_type == MessageType.STATE_BROADCAST:
                message = StateBroadcastMessage.unpack(data)
            elif header.message_type == MessageType.TASK_ANNOUNCE:
                message = TaskAnnounceMessage.unpack(data)
            elif header.message_type in (MessageType.SAFETY_ALERT, MessageType.COLLISION_WARNING):
                message = SafetyAlertMessage.unpack(data)
            elif header.message_type in (MessageType.EMERGENCY_OVERRIDE, MessageType.EMERGENCY_STOP):
                message = EmergencyOverrideMessage.unpack(data)
            elif header.message_type == MessageType.TRAJECTORY_SHARE:
                message = TrajectoryShareMessage.unpack(data)
            else:
                logger.debug(f"Unknown message type: {header.message_type}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse message body: {e}")
            return None
        
        # Call handlers
        for handler in self._handlers[header.message_type]:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error for {header.message_type}: {e}")
        
        return message
    
    def queue_message(
        self,
        message: Union[StateBroadcastMessage, TaskAnnounceMessage, 
                       SafetyAlertMessage, EmergencyOverrideMessage,
                       TrajectoryShareMessage]
    ) -> None:
        """
        Queue a message for sending.
        
        Args:
            message: Message to send
        """
        data = message.pack()
        priority = message.header.priority
        self._outgoing_queues[priority].append(data)
    
    def get_pending_messages(self) -> List[bytes]:
        """
        Get all pending messages in priority order.
        
        Returns:
            List of message bytes, highest priority first
        """
        messages = []
        
        # Process in priority order (highest first)
        for priority in sorted(MessagePriority, reverse=True):
            queue = self._outgoing_queues[priority]
            messages.extend(queue)
            self._stats["messages_sent"] += len(queue)
            self._stats["bytes_sent"] += sum(len(m) for m in queue)
            queue.clear()
        
        return messages
    
    def create_state_broadcast(
        self,
        platform_type: PlatformType,
        position: Vector3,
        velocity: Vector3,
        target_position: Vector3,
        **kwargs
    ) -> StateBroadcastMessage:
        """Create and return a state broadcast message."""
        return StateBroadcastMessage.create(
            sender_id=self.agent_id,
            sequence_number=self.get_next_sequence(),
            platform_type=platform_type,
            position=position,
            velocity=velocity,
            target_position=target_position,
            **kwargs
        )
    
    def create_trajectory_share(
        self,
        trajectory: List[Tuple[float, Vector3, Vector3]]
    ) -> TrajectoryShareMessage:
        """Create and return a trajectory share message."""
        return TrajectoryShareMessage.create(
            sender_id=self.agent_id,
            sequence_number=self.get_next_sequence(),
            trajectory=trajectory
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset router statistics."""
        for key in self._stats:
            self._stats[key] = 0


class MulticastGroup:
    """
    Multicast group for efficient message distribution.
    
    Groups agents for efficient broadcast (e.g., by region, task, formation).
    """
    
    def __init__(self, group_id: str, group_type: str = "region"):
        """
        Initialize multicast group.
        
        Args:
            group_id: Unique identifier for the group
            group_type: Type of group ("region", "task", "formation")
        """
        self.group_id = group_id
        self.group_type = group_type
        self.members: set[int] = set()
        self.creation_time = time.time()
    
    def add_member(self, agent_id: int) -> None:
        """Add an agent to the group."""
        self.members.add(agent_id)
    
    def remove_member(self, agent_id: int) -> bool:
        """Remove an agent from the group."""
        try:
            self.members.remove(agent_id)
            return True
        except KeyError:
            return False
    
    def is_member(self, agent_id: int) -> bool:
        """Check if an agent is a member."""
        return agent_id in self.members
    
    @property
    def size(self) -> int:
        """Get number of members."""
        return len(self.members)


class MessageBus:
    """
    Central message bus for coordinating communication.
    
    Manages multicast groups and message routing across the fleet.
    """
    
    def __init__(self):
        """Initialize message bus."""
        self.groups: Dict[str, MulticastGroup] = {}
        self.agent_routers: Dict[int, MessageRouter] = {}
        
        # Global handlers (receive all messages)
        self.global_handlers: List[Callable] = []
    
    def register_agent(self, agent_id: int) -> MessageRouter:
        """
        Register an agent and create its router.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            MessageRouter for the agent
        """
        router = MessageRouter(agent_id)
        self.agent_routers[agent_id] = router
        return router
    
    def unregister_agent(self, agent_id: int) -> None:
        """Unregister an agent."""
        self.agent_routers.pop(agent_id, None)
        
        # Remove from all groups
        for group in self.groups.values():
            group.remove_member(agent_id)
    
    def create_group(self, group_id: str, group_type: str = "region") -> MulticastGroup:
        """Create a new multicast group."""
        group = MulticastGroup(group_id, group_type)
        self.groups[group_id] = group
        return group
    
    def get_group(self, group_id: str) -> Optional[MulticastGroup]:
        """Get a multicast group."""
        return self.groups.get(group_id)
    
    def broadcast(self, message: bytes, exclude: Optional[set[int]] = None) -> int:
        """
        Broadcast message to all agents.
        
        Args:
            message: Message bytes
            exclude: Set of agent IDs to exclude
            
        Returns:
            Number of agents message was sent to
        """
        exclude = exclude or set()
        count = 0
        
        for agent_id, router in self.agent_routers.items():
            if agent_id not in exclude:
                router.receive_message(message)
                count += 1
        
        # Call global handlers
        for handler in self.global_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Global handler error: {e}")
        
        return count
    
    def multicast(self, group_id: str, message: bytes, exclude: Optional[set[int]] = None) -> int:
        """
        Send message to a multicast group.
        
        Args:
            group_id: Group to send to
            message: Message bytes
            exclude: Set of agent IDs to exclude
            
        Returns:
            Number of agents message was sent to
        """
        group = self.groups.get(group_id)
        if group is None:
            logger.warning(f"Unknown multicast group: {group_id}")
            return 0
        
        exclude = exclude or set()
        count = 0
        
        for agent_id in group.members:
            if agent_id not in exclude and agent_id in self.agent_routers:
                self.agent_routers[agent_id].receive_message(message)
                count += 1
        
        return count
    
    def unicast(self, target_id: int, message: bytes) -> bool:
        """
        Send message to a specific agent.
        
        Args:
            target_id: Target agent ID
            message: Message bytes
            
        Returns:
            True if message was delivered
        """
        router = self.agent_routers.get(target_id)
        if router is None:
            return False
        
        router.receive_message(message)
        return True
    
    def register_global_handler(self, handler: Callable) -> None:
        """Register a handler that receives all messages."""
        self.global_handlers.append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all routers."""
        total_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "duplicates_filtered": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "num_agents": len(self.agent_routers),
            "num_groups": len(self.groups),
        }
        
        for router in self.agent_routers.values():
            stats = router.get_statistics()
            for key in ["messages_sent", "messages_received", "duplicates_filtered",
                       "bytes_sent", "bytes_received"]:
                total_stats[key] += stats[key]
        
        return total_stats
