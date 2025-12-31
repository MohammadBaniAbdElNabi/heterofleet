"""
Communication Protocol for HeteroFleet.

Defines message formats, encoding, and protocol handling.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class MessageType(IntEnum):
    """Types of protocol messages."""
    # Control
    HEARTBEAT = 0x01
    ACK = 0x02
    NACK = 0x03
    
    # State
    STATE_UPDATE = 0x10
    STATE_REQUEST = 0x11
    POSITION_UPDATE = 0x12
    VELOCITY_UPDATE = 0x13
    
    # Coordination
    TASK_ASSIGNMENT = 0x20
    TASK_STATUS = 0x21
    TASK_COMPLETE = 0x22
    
    # Formation
    FORMATION_JOIN = 0x30
    FORMATION_LEAVE = 0x31
    FORMATION_UPDATE = 0x32
    
    # Safety
    EMERGENCY_STOP = 0x40
    COLLISION_WARNING = 0x41
    SAFETY_OVERRIDE = 0x42
    
    # Discovery
    ANNOUNCE = 0x50
    QUERY = 0x51
    RESPONSE = 0x52
    
    # Data
    SENSOR_DATA = 0x60
    TELEMETRY = 0x61
    LOG = 0x62
    
    # Custom
    CUSTOM = 0xFF


class MessagePriority(IntEnum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MessageHeader:
    """Protocol message header."""
    
    version: int = 1
    message_type: MessageType = MessageType.HEARTBEAT
    priority: MessagePriority = MessagePriority.NORMAL
    sequence_num: int = 0
    
    source_id: str = ""
    dest_id: str = ""  # Empty = broadcast
    
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live (hops)
    
    # Flags
    requires_ack: bool = False
    is_fragment: bool = False
    fragment_id: int = 0
    fragment_total: int = 1
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        # Pack header fields
        flags = (
            (self.requires_ack << 0) |
            (self.is_fragment << 1)
        )
        
        source_bytes = self.source_id.encode('utf-8')[:32].ljust(32, b'\x00')
        dest_bytes = self.dest_id.encode('utf-8')[:32].ljust(32, b'\x00')
        
        return struct.pack(
            '!BBBBII32s32sQBBB',
            self.version,
            self.message_type,
            self.priority,
            self.ttl,
            self.sequence_num,
            flags,
            source_bytes,
            dest_bytes,
            int(self.timestamp * 1000000),  # Microseconds
            self.fragment_id,
            self.fragment_total,
            0  # Reserved
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MessageHeader:
        """Deserialize header from bytes."""
        unpacked = struct.unpack('!BBBBII32s32sQBBB', data[:84])
        
        return cls(
            version=unpacked[0],
            message_type=MessageType(unpacked[1]),
            priority=MessagePriority(unpacked[2]),
            ttl=unpacked[3],
            sequence_num=unpacked[4],
            requires_ack=bool(unpacked[5] & 0x01),
            is_fragment=bool(unpacked[5] & 0x02),
            source_id=unpacked[6].rstrip(b'\x00').decode('utf-8'),
            dest_id=unpacked[7].rstrip(b'\x00').decode('utf-8'),
            timestamp=unpacked[8] / 1000000.0,
            fragment_id=unpacked[9],
            fragment_total=unpacked[10],
        )
    
    @staticmethod
    def header_size() -> int:
        """Get header size in bytes."""
        return 84


@dataclass
class ProtocolMessage:
    """Complete protocol message."""
    
    header: MessageHeader = field(default_factory=MessageHeader)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    checksum: str = ""
    raw_bytes: bytes = field(default=b"", repr=False)
    
    def compute_checksum(self) -> str:
        """Compute message checksum."""
        content = json.dumps(self.payload, sort_keys=True).encode('utf-8')
        return hashlib.md5(content).hexdigest()[:8]
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        header_bytes = self.header.to_bytes()
        payload_bytes = json.dumps(self.payload).encode('utf-8')
        
        # Length prefix for payload
        length = len(payload_bytes)
        
        # Checksum
        checksum = self.compute_checksum()
        checksum_bytes = checksum.encode('utf-8')
        
        return header_bytes + struct.pack('!I', length) + payload_bytes + checksum_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> ProtocolMessage:
        """Deserialize message from bytes."""
        header = MessageHeader.from_bytes(data[:84])
        
        length = struct.unpack('!I', data[84:88])[0]
        payload_bytes = data[88:88+length]
        checksum = data[88+length:88+length+8].decode('utf-8')
        
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        msg = cls(header=header, payload=payload, checksum=checksum)
        msg.raw_bytes = data
        
        return msg
    
    def validate(self) -> bool:
        """Validate message integrity."""
        return self.compute_checksum() == self.checksum
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        source_id: str,
        dest_id: str = "",
        payload: Dict[str, Any] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_ack: bool = False
    ) -> ProtocolMessage:
        """Create a new protocol message."""
        header = MessageHeader(
            message_type=message_type,
            priority=priority,
            source_id=source_id,
            dest_id=dest_id,
            requires_ack=requires_ack,
        )
        
        msg = cls(header=header, payload=payload or {})
        msg.checksum = msg.compute_checksum()
        
        return msg


class MessageProtocol:
    """
    Message protocol handler.
    
    Manages message creation, encoding, and validation.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize protocol handler.
        
        Args:
            agent_id: Local agent identifier
        """
        self.agent_id = agent_id
        self._sequence_num = 0
        
        # Fragment assembly
        self._fragment_buffers: Dict[str, Dict[int, bytes]] = {}
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "checksum_failures": 0,
        }
    
    def create_heartbeat(self) -> ProtocolMessage:
        """Create heartbeat message."""
        return self._create_message(
            MessageType.HEARTBEAT,
            {"status": "alive", "timestamp": time.time()}
        )
    
    def create_state_update(
        self,
        position: List[float],
        velocity: List[float],
        battery: float,
        mode: str
    ) -> ProtocolMessage:
        """Create state update message."""
        return self._create_message(
            MessageType.STATE_UPDATE,
            {
                "position": position,
                "velocity": velocity,
                "battery": battery,
                "mode": mode,
            },
            priority=MessagePriority.HIGH
        )
    
    def create_task_assignment(
        self,
        dest_id: str,
        task_id: str,
        task_type: str,
        location: List[float],
        priority: int = 1
    ) -> ProtocolMessage:
        """Create task assignment message."""
        return self._create_message(
            MessageType.TASK_ASSIGNMENT,
            {
                "task_id": task_id,
                "task_type": task_type,
                "location": location,
                "priority": priority,
            },
            dest_id=dest_id,
            requires_ack=True
        )
    
    def create_emergency_stop(self, reason: str = "") -> ProtocolMessage:
        """Create emergency stop message."""
        return self._create_message(
            MessageType.EMERGENCY_STOP,
            {"reason": reason, "timestamp": time.time()},
            priority=MessagePriority.CRITICAL
        )
    
    def create_collision_warning(
        self,
        other_id: str,
        distance: float,
        time_to_collision: float
    ) -> ProtocolMessage:
        """Create collision warning message."""
        return self._create_message(
            MessageType.COLLISION_WARNING,
            {
                "other_agent": other_id,
                "distance": distance,
                "ttc": time_to_collision,
            },
            dest_id=other_id,
            priority=MessagePriority.CRITICAL
        )
    
    def create_ack(self, original_msg: ProtocolMessage) -> ProtocolMessage:
        """Create acknowledgment message."""
        return self._create_message(
            MessageType.ACK,
            {
                "ack_seq": original_msg.header.sequence_num,
                "ack_type": original_msg.header.message_type,
            },
            dest_id=original_msg.header.source_id
        )
    
    def _create_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        dest_id: str = "",
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_ack: bool = False
    ) -> ProtocolMessage:
        """Create a protocol message."""
        self._sequence_num += 1
        
        header = MessageHeader(
            message_type=message_type,
            priority=priority,
            sequence_num=self._sequence_num,
            source_id=self.agent_id,
            dest_id=dest_id,
            requires_ack=requires_ack,
        )
        
        msg = ProtocolMessage(header=header, payload=payload)
        msg.checksum = msg.compute_checksum()
        
        return msg
    
    def encode(self, msg: ProtocolMessage) -> bytes:
        """Encode message to bytes."""
        data = msg.to_bytes()
        self._stats["messages_sent"] += 1
        self._stats["bytes_sent"] += len(data)
        return data
    
    def decode(self, data: bytes) -> Optional[ProtocolMessage]:
        """Decode message from bytes."""
        try:
            msg = ProtocolMessage.from_bytes(data)
            
            if not msg.validate():
                self._stats["checksum_failures"] += 1
                logger.warning(f"Checksum failure for message from {msg.header.source_id}")
                return None
            
            self._stats["messages_received"] += 1
            self._stats["bytes_received"] += len(data)
            
            return msg
            
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            return None
    
    def should_process(self, msg: ProtocolMessage) -> bool:
        """Check if message should be processed by this agent."""
        dest = msg.header.dest_id
        return dest == "" or dest == self.agent_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return self._stats.copy()
