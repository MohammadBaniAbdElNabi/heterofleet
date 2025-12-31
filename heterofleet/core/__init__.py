"""
Core module for HeteroFleet.

Contains fundamental classes for agents, platforms, states, and messages.
"""

from heterofleet.core.platform import (
    PlatformType,
    PlatformSpecification,
    PhysicalProperties,
    DynamicProperties,
    DomainConstraints,
    CommunicationProperties,
    EnergyProperties,
    SensorProperties,
    CapabilityProperties,
    CollisionEnvelopeType,
    DomainType,
    PlatformRegistry,
    PlatformFactory,
)
from heterofleet.core.agent import HeterogeneousAgent, AgentMode
from heterofleet.core.state import (
    AgentState,
    FleetState,
    StateEstimate,
    HealthIndicators,
    BehavioralState,
    AnomalyInfo,
)
from heterofleet.core.message import (
    MessageType,
    StateBroadcastMessage,
    TaskAnnounceMessage,
    SafetyAlertMessage,
    EmergencyOverrideMessage,
    MessageRouter,
)

__all__ = [
    # Platform
    "PlatformType",
    "PlatformSpecification",
    "PhysicalProperties",
    "DynamicProperties",
    "DomainConstraints",
    "CommunicationProperties",
    "EnergyProperties",
    "SensorProperties",
    "CapabilityProperties",
    "CollisionEnvelopeType",
    "DomainType",
    "PlatformRegistry",
    "PlatformFactory",
    # Agent
    "HeterogeneousAgent",
    "AgentMode",
    # State
    "AgentState",
    "FleetState",
    "StateEstimate",
    "HealthIndicators",
    "BehavioralState",
    "AnomalyInfo",
    # Message
    "MessageType",
    "StateBroadcastMessage",
    "TaskAnnounceMessage",
    "SafetyAlertMessage",
    "EmergencyOverrideMessage",
    "MessageRouter",
]
