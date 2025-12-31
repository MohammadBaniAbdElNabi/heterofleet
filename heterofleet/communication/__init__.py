"""
Communication module for HeteroFleet.

Implements communication protocols, routing, and
message handling for multi-agent coordination.

Based on HeteroFleet Architecture v1.0
"""

from heterofleet.communication.protocol import (
    MessageProtocol,
    MessageType,
    ProtocolMessage,
)
from heterofleet.communication.routing import (
    AdaptiveRouter,
    RoutingTable,
    LinkMetrics,
    MeshNetwork,
)

__all__ = [
    "MessageProtocol",
    "MessageType",
    "ProtocolMessage",
    "AdaptiveRouter",
    "RoutingTable",
    "LinkMetrics",
    "MeshNetwork",
]
