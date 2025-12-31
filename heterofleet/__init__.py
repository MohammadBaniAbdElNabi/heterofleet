"""
HeteroFleet: A Digital Twin-Enabled Multi-Objective Coordination Framework
for Heterogeneous Autonomous Vehicle Swarms.

This framework provides:
- Heterogeneous Agent Interaction Model (HAIM)
- Multi-Objective Pareto-Optimal Task Allocation (MOPOTA)
- Hierarchical Digital Twin Architecture (HDTA)
- Adaptive Network-Aware Coordination (ANAC)
- Compositional Safety Verification (CSV)
- Language-Guided Mission Adaptation (LGMA)

Author: HeteroFleet Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "HeteroFleet Research Team"

from heterofleet.core.platform import PlatformType, PlatformSpecification
from heterofleet.core.agent import HeterogeneousAgent
from heterofleet.core.state import AgentState, FleetState
from heterofleet.core.message import StateBroadcastMessage, MessageType

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes
    "PlatformType",
    "PlatformSpecification",
    "HeterogeneousAgent",
    "AgentState",
    "FleetState",
    "StateBroadcastMessage",
    "MessageType",
]
