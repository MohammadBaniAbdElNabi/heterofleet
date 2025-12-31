"""
Digital Twin module for HeteroFleet.

Implements Hierarchical Digital Twin Architecture (HDTA):
- Agent-level twins (individual robot state)
- Fleet-level twins (collective behavior)
- Mission-level twins (operational objectives)
- Synchronization and prediction

Based on HeteroFleet Architecture v1.0
"""

from heterofleet.digital_twin.agent_twin import (
    AgentTwin,
    AgentTwinState,
    StatePredictor,
)
from heterofleet.digital_twin.fleet_twin import (
    FleetTwin,
    FleetMetrics,
    FormationTracker,
)
from heterofleet.digital_twin.mission_twin import (
    MissionTwin,
    MissionState,
    MissionMetrics,
)
from heterofleet.digital_twin.synchronizer import (
    TwinSynchronizer,
    SyncEvent,
    SyncStatus,
)

__all__ = [
    "AgentTwin", "AgentTwinState", "StatePredictor",
    "FleetTwin", "FleetMetrics", "FormationTracker",
    "MissionTwin", "MissionState", "MissionMetrics",
    "TwinSynchronizer", "SyncEvent", "SyncStatus",
]
