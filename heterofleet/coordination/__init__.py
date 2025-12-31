"""
Coordination module for HeteroFleet.

Implements the Heterogeneous Agent Interaction Model (HAIM):
- Extended repulsion with platform-pair specific gains
- Cross-platform friction with adaptive parameters
- Network-energy-aware self-drive
- Priority hierarchy management

Also includes Adaptive Network-Aware Coordination (ANAC).
"""

from heterofleet.coordination.haim import (
    HAIMCoordinator,
    HAIMParameters,
    InteractionForce,
)
from heterofleet.coordination.repulsion import (
    RepulsionCalculator,
    RepulsionGainMatrix,
)
from heterofleet.coordination.friction import (
    FrictionCalculator,
    FrictionParameters,
)
from heterofleet.coordination.self_drive import (
    NetworkEnergyAwareSelfDrive,
    SelfDriveParameters,
    DangerCriteria,
    GeometricCriteria,
)
from heterofleet.coordination.priority import (
    PriorityManager,
    PriorityRule,
    PriorityResolution,
)

__all__ = [
    # HAIM
    "HAIMCoordinator",
    "HAIMParameters",
    "InteractionForce",
    # Repulsion
    "RepulsionCalculator",
    "RepulsionGainMatrix",
    # Friction
    "FrictionCalculator",
    "FrictionParameters",
    # Self-drive
    "NetworkEnergyAwareSelfDrive",
    "SelfDriveParameters",
    "DangerCriteria",
    "GeometricCriteria",
    # Priority
    "PriorityManager",
    "PriorityRule",
    "PriorityResolution",
]
