"""
Hardware Interfaces for HeteroFleet.

Provides interfaces to physical hardware platforms:
- Crazyflie micro drones
- MentorPi ground vehicles

Based on HeteroFleet Architecture v1.0
"""

from heterofleet.hardware.crazyflie import (
    CrazyflieInterface,
    CrazyflieSwarm,
    CrazyflieConfig,
)
from heterofleet.hardware.mentorpi import (
    MentorPiInterface,
    MentorPiConfig,
)
from heterofleet.hardware.base import (
    HardwareInterface,
    HardwareStatus,
    ConnectionState,
)

__all__ = [
    "CrazyflieInterface", "CrazyflieSwarm", "CrazyflieConfig",
    "MentorPiInterface", "MentorPiConfig",
    "HardwareInterface", "HardwareStatus", "ConnectionState",
]
