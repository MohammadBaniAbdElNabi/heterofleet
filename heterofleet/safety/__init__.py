"""
Safety module for HeteroFleet.

Implements Compositional Safety Verification (CSV) framework:
- Distributed QP-based safety filter
- STL (Signal Temporal Logic) monitoring
- Safety certificates
- Collision avoidance verification

Based on HeteroFleet Architecture v1.0
"""

from heterofleet.safety.distributed_filter import (
    DistributedSafetyFilter,
    SafetyFilterResult,
    QPSolver,
)
from heterofleet.safety.stl_monitor import (
    STLMonitor,
    STLFormula,
    STLPredicate,
    STLAlways,
    STLEventually,
    RobustnessResult,
)
from heterofleet.safety.certificates import (
    SafetyCertificate,
    CertificateType,
    CertificateManager,
)
from heterofleet.safety.collision import (
    CollisionChecker,
    CollisionPrediction,
    CollisionAvoidance,
)

__all__ = [
    # Distributed filter
    "DistributedSafetyFilter",
    "SafetyFilterResult",
    "QPSolver",
    # STL monitoring
    "STLMonitor",
    "STLFormula",
    "STLPredicate",
    "STLAlways",
    "STLEventually",
    "RobustnessResult",
    # Certificates
    "SafetyCertificate",
    "CertificateType",
    "CertificateManager",
    # Collision
    "CollisionChecker",
    "CollisionPrediction",
    "CollisionAvoidance",
]
