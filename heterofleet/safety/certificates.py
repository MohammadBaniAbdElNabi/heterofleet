"""
Safety Certificates for heterogeneous fleet operations.

Implements safety certificate generation and management
for compositional safety verification.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3


class CertificateType(Enum):
    """Types of safety certificates."""
    SEPARATION = auto()      # Minimum separation maintained
    GEOFENCE = auto()        # Within geofence bounds
    VELOCITY = auto()        # Within velocity limits
    ENERGY = auto()          # Sufficient energy for return
    COLLISION_FREE = auto()  # No collision predicted
    FORMATION = auto()       # Formation constraints met
    COMMUNICATION = auto()   # Communication maintained
    COMPOSITE = auto()       # Combination of certificates


class CertificateStatus(Enum):
    """Status of a certificate."""
    VALID = auto()           # Certificate is valid
    EXPIRED = auto()         # Certificate has expired
    REVOKED = auto()         # Certificate was revoked
    PENDING = auto()         # Certificate is pending validation


@dataclass
class SafetyCertificate:
    """
    Safety certificate for an agent or agent pair.
    
    A certificate attests that a safety property holds
    for a specified time period.
    """
    
    # Identity
    certificate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    certificate_type: CertificateType = CertificateType.SEPARATION
    
    # Scope
    agent_ids: List[str] = field(default_factory=list)
    
    # Validity
    issue_time: float = field(default_factory=time.time)
    expiry_time: float = 0.0
    status: CertificateStatus = CertificateStatus.VALID
    
    # Safety property
    property_name: str = ""
    property_value: float = 0.0  # e.g., minimum distance, max velocity
    threshold: float = 0.0       # Required threshold
    
    # Evidence
    robustness: float = 0.0      # STL robustness if applicable
    verification_method: str = ""
    evidence_hash: str = ""
    
    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        return (self.status == CertificateStatus.VALID and
                time.time() < self.expiry_time)
    
    @property
    def remaining_time(self) -> float:
        """Get remaining validity time in seconds."""
        return max(0.0, self.expiry_time - time.time())
    
    @property
    def margin(self) -> float:
        """Get margin above threshold."""
        return self.property_value - self.threshold
    
    def revoke(self, reason: str = "") -> None:
        """Revoke the certificate."""
        self.status = CertificateStatus.REVOKED
        logger.warning(f"Certificate {self.certificate_id} revoked: {reason}")
    
    def extend(self, additional_time: float) -> None:
        """Extend certificate validity."""
        if self.is_valid:
            self.expiry_time += additional_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "type": self.certificate_type.name,
            "agents": self.agent_ids,
            "issue_time": self.issue_time,
            "expiry_time": self.expiry_time,
            "status": self.status.name,
            "property_name": self.property_name,
            "property_value": self.property_value,
            "threshold": self.threshold,
            "margin": self.margin,
            "robustness": self.robustness,
            "is_valid": self.is_valid,
        }
    
    def compute_hash(self) -> str:
        """Compute hash of certificate data."""
        data = f"{self.certificate_type.name}:{':'.join(self.agent_ids)}:{self.property_name}:{self.property_value}:{self.threshold}:{self.issue_time}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class CertificateRequest:
    """Request for a safety certificate."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    certificate_type: CertificateType = CertificateType.SEPARATION
    agent_ids: List[str] = field(default_factory=list)
    property_name: str = ""
    required_threshold: float = 0.0
    validity_duration: float = 1.0  # seconds
    timestamp: float = field(default_factory=time.time)


class CertificateManager:
    """
    Manager for safety certificates.
    
    Handles certificate generation, validation, and lifecycle.
    """
    
    def __init__(
        self,
        default_validity: float = 1.0,  # seconds
        auto_revoke_on_violation: bool = True
    ):
        """
        Initialize certificate manager.
        
        Args:
            default_validity: Default certificate validity duration
            auto_revoke_on_violation: Whether to automatically revoke on violation
        """
        self.default_validity = default_validity
        self.auto_revoke = auto_revoke_on_violation
        
        # Certificate storage
        self._certificates: Dict[str, SafetyCertificate] = {}
        self._agent_certificates: Dict[str, List[str]] = {}  # agent_id -> cert_ids
        
        # Statistics
        self._stats = {
            "issued": 0,
            "revoked": 0,
            "expired": 0,
            "violations": 0,
        }
    
    def issue_certificate(
        self,
        certificate_type: CertificateType,
        agent_ids: List[str],
        property_name: str,
        property_value: float,
        threshold: float,
        robustness: float = 0.0,
        validity_duration: float = None,
        verification_method: str = "runtime",
        conditions: Dict[str, Any] = None
    ) -> Optional[SafetyCertificate]:
        """
        Issue a new safety certificate.
        
        Args:
            certificate_type: Type of certificate
            agent_ids: Agents covered by certificate
            property_name: Safety property name
            property_value: Current property value
            threshold: Required threshold
            robustness: STL robustness value
            validity_duration: How long certificate is valid
            verification_method: How safety was verified
            conditions: Conditions under which certificate is valid
            
        Returns:
            Issued certificate or None if property not satisfied
        """
        # Check if property meets threshold
        if property_value < threshold:
            logger.warning(f"Cannot issue certificate: {property_name} = {property_value} < {threshold}")
            return None
        
        validity = validity_duration or self.default_validity
        
        cert = SafetyCertificate(
            certificate_type=certificate_type,
            agent_ids=agent_ids,
            expiry_time=time.time() + validity,
            property_name=property_name,
            property_value=property_value,
            threshold=threshold,
            robustness=robustness,
            verification_method=verification_method,
            conditions=conditions or {},
        )
        
        cert.evidence_hash = cert.compute_hash()
        
        # Store certificate
        self._certificates[cert.certificate_id] = cert
        
        for agent_id in agent_ids:
            if agent_id not in self._agent_certificates:
                self._agent_certificates[agent_id] = []
            self._agent_certificates[agent_id].append(cert.certificate_id)
        
        self._stats["issued"] += 1
        
        return cert
    
    def get_certificate(self, certificate_id: str) -> Optional[SafetyCertificate]:
        """Get a certificate by ID."""
        return self._certificates.get(certificate_id)
    
    def get_agent_certificates(
        self,
        agent_id: str,
        valid_only: bool = True
    ) -> List[SafetyCertificate]:
        """Get all certificates for an agent."""
        cert_ids = self._agent_certificates.get(agent_id, [])
        certs = [self._certificates[cid] for cid in cert_ids if cid in self._certificates]
        
        if valid_only:
            certs = [c for c in certs if c.is_valid]
        
        return certs
    
    def get_certificates_by_type(
        self,
        cert_type: CertificateType,
        valid_only: bool = True
    ) -> List[SafetyCertificate]:
        """Get all certificates of a given type."""
        certs = [c for c in self._certificates.values() if c.certificate_type == cert_type]
        
        if valid_only:
            certs = [c for c in certs if c.is_valid]
        
        return certs
    
    def validate_certificate(self, certificate_id: str) -> Tuple[bool, str]:
        """
        Validate a certificate.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        cert = self._certificates.get(certificate_id)
        
        if cert is None:
            return False, "Certificate not found"
        
        if cert.status == CertificateStatus.REVOKED:
            return False, "Certificate revoked"
        
        if cert.status == CertificateStatus.EXPIRED or time.time() >= cert.expiry_time:
            cert.status = CertificateStatus.EXPIRED
            return False, "Certificate expired"
        
        # Verify hash integrity
        expected_hash = cert.compute_hash()
        if cert.evidence_hash != expected_hash:
            cert.revoke("Hash verification failed")
            return False, "Certificate integrity check failed"
        
        return True, "Valid"
    
    def revoke_certificate(
        self,
        certificate_id: str,
        reason: str = ""
    ) -> bool:
        """Revoke a certificate."""
        cert = self._certificates.get(certificate_id)
        
        if cert is None:
            return False
        
        cert.revoke(reason)
        self._stats["revoked"] += 1
        
        return True
    
    def check_violation(
        self,
        certificate_id: str,
        current_value: float
    ) -> bool:
        """
        Check if a certificate is violated.
        
        Args:
            certificate_id: Certificate to check
            current_value: Current property value
            
        Returns:
            True if violated
        """
        cert = self._certificates.get(certificate_id)
        
        if cert is None or not cert.is_valid:
            return False
        
        if current_value < cert.threshold:
            self._stats["violations"] += 1
            
            if self.auto_revoke:
                cert.revoke(f"Property violated: {current_value} < {cert.threshold}")
            
            return True
        
        return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired certificates."""
        current_time = time.time()
        expired = []
        
        for cert_id, cert in self._certificates.items():
            if cert.status == CertificateStatus.VALID and current_time >= cert.expiry_time:
                cert.status = CertificateStatus.EXPIRED
                expired.append(cert_id)
        
        self._stats["expired"] += len(expired)
        
        return len(expired)
    
    def get_fleet_safety_status(self) -> Dict[str, Any]:
        """Get overall fleet safety status based on certificates."""
        valid_certs = [c for c in self._certificates.values() if c.is_valid]
        
        # Group by type
        type_counts = {}
        for cert in valid_certs:
            type_name = cert.certificate_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Calculate minimum margins
        min_margins = {}
        for cert in valid_certs:
            type_name = cert.certificate_type.name
            if type_name not in min_margins:
                min_margins[type_name] = float('inf')
            min_margins[type_name] = min(min_margins[type_name], cert.margin)
        
        return {
            "total_valid_certificates": len(valid_certs),
            "certificates_by_type": type_counts,
            "minimum_margins": min_margins,
            "statistics": self._stats.copy(),
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get certificate statistics."""
        return self._stats.copy()


class CompositeCertificateBuilder:
    """
    Builder for composite safety certificates.
    
    Combines multiple certificates into a single composite certificate.
    """
    
    def __init__(self, name: str = "composite"):
        """Initialize builder."""
        self.name = name
        self._certificates: List[SafetyCertificate] = []
    
    def add(self, certificate: SafetyCertificate) -> CompositeCertificateBuilder:
        """Add a certificate to the composite."""
        self._certificates.append(certificate)
        return self
    
    def build(self, validity_duration: float = None) -> Optional[SafetyCertificate]:
        """
        Build composite certificate.
        
        The composite is valid only if all component certificates are valid.
        Validity is the minimum of all components.
        """
        if not self._certificates:
            return None
        
        # Check all components are valid
        for cert in self._certificates:
            if not cert.is_valid:
                return None
        
        # Collect all agents
        all_agents = set()
        for cert in self._certificates:
            all_agents.update(cert.agent_ids)
        
        # Minimum expiry
        min_expiry = min(cert.expiry_time for cert in self._certificates)
        
        # Minimum robustness
        min_robustness = min(cert.robustness for cert in self._certificates)
        
        # Build composite
        composite = SafetyCertificate(
            certificate_type=CertificateType.COMPOSITE,
            agent_ids=list(all_agents),
            expiry_time=validity_duration and (time.time() + validity_duration) or min_expiry,
            property_name=self.name,
            property_value=min_robustness,
            threshold=0.0,
            robustness=min_robustness,
            verification_method="composite",
            conditions={
                "component_certificates": [c.certificate_id for c in self._certificates],
                "num_components": len(self._certificates),
            }
        )
        
        composite.evidence_hash = composite.compute_hash()
        
        return composite
    
    def clear(self) -> None:
        """Clear builder state."""
        self._certificates = []


class SeparationCertificateIssuer:
    """
    Specialized issuer for separation certificates.
    
    Issues certificates attesting minimum separation between agents.
    """
    
    def __init__(
        self,
        certificate_manager: CertificateManager,
        default_validity: float = 0.5
    ):
        """Initialize issuer."""
        self.manager = certificate_manager
        self.default_validity = default_validity
    
    def issue_pairwise_certificate(
        self,
        agent_i: str,
        agent_j: str,
        distance: float,
        min_separation: float,
        validity: float = None
    ) -> Optional[SafetyCertificate]:
        """Issue separation certificate for agent pair."""
        return self.manager.issue_certificate(
            certificate_type=CertificateType.SEPARATION,
            agent_ids=[agent_i, agent_j],
            property_name="pairwise_separation",
            property_value=distance,
            threshold=min_separation,
            robustness=distance - min_separation,
            validity_duration=validity or self.default_validity,
            verification_method="distance_measurement",
        )
    
    def issue_fleet_separation_certificate(
        self,
        agent_ids: List[str],
        distances: Dict[Tuple[str, str], float],
        min_separation: float,
        validity: float = None
    ) -> Optional[SafetyCertificate]:
        """Issue fleet-wide separation certificate."""
        # Find minimum distance
        min_distance = float('inf')
        for (i, j), dist in distances.items():
            min_distance = min(min_distance, dist)
        
        if min_distance < min_separation:
            return None
        
        return self.manager.issue_certificate(
            certificate_type=CertificateType.SEPARATION,
            agent_ids=agent_ids,
            property_name="fleet_separation",
            property_value=min_distance,
            threshold=min_separation,
            robustness=min_distance - min_separation,
            validity_duration=validity or self.default_validity,
            verification_method="fleet_distance_check",
            conditions={"num_pairs": len(distances)},
        )
