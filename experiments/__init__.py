"""
Experiments for HeteroFleet evaluation.

Implements five core experiments from the architecture:
1. Scalability Analysis
2. Formation Control Performance
3. Task Allocation Efficiency
4. Communication Resilience
5. Emergency Response

Based on HeteroFleet Architecture v1.0
"""

from experiments.scalability import ScalabilityExperiment
from experiments.formation import FormationExperiment
from experiments.task_allocation import TaskAllocationExperiment
from experiments.communication import CommunicationExperiment
from experiments.emergency import EmergencyExperiment

__all__ = [
    "ScalabilityExperiment",
    "FormationExperiment",
    "TaskAllocationExperiment",
    "CommunicationExperiment",
    "EmergencyExperiment",
]
