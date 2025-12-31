"""
Digital Twin Synchronization for HeteroFleet.

Handles synchronization between physical agents and digital twins,
including state updates, prediction alignment, and conflict resolution.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from loguru import logger

from heterofleet.core.platform import Vector3
from heterofleet.core.state import AgentState
from heterofleet.digital_twin.agent_twin import AgentTwin, AgentTwinState, TwinStatus
from heterofleet.digital_twin.fleet_twin import FleetTwin
from heterofleet.digital_twin.mission_twin import MissionTwin


class SyncStatus(Enum):
    """Synchronization status."""
    IDLE = auto()
    SYNCING = auto()
    SYNCHRONIZED = auto()
    ERROR = auto()


class SyncEventType(Enum):
    """Types of sync events."""
    STATE_UPDATE = auto()
    PREDICTION_DIVERGENCE = auto()
    CONNECTION_LOST = auto()
    CONNECTION_RESTORED = auto()
    CONFLICT_DETECTED = auto()
    SYNC_ERROR = auto()


@dataclass
class SyncEvent:
    """A synchronization event."""
    
    event_type: SyncEventType
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class SyncMetrics:
    """Metrics for synchronization performance."""
    
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    prediction_accuracy: float = 0.0
    divergence_count: int = 0
    
    last_sync_time: float = 0.0
    sync_rate_hz: float = 0.0


class TwinSynchronizer:
    """
    Synchronizer for digital twins.
    
    Manages state synchronization between physical agents
    and their digital twin representations.
    """
    
    def __init__(
        self,
        fleet_twin: FleetTwin,
        sync_rate: float = 10.0,
        prediction_threshold: float = 0.5,
        stale_threshold: float = 2.0
    ):
        """
        Initialize synchronizer.
        
        Args:
            fleet_twin: Fleet twin to synchronize
            sync_rate: Target synchronization rate (Hz)
            prediction_threshold: Max prediction error before divergence
            stale_threshold: Time before marking agent stale (seconds)
        """
        self.fleet_twin = fleet_twin
        self.sync_rate = sync_rate
        self.prediction_threshold = prediction_threshold
        self.stale_threshold = stale_threshold
        
        # Status
        self._status = SyncStatus.IDLE
        self._running = False
        
        # Tracking
        self._last_update_times: Dict[str, float] = {}
        self._pending_updates: Dict[str, AgentState] = {}
        self._update_latencies: List[float] = []
        
        # Metrics
        self._metrics = SyncMetrics()
        
        # Event handling
        self._event_handlers: List[Callable[[SyncEvent], None]] = []
        self._event_queue: List[SyncEvent] = []
    
    @property
    def status(self) -> SyncStatus:
        """Get synchronization status."""
        return self._status
    
    @property
    def metrics(self) -> SyncMetrics:
        """Get synchronization metrics."""
        return self._metrics
    
    def register_event_handler(self, handler: Callable[[SyncEvent], None]) -> None:
        """Register handler for sync events."""
        self._event_handlers.append(handler)
    
    def queue_state_update(self, agent_id: str, state: AgentState) -> None:
        """Queue a state update for processing."""
        self._pending_updates[agent_id] = state
    
    def process_state_update(
        self,
        agent_id: str,
        state: AgentState,
        receive_time: float = None
    ) -> bool:
        """
        Process a state update from physical agent.
        
        Args:
            agent_id: Agent identifier
            state: New state from physical agent
            receive_time: Time update was received
            
        Returns:
            True if update successful
        """
        receive_time = receive_time or time.time()
        
        twin = self.fleet_twin.get_agent_twin(agent_id)
        if twin is None:
            logger.warning(f"No twin found for agent {agent_id}")
            return False
        
        try:
            # Check for prediction divergence
            if twin.status == TwinStatus.SYNCHRONIZED:
                self._check_prediction_divergence(twin, state)
            
            # Update twin
            twin.update_from_agent_state(state)
            
            # Track latency
            latency = (receive_time - state.timestamp) * 1000
            self._update_latencies.append(latency)
            if len(self._update_latencies) > 100:
                self._update_latencies = self._update_latencies[-100:]
            
            # Update metrics
            self._metrics.total_updates += 1
            self._metrics.successful_updates += 1
            self._last_update_times[agent_id] = receive_time
            
            # Emit event
            self._emit_event(SyncEvent(
                event_type=SyncEventType.STATE_UPDATE,
                agent_id=agent_id,
                data={"latency_ms": latency}
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update twin {agent_id}: {e}")
            self._metrics.failed_updates += 1
            
            self._emit_event(SyncEvent(
                event_type=SyncEventType.SYNC_ERROR,
                agent_id=agent_id,
                data={"error": str(e)}
            ))
            
            return False
    
    def _check_prediction_divergence(
        self,
        twin: AgentTwin,
        actual_state: AgentState
    ) -> None:
        """Check if prediction diverged from actual state."""
        # Get predicted state at actual state's timestamp
        dt = actual_state.timestamp - twin.state.timestamp
        if dt <= 0:
            return
        
        predicted = twin.predict_state(dt)
        
        # Compare positions
        actual_pos = actual_state.position
        pred_pos = predicted.position
        
        error = (actual_pos - pred_pos).norm()
        
        if error > self.prediction_threshold:
            self._metrics.divergence_count += 1
            
            self._emit_event(SyncEvent(
                event_type=SyncEventType.PREDICTION_DIVERGENCE,
                agent_id=twin.agent_id,
                data={
                    "prediction_error": error,
                    "predicted_pos": [pred_pos.x, pred_pos.y, pred_pos.z],
                    "actual_pos": [actual_pos.x, actual_pos.y, actual_pos.z],
                }
            ))
    
    def check_stale_agents(self) -> List[str]:
        """Check for agents with stale data."""
        stale = []
        current_time = time.time()
        
        for agent_id in self.fleet_twin.get_agent_ids():
            last_update = self._last_update_times.get(agent_id, 0)
            if current_time - last_update > self.stale_threshold:
                stale.append(agent_id)
                
                twin = self.fleet_twin.get_agent_twin(agent_id)
                if twin and twin.status != TwinStatus.DISCONNECTED:
                    self._emit_event(SyncEvent(
                        event_type=SyncEventType.CONNECTION_LOST,
                        agent_id=agent_id,
                        data={"last_update": last_update}
                    ))
        
        return stale
    
    def process_pending_updates(self) -> int:
        """Process all pending state updates."""
        processed = 0
        
        for agent_id, state in list(self._pending_updates.items()):
            if self.process_state_update(agent_id, state):
                processed += 1
            del self._pending_updates[agent_id]
        
        return processed
    
    def synchronize(self) -> None:
        """Perform a synchronization cycle."""
        self._status = SyncStatus.SYNCING
        
        # Process pending updates
        self.process_pending_updates()
        
        # Check for stale agents
        self.check_stale_agents()
        
        # Update fleet metrics
        self.fleet_twin.update()
        
        # Update sync metrics
        self._update_metrics()
        
        self._status = SyncStatus.SYNCHRONIZED
    
    def _update_metrics(self) -> None:
        """Update synchronization metrics."""
        if self._update_latencies:
            self._metrics.avg_latency_ms = sum(self._update_latencies) / len(self._update_latencies)
            self._metrics.max_latency_ms = max(self._update_latencies)
        
        self._metrics.last_sync_time = time.time()
        
        # Compute sync rate
        total = self._metrics.total_updates
        if total > 0 and self._metrics.last_sync_time > 0:
            # Estimate based on recent updates
            pass  # Would need time window tracking
    
    def _emit_event(self, event: SyncEvent) -> None:
        """Emit a sync event."""
        self._event_queue.append(event)
        
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_pending_events(self, clear: bool = True) -> List[SyncEvent]:
        """Get pending events."""
        events = self._event_queue.copy()
        if clear:
            self._event_queue = []
        return events
    
    async def run_async(self) -> None:
        """Run synchronization loop asynchronously."""
        self._running = True
        interval = 1.0 / self.sync_rate
        
        while self._running:
            start = time.time()
            
            self.synchronize()
            
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    def stop(self) -> None:
        """Stop synchronization loop."""
        self._running = False
        self._status = SyncStatus.IDLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronizer statistics."""
        return {
            "status": self._status.name,
            "total_updates": self._metrics.total_updates,
            "successful_updates": self._metrics.successful_updates,
            "failed_updates": self._metrics.failed_updates,
            "avg_latency_ms": self._metrics.avg_latency_ms,
            "max_latency_ms": self._metrics.max_latency_ms,
            "divergence_count": self._metrics.divergence_count,
            "pending_updates": len(self._pending_updates),
            "event_queue_size": len(self._event_queue),
        }


class HierarchicalSynchronizer:
    """
    Hierarchical synchronizer for all twin levels.
    
    Coordinates synchronization across agent, fleet, and mission twins.
    """
    
    def __init__(
        self,
        fleet_twin: FleetTwin,
        mission_twins: List[MissionTwin] = None
    ):
        """
        Initialize hierarchical synchronizer.
        
        Args:
            fleet_twin: Fleet-level twin
            mission_twins: Mission-level twins
        """
        self.fleet_twin = fleet_twin
        self.mission_twins = mission_twins or []
        
        # Agent-level synchronizer
        self.agent_sync = TwinSynchronizer(fleet_twin)
        
        # Cross-level event handlers
        self.agent_sync.register_event_handler(self._on_agent_event)
    
    def add_mission_twin(self, mission_twin: MissionTwin) -> None:
        """Add a mission twin."""
        self.mission_twins.append(mission_twin)
    
    def remove_mission_twin(self, mission_id: str) -> None:
        """Remove a mission twin."""
        self.mission_twins = [m for m in self.mission_twins 
                            if m.mission_id != mission_id]
    
    def synchronize_all(self) -> None:
        """Synchronize all levels."""
        # Agent level
        self.agent_sync.synchronize()
        
        # Fleet level (already updated by agent sync)
        
        # Mission level - update metrics based on fleet state
        for mission in self.mission_twins:
            self._sync_mission_twin(mission)
    
    def _sync_mission_twin(self, mission: MissionTwin) -> None:
        """Synchronize a mission twin with fleet state."""
        # Update task progress based on agent states
        for agent_id in mission.state.assigned_agents:
            twin = self.fleet_twin.get_agent_twin(agent_id)
            if twin and twin.state.current_task_id:
                # Could update task progress here
                pass
    
    def _on_agent_event(self, event: SyncEvent) -> None:
        """Handle agent-level sync events."""
        if event.event_type == SyncEventType.CONNECTION_LOST:
            # Could trigger mission-level replanning
            logger.warning(f"Agent {event.agent_id} connection lost - "
                         "may need mission replanning")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all levels."""
        return {
            "agent_sync": self.agent_sync.get_statistics(),
            "fleet": self.fleet_twin.get_statistics(),
            "missions": [m.get_statistics() for m in self.mission_twins],
        }
