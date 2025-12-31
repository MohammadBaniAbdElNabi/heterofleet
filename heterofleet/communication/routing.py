"""
Adaptive Routing for HeteroFleet Communication.

Implements intelligent message routing with dynamic topology adaptation.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import Vector3
from heterofleet.core.message import Message, MessageType, MessagePriority


class RouteStatus(Enum):
    """Status of a route."""
    ACTIVE = auto()
    DEGRADED = auto()
    FAILED = auto()
    UNKNOWN = auto()


@dataclass
class LinkMetrics:
    """Metrics for a communication link."""
    source: str
    target: str
    
    # Quality metrics
    rssi: float = -50.0  # dBm
    snr: float = 20.0    # dB
    packet_loss: float = 0.0  # 0-1
    latency_ms: float = 10.0
    bandwidth_kbps: float = 1000.0
    
    # Reliability
    uptime: float = 1.0  # 0-1
    last_seen: float = field(default_factory=time.time)
    
    # Computed
    quality_score: float = 1.0
    
    def update_quality(self) -> None:
        """Update overall quality score."""
        rssi_score = min(1.0, max(0.0, (self.rssi + 100) / 60))
        loss_score = 1.0 - self.packet_loss
        latency_score = min(1.0, 100 / max(1, self.latency_ms))
        
        self.quality_score = (rssi_score * 0.3 + loss_score * 0.4 + 
                            latency_score * 0.2 + self.uptime * 0.1)


@dataclass 
class RouteEntry:
    """Entry in routing table."""
    destination: str
    next_hop: str
    metric: float  # Lower is better
    hop_count: int
    last_updated: float = field(default_factory=time.time)
    status: RouteStatus = RouteStatus.ACTIVE
    
    @property
    def is_valid(self) -> bool:
        return self.status == RouteStatus.ACTIVE and (time.time() - self.last_updated) < 30.0


class RoutingTable:
    """Routing table for message forwarding."""
    
    def __init__(self, local_id: str):
        self.local_id = local_id
        self._routes: Dict[str, RouteEntry] = {}
        self._neighbors: Set[str] = set()
    
    def add_route(self, entry: RouteEntry) -> None:
        """Add or update a route."""
        existing = self._routes.get(entry.destination)
        
        if existing is None or entry.metric < existing.metric:
            self._routes[entry.destination] = entry
            logger.debug(f"Route to {entry.destination} via {entry.next_hop} (metric={entry.metric:.2f})")
    
    def remove_route(self, destination: str) -> None:
        """Remove a route."""
        self._routes.pop(destination, None)
    
    def get_route(self, destination: str) -> Optional[RouteEntry]:
        """Get route to destination."""
        route = self._routes.get(destination)
        if route and route.is_valid:
            return route
        return None
    
    def get_next_hop(self, destination: str) -> Optional[str]:
        """Get next hop for destination."""
        route = self.get_route(destination)
        return route.next_hop if route else None
    
    def add_neighbor(self, neighbor_id: str) -> None:
        """Add direct neighbor."""
        self._neighbors.add(neighbor_id)
        self.add_route(RouteEntry(
            destination=neighbor_id,
            next_hop=neighbor_id,
            metric=1.0,
            hop_count=1
        ))
    
    def remove_neighbor(self, neighbor_id: str) -> None:
        """Remove neighbor and invalidate routes through it."""
        self._neighbors.discard(neighbor_id)
        
        for dest, route in list(self._routes.items()):
            if route.next_hop == neighbor_id:
                route.status = RouteStatus.FAILED
    
    def get_all_destinations(self) -> List[str]:
        """Get all reachable destinations."""
        return [d for d, r in self._routes.items() if r.is_valid]
    
    def export_routes(self) -> List[RouteEntry]:
        """Export routes for sharing."""
        return [r for r in self._routes.values() if r.is_valid]


class AdaptiveRouter:
    """
    Adaptive message router for heterogeneous fleet.
    
    Implements dynamic routing with quality-aware path selection.
    """
    
    def __init__(
        self,
        local_id: str,
        broadcast_interval: float = 5.0,
        route_timeout: float = 30.0
    ):
        """
        Initialize router.
        
        Args:
            local_id: Local node identifier
            broadcast_interval: Interval for route broadcasts
            route_timeout: Timeout for route entries
        """
        self.local_id = local_id
        self.broadcast_interval = broadcast_interval
        self.route_timeout = route_timeout
        
        # Routing table
        self.routing_table = RoutingTable(local_id)
        
        # Link metrics
        self._link_metrics: Dict[str, LinkMetrics] = {}
        
        # Message queues
        self._outbound_queue: List[Tuple[Message, str]] = []
        self._pending_acks: Dict[str, Tuple[Message, float]] = {}
        
        # Statistics
        self._messages_sent = 0
        self._messages_received = 0
        self._messages_forwarded = 0
        self._messages_dropped = 0
        
        # Callbacks
        self._send_callback: Optional[Callable[[Message, str], bool]] = None
        self._receive_callbacks: List[Callable[[Message], None]] = []
    
    def set_send_callback(self, callback: Callable[[Message, str], bool]) -> None:
        """Set callback for sending messages."""
        self._send_callback = callback
    
    def register_receive_callback(self, callback: Callable[[Message], None]) -> None:
        """Register callback for received messages."""
        self._receive_callbacks.append(callback)
    
    def update_link(self, neighbor_id: str, metrics: LinkMetrics) -> None:
        """Update link metrics for a neighbor."""
        metrics.update_quality()
        self._link_metrics[neighbor_id] = metrics
        
        # Update routing table
        if metrics.quality_score > 0.1:
            self.routing_table.add_neighbor(neighbor_id)
        else:
            self.routing_table.remove_neighbor(neighbor_id)
    
    def send_message(self, message: Message, destination: str) -> bool:
        """
        Send a message to destination.
        
        Args:
            message: Message to send
            destination: Destination node ID
            
        Returns:
            True if message was queued/sent
        """
        if destination == self.local_id:
            # Local delivery
            self._deliver_local(message)
            return True
        
        # Find route
        next_hop = self.routing_table.get_next_hop(destination)
        
        if next_hop is None:
            logger.warning(f"No route to {destination}")
            self._messages_dropped += 1
            return False
        
        # Queue for sending
        self._outbound_queue.append((message, next_hop))
        return True
    
    def receive_message(self, message: Message, from_node: str) -> None:
        """
        Process received message.
        
        Args:
            message: Received message
            from_node: Node that sent the message
        """
        self._messages_received += 1
        
        # Update link metrics
        if from_node in self._link_metrics:
            self._link_metrics[from_node].last_seen = time.time()
        
        # Check if for us
        if message.receiver_id == self.local_id or message.receiver_id == "broadcast":
            self._deliver_local(message)
            
            # Don't forward broadcasts we're the destination for
            if message.receiver_id == self.local_id:
                return
        
        # Forward if needed
        if message.receiver_id != self.local_id and message.receiver_id != "broadcast":
            self._forward_message(message)
    
    def _deliver_local(self, message: Message) -> None:
        """Deliver message to local handlers."""
        for callback in self._receive_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Receive callback error: {e}")
    
    def _forward_message(self, message: Message) -> None:
        """Forward message toward destination."""
        # Check TTL
        # Would need TTL field in message
        
        next_hop = self.routing_table.get_next_hop(message.receiver_id)
        
        if next_hop is None:
            logger.warning(f"Cannot forward to {message.receiver_id} - no route")
            self._messages_dropped += 1
            return
        
        # Queue for forwarding
        self._outbound_queue.append((message, next_hop))
        self._messages_forwarded += 1
    
    def process_outbound(self) -> int:
        """Process outbound message queue."""
        sent = 0
        
        while self._outbound_queue:
            message, next_hop = self._outbound_queue.pop(0)
            
            if self._send_callback:
                if self._send_callback(message, next_hop):
                    self._messages_sent += 1
                    sent += 1
                else:
                    self._messages_dropped += 1
        
        return sent
    
    def broadcast_routes(self) -> Message:
        """Create route broadcast message."""
        routes = self.routing_table.export_routes()
        
        route_data = [
            {
                "destination": r.destination,
                "metric": r.metric + 1,  # Add our hop cost
                "hop_count": r.hop_count + 1,
            }
            for r in routes
        ]
        
        return Message(
            message_id=f"route_{self.local_id}_{time.time()}",
            message_type=MessageType.STATUS,
            sender_id=self.local_id,
            receiver_id="broadcast",
            payload={"type": "routing_update", "routes": route_data},
            priority=MessagePriority.LOW,
        )
    
    def process_route_update(self, from_node: str, routes: List[Dict]) -> None:
        """Process routing update from neighbor."""
        for route_data in routes:
            dest = route_data["destination"]
            
            if dest == self.local_id:
                continue
            
            # Compute metric including link quality
            link = self._link_metrics.get(from_node)
            link_cost = 1.0 / link.quality_score if link else 2.0
            
            metric = route_data["metric"] + link_cost
            
            entry = RouteEntry(
                destination=dest,
                next_hop=from_node,
                metric=metric,
                hop_count=route_data["hop_count"],
            )
            
            self.routing_table.add_route(entry)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "local_id": self.local_id,
            "neighbors": len(self.routing_table._neighbors),
            "routes": len(self.routing_table._routes),
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "messages_forwarded": self._messages_forwarded,
            "messages_dropped": self._messages_dropped,
            "queue_size": len(self._outbound_queue),
        }


class MeshNetwork:
    """
    Mesh network manager for fleet communication.
    
    Coordinates multiple routers for fleet-wide communication.
    """
    
    def __init__(self):
        """Initialize mesh network."""
        self._routers: Dict[str, AdaptiveRouter] = {}
        self._positions: Dict[str, Vector3] = {}
        self._comm_range = 50.0  # meters
    
    def add_node(self, node_id: str, position: Vector3) -> AdaptiveRouter:
        """Add a node to the network."""
        router = AdaptiveRouter(node_id)
        self._routers[node_id] = router
        self._positions[node_id] = position
        
        # Set up send callback
        router.set_send_callback(lambda msg, hop: self._route_message(node_id, msg, hop))
        
        # Update connectivity
        self._update_connectivity(node_id)
        
        return router
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        self._routers.pop(node_id, None)
        self._positions.pop(node_id, None)
        
        # Update other nodes
        for router in self._routers.values():
            router.routing_table.remove_neighbor(node_id)
    
    def update_position(self, node_id: str, position: Vector3) -> None:
        """Update node position."""
        self._positions[node_id] = position
        self._update_connectivity(node_id)
    
    def _update_connectivity(self, node_id: str) -> None:
        """Update connectivity for a node."""
        if node_id not in self._positions:
            return
        
        pos = self._positions[node_id]
        router = self._routers.get(node_id)
        
        if router is None:
            return
        
        for other_id, other_pos in self._positions.items():
            if other_id == node_id:
                continue
            
            dist = (pos - other_pos).norm()
            
            if dist <= self._comm_range:
                # Compute link quality based on distance
                quality = 1.0 - (dist / self._comm_range) ** 2
                rssi = -50 - 50 * (dist / self._comm_range)
                
                metrics = LinkMetrics(
                    source=node_id,
                    target=other_id,
                    rssi=rssi,
                    quality_score=quality,
                )
                
                router.update_link(other_id, metrics)
            else:
                router.routing_table.remove_neighbor(other_id)
    
    def _route_message(self, from_node: str, message: Message, to_node: str) -> bool:
        """Route message between nodes."""
        if to_node not in self._routers:
            return False
        
        # Simulate transmission
        self._routers[to_node].receive_message(message, from_node)
        return True
    
    def send_message(self, from_node: str, to_node: str, message: Message) -> bool:
        """Send message from one node to another."""
        router = self._routers.get(from_node)
        if router is None:
            return False
        
        return router.send_message(message, to_node)
    
    def broadcast_message(self, from_node: str, message: Message) -> int:
        """Broadcast message from a node."""
        router = self._routers.get(from_node)
        if router is None:
            return 0
        
        message.receiver_id = "broadcast"
        
        # Send to all neighbors
        sent = 0
        for neighbor in router.routing_table._neighbors:
            if self._route_message(from_node, message, neighbor):
                sent += 1
        
        return sent
    
    def update_routing(self) -> None:
        """Update routing tables across network."""
        # Broadcast route updates
        for node_id, router in self._routers.items():
            update_msg = router.broadcast_routes()
            
            for neighbor in router.routing_table._neighbors:
                other_router = self._routers.get(neighbor)
                if other_router:
                    routes = update_msg.payload.get("routes", [])
                    other_router.process_route_update(node_id, routes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "num_nodes": len(self._routers),
            "nodes": {nid: r.get_statistics() for nid, r in self._routers.items()},
        }
