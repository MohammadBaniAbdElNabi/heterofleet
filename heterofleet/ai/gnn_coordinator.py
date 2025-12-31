"""
Graph Neural Network Coordinator for HeteroFleet.

Implements GNN-based coordination for heterogeneous fleets.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3


@dataclass
class NodeFeatures:
    """Features for a graph node (agent)."""
    
    agent_id: str = ""
    platform_type: int = 0  # Encoded platform type
    
    # State features
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Goal features
    goal_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Status features
    battery_level: float = 1.0
    is_active: bool = True
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.concatenate([
            [self.platform_type],
            self.position,
            self.velocity,
            self.goal_position,
            self.goal_velocity,
            [self.battery_level, float(self.is_active)]
        ])


@dataclass
class EdgeFeatures:
    """Features for a graph edge (interaction)."""
    
    source_id: str = ""
    target_id: str = ""
    
    # Relative features
    relative_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    relative_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 0.0
    
    # Interaction type
    is_same_type: bool = True
    communication_quality: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.concatenate([
            self.relative_position,
            self.relative_velocity,
            [self.distance, float(self.is_same_type), self.communication_quality]
        ])


@dataclass
class FleetGraph:
    """Graph representation of fleet for GNN processing."""
    
    # Node data
    node_ids: List[str] = field(default_factory=list)
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    
    # Edge data (adjacency)
    edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=int))
    edge_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if len(self.edge_index.shape) > 1 else 0


@dataclass
class CoordinationPrediction:
    """Coordination prediction from GNN."""
    
    agent_id: str = ""
    
    # Velocity adjustment
    velocity_adjustment: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    
    # Confidence
    confidence: float = 0.0
    
    # Attention weights (which neighbors influenced this)
    attention_weights: Dict[str, float] = field(default_factory=dict)


class MessagePassingLayer:
    """
    Message passing layer for GNN.
    
    Simple implementation without deep learning framework.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """
        Initialize message passing layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (would be learned in real implementation)
        np.random.seed(42)
        self.W_msg = np.random.randn(input_dim * 2, hidden_dim) * 0.1
        self.W_update = np.random.randn(hidden_dim + input_dim, output_dim) * 0.1
        self.W_attention = np.random.randn(hidden_dim, 1) * 0.1
    
    def forward(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of message passing.
        
        Args:
            node_features: Node feature matrix (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_features: Optional edge features
            
        Returns:
            Tuple of (updated_features, attention_weights)
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1] if len(edge_index.shape) > 1 else 0
        
        if num_edges == 0:
            return node_features[:, :self.output_dim], np.zeros((num_nodes, num_nodes))
        
        # Compute messages
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        # Concatenate source and target features
        edge_inputs = np.concatenate([source_features, target_features], axis=1)
        
        # Compute messages
        messages = np.tanh(edge_inputs @ self.W_msg)
        
        # Compute attention weights
        attention_logits = messages @ self.W_attention
        
        # Softmax per target node
        attention_weights_per_edge = np.exp(attention_logits)
        
        # Aggregate messages per target
        aggregated = np.zeros((num_nodes, self.hidden_dim))
        attention_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_edges):
            src = source_nodes[i]
            tgt = target_nodes[i]
            aggregated[tgt] += messages[i] * attention_weights_per_edge[i, 0]
            attention_matrix[tgt, src] = attention_weights_per_edge[i, 0]
        
        # Normalize attention
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        attention_matrix = attention_matrix / row_sums
        
        # Normalize aggregated messages
        for i in range(num_nodes):
            if row_sums[i] > 0:
                aggregated[i] /= row_sums[i]
        
        # Update node features
        update_input = np.concatenate([aggregated, node_features], axis=1)
        updated = np.tanh(update_input @ self.W_update)
        
        return updated, attention_matrix


class GNNCoordinator:
    """
    GNN-based coordinator for heterogeneous fleets.
    
    Uses graph neural networks for distributed coordination
    decisions based on local observations.
    """
    
    def __init__(
        self,
        communication_range: float = 10.0,
        num_layers: int = 2,
        hidden_dim: int = 64
    ):
        """
        Initialize GNN coordinator.
        
        Args:
            communication_range: Range for edge creation
            num_layers: Number of message passing layers
            hidden_dim: Hidden dimension
        """
        self.communication_range = communication_range
        self.num_layers = num_layers
        
        # Platform type encoding
        self._platform_encoding = {
            PlatformType.MICRO_UAV: 0,
            PlatformType.SMALL_UAV: 1,
            PlatformType.MEDIUM_UAV: 2,
            PlatformType.LARGE_UAV: 3,
            PlatformType.SMALL_UGV: 4,
            PlatformType.MEDIUM_UGV: 5,
            PlatformType.LARGE_UGV: 6,
        }
        
        # Node feature dimension: platform_type(1) + pos(3) + vel(3) + goal_pos(3) + goal_vel(3) + battery(1) + active(1) = 15
        # Edge feature dimension: rel_pos(3) + rel_vel(3) + dist(1) + same_type(1) + comm(1) = 9
        self.node_feature_dim = 15
        self.edge_feature_dim = 9
        
        # Output dimension: velocity adjustment (3)
        self.output_dim = 3
        
        # Initialize layers
        self._layers: List[MessagePassingLayer] = []
        
        in_dim = self.node_feature_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.output_dim
            layer = MessagePassingLayer(in_dim, out_dim, hidden_dim)
            self._layers.append(layer)
            in_dim = out_dim
    
    def build_graph(
        self,
        positions: Dict[str, Vector3],
        velocities: Dict[str, Vector3],
        goals: Dict[str, Vector3],
        platform_types: Dict[str, PlatformType],
        battery_levels: Dict[str, float] = None
    ) -> FleetGraph:
        """
        Build fleet graph from current state.
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            goals: Agent goal positions
            platform_types: Agent platform types
            battery_levels: Optional battery levels
            
        Returns:
            Fleet graph representation
        """
        battery_levels = battery_levels or {}
        
        agent_ids = list(positions.keys())
        num_agents = len(agent_ids)
        
        # Build node features
        node_features = np.zeros((num_agents, self.node_feature_dim))
        
        for i, agent_id in enumerate(agent_ids):
            pos = positions[agent_id]
            vel = velocities.get(agent_id, Vector3(0, 0, 0))
            goal = goals.get(agent_id, pos)
            ptype = platform_types.get(agent_id, PlatformType.SMALL_UAV)
            battery = battery_levels.get(agent_id, 1.0)
            
            features = NodeFeatures(
                agent_id=agent_id,
                platform_type=self._platform_encoding.get(ptype, 0),
                position=np.array([pos.x, pos.y, pos.z]),
                velocity=np.array([vel.x, vel.y, vel.z]),
                goal_position=np.array([goal.x, goal.y, goal.z]),
                goal_velocity=np.zeros(3),
                battery_level=battery,
                is_active=True
            )
            
            node_features[i] = features.to_vector()
        
        # Build edges based on communication range
        edges = []
        edge_features_list = []
        
        for i, id_i in enumerate(agent_ids):
            pos_i = positions[id_i]
            ptype_i = platform_types.get(id_i, PlatformType.SMALL_UAV)
            
            for j, id_j in enumerate(agent_ids):
                if i == j:
                    continue
                
                pos_j = positions[id_j]
                ptype_j = platform_types.get(id_j, PlatformType.SMALL_UAV)
                
                dist = (pos_i - pos_j).norm()
                
                if dist <= self.communication_range:
                    edges.append([i, j])
                    
                    vel_i = velocities.get(id_i, Vector3(0, 0, 0))
                    vel_j = velocities.get(id_j, Vector3(0, 0, 0))
                    
                    rel_pos = np.array([
                        pos_j.x - pos_i.x,
                        pos_j.y - pos_i.y,
                        pos_j.z - pos_i.z
                    ])
                    rel_vel = np.array([
                        vel_j.x - vel_i.x,
                        vel_j.y - vel_i.y,
                        vel_j.z - vel_i.z
                    ])
                    
                    edge_feat = EdgeFeatures(
                        source_id=id_i,
                        target_id=id_j,
                        relative_position=rel_pos,
                        relative_velocity=rel_vel,
                        distance=dist,
                        is_same_type=(ptype_i == ptype_j),
                        communication_quality=1.0 - dist / self.communication_range
                    )
                    
                    edge_features_list.append(edge_feat.to_vector())
        
        # Convert to arrays
        if edges:
            edge_index = np.array(edges).T
            edge_features = np.array(edge_features_list)
        else:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_features = np.zeros((0, self.edge_feature_dim))
        
        return FleetGraph(
            node_ids=agent_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
    
    def forward(self, graph: FleetGraph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through GNN.
        
        Args:
            graph: Fleet graph
            
        Returns:
            Tuple of (velocity_adjustments, attention_weights)
        """
        features = graph.node_features
        attention = None
        
        for layer in self._layers:
            features, attention = layer.forward(
                features,
                graph.edge_index,
                graph.edge_features
            )
        
        return features, attention
    
    def compute_coordination(
        self,
        positions: Dict[str, Vector3],
        velocities: Dict[str, Vector3],
        goals: Dict[str, Vector3],
        platform_types: Dict[str, PlatformType],
        battery_levels: Dict[str, float] = None
    ) -> Dict[str, CoordinationPrediction]:
        """
        Compute coordination predictions for all agents.
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            goals: Agent goal positions
            platform_types: Agent platform types
            battery_levels: Optional battery levels
            
        Returns:
            Dictionary of coordination predictions
        """
        # Build graph
        graph = self.build_graph(
            positions, velocities, goals, platform_types, battery_levels
        )
        
        if graph.num_nodes == 0:
            return {}
        
        # Forward pass
        velocity_adjustments, attention = self.forward(graph)
        
        # Build predictions
        predictions = {}
        
        for i, agent_id in enumerate(graph.node_ids):
            adj = velocity_adjustments[i]
            
            # Build attention weights dict
            attn_weights = {}
            for j, other_id in enumerate(graph.node_ids):
                if attention[i, j] > 0.01:
                    attn_weights[other_id] = float(attention[i, j])
            
            predictions[agent_id] = CoordinationPrediction(
                agent_id=agent_id,
                velocity_adjustment=Vector3(adj[0], adj[1], adj[2]),
                confidence=float(np.linalg.norm(adj)),
                attention_weights=attn_weights
            )
        
        return predictions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "num_layers": self.num_layers,
            "communication_range": self.communication_range,
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
        }
