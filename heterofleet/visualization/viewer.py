"""
3D Fleet Viewer for HeteroFleet.

Provides real-time 3D visualization of fleet operations.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from heterofleet.core.platform import PlatformType, Vector3
from heterofleet.digital_twin.fleet_twin import FleetTwin


@dataclass
class ViewerConfig:
    """Configuration for fleet viewer."""
    
    # Window
    width: int = 1280
    height: int = 720
    title: str = "HeteroFleet Viewer"
    
    # Camera
    camera_distance: float = 20.0
    camera_elevation: float = 45.0
    camera_azimuth: float = 45.0
    
    # Grid
    show_grid: bool = True
    grid_size: float = 50.0
    grid_spacing: float = 1.0
    
    # Agents
    agent_scale: float = 1.0
    show_trajectories: bool = True
    trajectory_length: int = 100
    
    # Colors (RGB 0-1)
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    uav_color: Tuple[float, float, float] = (0.2, 0.6, 1.0)
    ugv_color: Tuple[float, float, float] = (0.2, 0.8, 0.2)
    
    # UI
    show_labels: bool = True
    show_stats: bool = True


@dataclass
class AgentVisual:
    """Visual representation of an agent."""
    agent_id: str
    platform_type: PlatformType
    position: Vector3
    orientation: Tuple[float, float, float]
    color: Tuple[float, float, float]
    trajectory: List[Vector3] = field(default_factory=list)
    label: str = ""
    visible: bool = True


class FleetViewer:
    """
    3D viewer for heterogeneous fleet visualization.
    
    Provides real-time rendering of:
    - Agent positions and orientations
    - Trajectories
    - Environment obstacles
    - Communication links
    """
    
    def __init__(self, config: ViewerConfig = None):
        """
        Initialize fleet viewer.
        
        Args:
            config: Viewer configuration
        """
        self.config = config or ViewerConfig()
        
        # Agent visuals
        self._agents: Dict[str, AgentVisual] = {}
        
        # Environment
        self._obstacles: List[Dict[str, Any]] = []
        self._waypoints: List[Vector3] = []
        
        # Camera
        self._camera_pos = np.array([0.0, 0.0, 0.0])
        self._camera_target = np.array([0.0, 0.0, 0.0])
        
        # State
        self._is_running = False
        self._frame_count = 0
        
        # Try to import visualization backend
        self._backend = None
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize visualization backend."""
        # Try matplotlib for basic 3D
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self._backend = "matplotlib"
            logger.info("Using matplotlib backend for visualization")
        except ImportError:
            logger.warning("matplotlib not available - visualization will be limited")
            self._backend = None
    
    def add_agent(
        self,
        agent_id: str,
        platform_type: PlatformType,
        position: Vector3 = None,
        color: Tuple[float, float, float] = None
    ) -> None:
        """Add an agent to the viewer."""
        if color is None:
            if platform_type.name.startswith("AERIAL"):
                color = self.config.uav_color
            else:
                color = self.config.ugv_color
        
        self._agents[agent_id] = AgentVisual(
            agent_id=agent_id,
            platform_type=platform_type,
            position=position or Vector3(0, 0, 0),
            orientation=(0, 0, 0),
            color=color,
            label=agent_id,
        )
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the viewer."""
        self._agents.pop(agent_id, None)
    
    def update_agent(
        self,
        agent_id: str,
        position: Vector3,
        orientation: Tuple[float, float, float] = None
    ) -> None:
        """Update agent position and orientation."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        
        # Store trajectory
        if self.config.show_trajectories:
            agent.trajectory.append(Vector3(position.x, position.y, position.z))
            if len(agent.trajectory) > self.config.trajectory_length:
                agent.trajectory = agent.trajectory[-self.config.trajectory_length:]
        
        agent.position = position
        if orientation:
            agent.orientation = orientation
    
    def update_from_fleet_twin(self, fleet_twin: FleetTwin) -> None:
        """Update viewer from fleet twin."""
        for agent_id in fleet_twin.get_agent_ids():
            twin = fleet_twin.get_agent_twin(agent_id)
            if twin is None:
                continue
            
            # Add if not exists
            if agent_id not in self._agents:
                self.add_agent(agent_id, twin.platform_spec.platform_type)
            
            # Update
            self.update_agent(
                agent_id,
                twin.state.position,
                (twin.state.orientation.x, twin.state.orientation.y, twin.state.orientation.z)
            )
    
    def add_obstacle(
        self,
        position: Vector3,
        size: Vector3,
        color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> None:
        """Add an obstacle to the scene."""
        self._obstacles.append({
            "position": position,
            "size": size,
            "color": color,
        })
    
    def add_waypoint(self, position: Vector3) -> None:
        """Add a waypoint marker."""
        self._waypoints.append(position)
    
    def clear_waypoints(self) -> None:
        """Clear all waypoints."""
        self._waypoints = []
    
    def render_frame(self) -> Optional[np.ndarray]:
        """
        Render a frame.
        
        Returns:
            Frame as numpy array (RGB) or None
        """
        if self._backend == "matplotlib":
            return self._render_matplotlib()
        return None
    
    def _render_matplotlib(self) -> Optional[np.ndarray]:
        """Render using matplotlib."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from io import BytesIO
        
        fig = plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background
        ax.set_facecolor(self.config.background_color)
        fig.patch.set_facecolor(self.config.background_color)
        
        # Draw grid
        if self.config.show_grid:
            grid_size = self.config.grid_size
            x = np.linspace(-grid_size/2, grid_size/2, int(grid_size/self.config.grid_spacing)+1)
            y = np.linspace(-grid_size/2, grid_size/2, int(grid_size/self.config.grid_spacing)+1)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            ax.plot_wireframe(X, Y, Z, color=self.config.grid_color, alpha=0.3, linewidth=0.5)
        
        # Draw obstacles
        for obs in self._obstacles:
            pos = obs["position"]
            size = obs["size"]
            color = obs["color"]
            
            # Draw as simple cube representation
            ax.scatter([pos.x], [pos.y], [pos.z], c=[color], s=100, marker='s', alpha=0.5)
        
        # Draw agents
        for agent_id, agent in self._agents.items():
            if not agent.visible:
                continue
            
            pos = agent.position
            
            # Draw agent marker
            if agent.platform_type.name.startswith("AERIAL"):
                marker = '^'
            else:
                marker = 's'
            
            ax.scatter([pos.x], [pos.y], [pos.z], c=[agent.color], s=100, marker=marker)
            
            # Draw trajectory
            if self.config.show_trajectories and agent.trajectory:
                xs = [p.x for p in agent.trajectory]
                ys = [p.y for p in agent.trajectory]
                zs = [p.z for p in agent.trajectory]
                ax.plot(xs, ys, zs, c=agent.color, alpha=0.5, linewidth=1)
            
            # Draw label
            if self.config.show_labels:
                ax.text(pos.x, pos.y, pos.z + 0.5, agent.label, fontsize=8, color='white')
        
        # Draw waypoints
        for wp in self._waypoints:
            ax.scatter([wp.x], [wp.y], [wp.z], c='yellow', s=50, marker='*')
        
        # Set view
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        
        # Set limits
        ax.set_xlim(-self.config.grid_size/2, self.config.grid_size/2)
        ax.set_ylim(-self.config.grid_size/2, self.config.grid_size/2)
        ax.set_zlim(0, self.config.grid_size/4)
        
        ax.view_init(elev=self.config.camera_elevation, azim=self.config.camera_azimuth)
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        
        from PIL import Image
        img = Image.open(buf)
        frame = np.array(img)
        
        plt.close(fig)
        
        self._frame_count += 1
        
        return frame
    
    def save_frame(self, filename: str) -> bool:
        """Save current frame to file."""
        frame = self.render_frame()
        if frame is None:
            return False
        
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(filename)
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get viewer statistics."""
        return {
            "num_agents": len(self._agents),
            "num_obstacles": len(self._obstacles),
            "num_waypoints": len(self._waypoints),
            "frame_count": self._frame_count,
            "backend": self._backend,
        }


class TrajectoryExporter:
    """Export trajectories to various formats."""
    
    @staticmethod
    def to_csv(trajectories: Dict[str, List[Vector3]], filename: str) -> bool:
        """Export trajectories to CSV."""
        try:
            with open(filename, 'w') as f:
                f.write("agent_id,step,x,y,z\n")
                for agent_id, traj in trajectories.items():
                    for i, pos in enumerate(traj):
                        f.write(f"{agent_id},{i},{pos.x:.4f},{pos.y:.4f},{pos.z:.4f}\n")
            return True
        except Exception as e:
            logger.error(f"Failed to export trajectories: {e}")
            return False
    
    @staticmethod
    def to_json(trajectories: Dict[str, List[Vector3]], filename: str) -> bool:
        """Export trajectories to JSON."""
        import json
        
        try:
            data = {
                agent_id: [[p.x, p.y, p.z] for p in traj]
                for agent_id, traj in trajectories.items()
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export trajectories: {e}")
            return False
