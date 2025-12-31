"""
Dashboard for HeteroFleet Monitoring.

Provides real-time fleet status and metrics display.

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

from heterofleet.digital_twin.fleet_twin import FleetTwin, FleetMetrics
from heterofleet.digital_twin.mission_twin import MissionTwin, MissionMetrics


class WidgetType(Enum):
    """Types of dashboard widgets."""
    METRIC = auto()
    CHART = auto()
    TABLE = auto()
    MAP = auto()
    ALERT = auto()
    STATUS = auto()


@dataclass
class DashboardWidget:
    """A dashboard widget."""
    
    widget_id: str
    widget_type: WidgetType
    title: str
    
    # Position and size (grid units)
    x: int = 0
    y: int = 0
    width: int = 1
    height: int = 1
    
    # Data binding
    data_source: str = ""
    data_key: str = ""
    
    # Display options
    format_string: str = "{value}"
    unit: str = ""
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Current value
    value: Any = None
    last_updated: float = field(default_factory=time.time)
    
    def update(self, value: Any) -> None:
        """Update widget value."""
        self.value = value
        self.last_updated = time.time()
    
    def get_display_value(self) -> str:
        """Get formatted display value."""
        if self.value is None:
            return "N/A"
        
        try:
            formatted = self.format_string.format(value=self.value)
            if self.unit:
                formatted += f" {self.unit}"
            return formatted
        except:
            return str(self.value)
    
    def get_status(self) -> str:
        """Get status based on thresholds."""
        if self.value is None or not self.thresholds:
            return "normal"
        
        try:
            val = float(self.value)
            if "critical" in self.thresholds and val <= self.thresholds["critical"]:
                return "critical"
            if "warning" in self.thresholds and val <= self.thresholds["warning"]:
                return "warning"
            return "normal"
        except:
            return "normal"


@dataclass
class Alert:
    """A dashboard alert."""
    
    alert_id: str
    severity: str  # info, warning, critical
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class Dashboard:
    """
    Fleet monitoring dashboard.
    
    Provides real-time display of:
    - Fleet metrics
    - Agent status
    - Mission progress
    - Alerts and notifications
    """
    
    def __init__(self, title: str = "HeteroFleet Dashboard"):
        """
        Initialize dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        
        # Widgets
        self._widgets: Dict[str, DashboardWidget] = {}
        
        # Data sources
        self._fleet_twin: Optional[FleetTwin] = None
        self._mission_twins: Dict[str, MissionTwin] = {}
        
        # Alerts
        self._alerts: List[Alert] = []
        self._max_alerts = 100
        
        # History
        self._metrics_history: List[Dict[str, Any]] = []
        self._history_size = 1000
        
        # Callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Create default widgets
        self._create_default_widgets()
    
    def _create_default_widgets(self) -> None:
        """Create default dashboard widgets."""
        # Fleet status widgets
        self.add_widget(DashboardWidget(
            widget_id="total_agents",
            widget_type=WidgetType.METRIC,
            title="Total Agents",
            data_source="fleet",
            data_key="total_agents",
            format_string="{value}",
            x=0, y=0, width=1, height=1
        ))
        
        self.add_widget(DashboardWidget(
            widget_id="active_agents",
            widget_type=WidgetType.METRIC,
            title="Active Agents",
            data_source="fleet",
            data_key="active_agents",
            format_string="{value}",
            x=1, y=0, width=1, height=1
        ))
        
        self.add_widget(DashboardWidget(
            widget_id="avg_battery",
            widget_type=WidgetType.METRIC,
            title="Avg Battery",
            data_source="fleet",
            data_key="avg_battery_level",
            format_string="{value:.0%}",
            thresholds={"warning": 0.3, "critical": 0.1},
            x=2, y=0, width=1, height=1
        ))
        
        self.add_widget(DashboardWidget(
            widget_id="min_battery",
            widget_type=WidgetType.METRIC,
            title="Min Battery",
            data_source="fleet",
            data_key="min_battery_level",
            format_string="{value:.0%}",
            thresholds={"warning": 0.2, "critical": 0.1},
            x=3, y=0, width=1, height=1
        ))
        
        self.add_widget(DashboardWidget(
            widget_id="disconnected",
            widget_type=WidgetType.METRIC,
            title="Disconnected",
            data_source="fleet",
            data_key="disconnected_agents",
            format_string="{value}",
            thresholds={"warning": 1, "critical": 3},
            x=4, y=0, width=1, height=1
        ))
        
        # Alert widget
        self.add_widget(DashboardWidget(
            widget_id="alerts",
            widget_type=WidgetType.ALERT,
            title="Recent Alerts",
            x=0, y=2, width=5, height=2
        ))
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add a widget to the dashboard."""
        self._widgets[widget.widget_id] = widget
    
    def remove_widget(self, widget_id: str) -> None:
        """Remove a widget from the dashboard."""
        self._widgets.pop(widget_id, None)
    
    def get_widget(self, widget_id: str) -> Optional[DashboardWidget]:
        """Get a widget by ID."""
        return self._widgets.get(widget_id)
    
    def set_fleet_twin(self, fleet_twin: FleetTwin) -> None:
        """Set fleet twin as data source."""
        self._fleet_twin = fleet_twin
    
    def add_mission_twin(self, mission_twin: MissionTwin) -> None:
        """Add mission twin as data source."""
        self._mission_twins[mission_twin.mission_id] = mission_twin
    
    def update(self) -> None:
        """Update all widgets with current data."""
        # Update fleet widgets
        if self._fleet_twin:
            metrics = self._fleet_twin.metrics
            
            for widget in self._widgets.values():
                if widget.data_source == "fleet" and widget.data_key:
                    value = getattr(metrics, widget.data_key, None)
                    if value is not None:
                        widget.update(value)
                        
                        # Check thresholds for alerts
                        status = widget.get_status()
                        if status == "critical":
                            self._create_alert(
                                "critical",
                                f"{widget.title} is critical: {widget.get_display_value()}",
                                "fleet"
                            )
                        elif status == "warning":
                            self._create_alert(
                                "warning",
                                f"{widget.title} warning: {widget.get_display_value()}",
                                "fleet"
                            )
        
        # Update mission widgets
        for mission_id, mission_twin in self._mission_twins.items():
            for widget in self._widgets.values():
                if widget.data_source == f"mission:{mission_id}" and widget.data_key:
                    value = getattr(mission_twin.state.metrics, widget.data_key, None)
                    if value is not None:
                        widget.update(value)
        
        # Store history
        self._record_history()
    
    def _record_history(self) -> None:
        """Record metrics history."""
        if self._fleet_twin is None:
            return
        
        metrics = self._fleet_twin.metrics
        
        record = {
            "timestamp": time.time(),
            "total_agents": metrics.total_agents,
            "active_agents": metrics.active_agents,
            "avg_battery": metrics.avg_battery_level,
            "min_battery": metrics.min_battery_level,
            "disconnected": metrics.disconnected_agents,
        }
        
        self._metrics_history.append(record)
        
        if len(self._metrics_history) > self._history_size:
            self._metrics_history = self._metrics_history[-self._history_size:]
    
    def _create_alert(self, severity: str, message: str, source: str) -> None:
        """Create an alert."""
        # Check for duplicate recent alerts
        for alert in self._alerts[-10:]:
            if alert.message == message and not alert.acknowledged:
                return
        
        alert = Alert(
            alert_id=f"alert_{len(self._alerts)}_{time.time()}",
            severity=severity,
            message=message,
            source=source,
        )
        
        self._alerts.append(alert)
        
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            callback(alert)
        
        logger.log(
            "WARNING" if severity == "warning" else "ERROR" if severity == "critical" else "INFO",
            f"Alert: {message}"
        )
    
    def add_alert(self, severity: str, message: str, source: str = "user") -> Alert:
        """Add a manual alert."""
        self._create_alert(severity, message, source)
        return self._alerts[-1]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_alerts(self, acknowledged_only: bool = True) -> None:
        """Clear alerts."""
        if acknowledged_only:
            self._alerts = [a for a in self._alerts if not a.acknowledged]
        else:
            self._alerts = []
    
    def get_alerts(self, severity: str = None, limit: int = None) -> List[Alert]:
        """Get alerts, optionally filtered."""
        alerts = self._alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def get_history(self, duration: float = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        if duration is None:
            return self._metrics_history.copy()
        
        cutoff = time.time() - duration
        return [r for r in self._metrics_history if r["timestamp"] >= cutoff]
    
    def render_text(self) -> str:
        """Render dashboard as text."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f" {self.title}")
        lines.append(f"{'='*60}")
        
        # Group widgets by row
        widgets_by_row: Dict[int, List[DashboardWidget]] = {}
        for widget in self._widgets.values():
            if widget.widget_type == WidgetType.METRIC:
                if widget.y not in widgets_by_row:
                    widgets_by_row[widget.y] = []
                widgets_by_row[widget.y].append(widget)
        
        for y in sorted(widgets_by_row.keys()):
            row_widgets = sorted(widgets_by_row[y], key=lambda w: w.x)
            
            row_text = " | ".join(
                f"{w.title}: {w.get_display_value()}" for w in row_widgets
            )
            lines.append(row_text)
        
        # Alerts
        lines.append(f"\n{'Recent Alerts':-^60}")
        recent_alerts = self.get_alerts(limit=5)
        
        if recent_alerts:
            for alert in recent_alerts:
                ack = "[ACK]" if alert.acknowledged else "[NEW]"
                lines.append(f"  {ack} [{alert.severity.upper()}] {alert.message}")
        else:
            lines.append("  No alerts")
        
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export dashboard state to dictionary."""
        return {
            "title": self.title,
            "widgets": {
                wid: {
                    "widget_id": w.widget_id,
                    "type": w.widget_type.name,
                    "title": w.title,
                    "value": w.get_display_value(),
                    "status": w.get_status(),
                    "position": {"x": w.x, "y": w.y},
                    "size": {"width": w.width, "height": w.height},
                }
                for wid, w in self._widgets.items()
            },
            "alerts": [a.to_dict() for a in self.get_alerts(limit=10)],
            "timestamp": time.time(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "num_widgets": len(self._widgets),
            "num_alerts": len(self._alerts),
            "unacknowledged_alerts": sum(1 for a in self._alerts if not a.acknowledged),
            "history_size": len(self._metrics_history),
        }
