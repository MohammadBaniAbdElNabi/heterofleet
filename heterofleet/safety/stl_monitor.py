"""
Signal Temporal Logic (STL) Monitor for safety verification.

Implements STL formula evaluation and robustness computation
for runtime verification of safety properties.

Key features:
- STL formula representation
- Online monitoring
- Robustness computation
- Predicate abstraction

Based on HeteroFleet Architecture v1.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger


@dataclass
class Signal:
    """A signal (time series) for STL evaluation."""
    
    timestamps: np.ndarray
    values: np.ndarray  # Can be 1D or 2D (multiple signals)
    name: str = ""
    
    def __post_init__(self):
        """Validate signal data."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have same length")
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def at(self, t: float) -> np.ndarray:
        """Get interpolated value at time t."""
        if len(self.timestamps) == 0:
            return np.array([])
        
        if t <= self.timestamps[0]:
            return self.values[0]
        if t >= self.timestamps[-1]:
            return self.values[-1]
        
        # Linear interpolation
        idx = np.searchsorted(self.timestamps, t)
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        v0, v1 = self.values[idx - 1], self.values[idx]
        
        alpha = (t - t0) / (t1 - t0)
        return v0 + alpha * (v1 - v0)
    
    def slice(self, t_start: float, t_end: float) -> Signal:
        """Get signal slice between t_start and t_end."""
        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)
        return Signal(
            timestamps=self.timestamps[mask],
            values=self.values[mask],
            name=self.name
        )


@dataclass
class RobustnessResult:
    """Result of STL robustness computation."""
    
    value: float  # Robustness value (positive = satisfied)
    satisfied: bool
    formula_name: str = ""
    timestamp: float = 0.0
    
    # Detailed info
    sub_results: List[RobustnessResult] = field(default_factory=list)
    
    @property
    def is_satisfied(self) -> bool:
        return self.value > 0 or (self.value == 0 and self.satisfied)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "satisfied": self.satisfied,
            "formula": self.formula_name,
            "timestamp": self.timestamp,
        }


class STLFormula(ABC):
    """Abstract base class for STL formulas."""
    
    def __init__(self, name: str = ""):
        self.name = name
    
    @abstractmethod
    def evaluate(self, signal: Signal, t: float) -> bool:
        """Evaluate formula satisfaction at time t."""
        pass
    
    @abstractmethod
    def robustness(self, signal: Signal, t: float) -> float:
        """Compute robustness at time t."""
        pass
    
    def evaluate_signal(self, signal: Signal) -> List[bool]:
        """Evaluate formula over entire signal."""
        return [self.evaluate(signal, t) for t in signal.timestamps]
    
    def robustness_signal(self, signal: Signal) -> np.ndarray:
        """Compute robustness over entire signal."""
        return np.array([self.robustness(signal, t) for t in signal.timestamps])
    
    def get_robustness_result(self, signal: Signal, t: float) -> RobustnessResult:
        """Get detailed robustness result."""
        rob = self.robustness(signal, t)
        return RobustnessResult(
            value=rob,
            satisfied=rob >= 0,
            formula_name=self.name,
            timestamp=t
        )


class STLPredicate(STLFormula):
    """
    Atomic predicate: f(x) >= 0
    
    The predicate function f returns the "signed distance" to satisfaction.
    Positive means satisfied, negative means violated.
    """
    
    def __init__(
        self,
        predicate_fn: Callable[[np.ndarray], float],
        name: str = "predicate"
    ):
        """
        Initialize predicate.
        
        Args:
            predicate_fn: Function that returns signed distance
            name: Predicate name
        """
        super().__init__(name)
        self.predicate_fn = predicate_fn
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        x = signal.at(t)
        return self.predicate_fn(x) >= 0
    
    def robustness(self, signal: Signal, t: float) -> float:
        x = signal.at(t)
        return self.predicate_fn(x)
    
    @staticmethod
    def greater_than(index: int, threshold: float, name: str = "") -> STLPredicate:
        """Create predicate x[index] > threshold."""
        return STLPredicate(
            lambda x: x[index] - threshold if len(x) > index else -float('inf'),
            name or f"x[{index}] > {threshold}"
        )
    
    @staticmethod
    def less_than(index: int, threshold: float, name: str = "") -> STLPredicate:
        """Create predicate x[index] < threshold."""
        return STLPredicate(
            lambda x: threshold - x[index] if len(x) > index else -float('inf'),
            name or f"x[{index}] < {threshold}"
        )
    
    @staticmethod
    def in_range(index: int, low: float, high: float, name: str = "") -> STLPredicate:
        """Create predicate low < x[index] < high."""
        return STLPredicate(
            lambda x: min(x[index] - low, high - x[index]) if len(x) > index else -float('inf'),
            name or f"{low} < x[{index}] < {high}"
        )


class STLNot(STLFormula):
    """Negation: NOT phi."""
    
    def __init__(self, formula: STLFormula):
        super().__init__(f"NOT({formula.name})")
        self.formula = formula
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        return not self.formula.evaluate(signal, t)
    
    def robustness(self, signal: Signal, t: float) -> float:
        return -self.formula.robustness(signal, t)


class STLAnd(STLFormula):
    """Conjunction: phi1 AND phi2."""
    
    def __init__(self, *formulas: STLFormula):
        names = [f.name for f in formulas]
        super().__init__(f"AND({', '.join(names)})")
        self.formulas = list(formulas)
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        return all(f.evaluate(signal, t) for f in self.formulas)
    
    def robustness(self, signal: Signal, t: float) -> float:
        if not self.formulas:
            return float('inf')
        return min(f.robustness(signal, t) for f in self.formulas)


class STLOr(STLFormula):
    """Disjunction: phi1 OR phi2."""
    
    def __init__(self, *formulas: STLFormula):
        names = [f.name for f in formulas]
        super().__init__(f"OR({', '.join(names)})")
        self.formulas = list(formulas)
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        return any(f.evaluate(signal, t) for f in self.formulas)
    
    def robustness(self, signal: Signal, t: float) -> float:
        if not self.formulas:
            return float('-inf')
        return max(f.robustness(signal, t) for f in self.formulas)


class STLImplies(STLFormula):
    """Implication: phi1 => phi2 (equivalent to NOT(phi1) OR phi2)."""
    
    def __init__(self, antecedent: STLFormula, consequent: STLFormula):
        super().__init__(f"{antecedent.name} => {consequent.name}")
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        return not self.antecedent.evaluate(signal, t) or self.consequent.evaluate(signal, t)
    
    def robustness(self, signal: Signal, t: float) -> float:
        return max(-self.antecedent.robustness(signal, t), 
                   self.consequent.robustness(signal, t))


class STLAlways(STLFormula):
    """
    Globally/Always: G[a,b] phi
    
    phi must hold for all time points in [t+a, t+b].
    """
    
    def __init__(
        self,
        formula: STLFormula,
        interval: Tuple[float, float] = (0, float('inf'))
    ):
        a, b = interval
        super().__init__(f"G[{a},{b}]({formula.name})")
        self.formula = formula
        self.interval = interval
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        # Check at all time points in interval
        for ts in signal.timestamps:
            if t_start <= ts <= t_end:
                if not self.formula.evaluate(signal, ts):
                    return False
        
        return True
    
    def robustness(self, signal: Signal, t: float) -> float:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        if len(signal) == 0:
            return float('-inf')
        
        # Minimum robustness over interval
        min_rob = float('inf')
        found_point = False
        
        for ts in signal.timestamps:
            if t_start <= ts <= t_end:
                rob = self.formula.robustness(signal, ts)
                min_rob = min(min_rob, rob)
                found_point = True
        
        return min_rob if found_point else float('-inf')


class STLEventually(STLFormula):
    """
    Eventually: F[a,b] phi
    
    phi must hold for at least one time point in [t+a, t+b].
    """
    
    def __init__(
        self,
        formula: STLFormula,
        interval: Tuple[float, float] = (0, float('inf'))
    ):
        a, b = interval
        super().__init__(f"F[{a},{b}]({formula.name})")
        self.formula = formula
        self.interval = interval
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        for ts in signal.timestamps:
            if t_start <= ts <= t_end:
                if self.formula.evaluate(signal, ts):
                    return True
        
        return False
    
    def robustness(self, signal: Signal, t: float) -> float:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        if len(signal) == 0:
            return float('-inf')
        
        # Maximum robustness over interval
        max_rob = float('-inf')
        
        for ts in signal.timestamps:
            if t_start <= ts <= t_end:
                rob = self.formula.robustness(signal, ts)
                max_rob = max(max_rob, rob)
        
        return max_rob


class STLUntil(STLFormula):
    """
    Until: phi1 U[a,b] phi2
    
    phi1 must hold until phi2 becomes true (within [t+a, t+b]).
    """
    
    def __init__(
        self,
        formula1: STLFormula,
        formula2: STLFormula,
        interval: Tuple[float, float] = (0, float('inf'))
    ):
        a, b = interval
        super().__init__(f"({formula1.name}) U[{a},{b}] ({formula2.name})")
        self.formula1 = formula1
        self.formula2 = formula2
        self.interval = interval
    
    def evaluate(self, signal: Signal, t: float) -> bool:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        for i, ts in enumerate(signal.timestamps):
            if t_start <= ts <= t_end:
                if self.formula2.evaluate(signal, ts):
                    # Check phi1 holds for all prior points
                    phi1_holds = True
                    for prior_ts in signal.timestamps[:i]:
                        if t <= prior_ts < ts:
                            if not self.formula1.evaluate(signal, prior_ts):
                                phi1_holds = False
                                break
                    if phi1_holds:
                        return True
        
        return False
    
    def robustness(self, signal: Signal, t: float) -> float:
        a, b = self.interval
        t_start = t + a
        t_end = min(t + b, signal.timestamps[-1]) if len(signal) > 0 else t + b
        
        if len(signal) == 0:
            return float('-inf')
        
        max_rob = float('-inf')
        
        for i, ts in enumerate(signal.timestamps):
            if t_start <= ts <= t_end:
                rob2 = self.formula2.robustness(signal, ts)
                
                # Minimum of phi1 robustness up to ts
                min_rob1 = float('inf')
                for prior_ts in signal.timestamps[:i]:
                    if t <= prior_ts < ts:
                        rob1 = self.formula1.robustness(signal, prior_ts)
                        min_rob1 = min(min_rob1, rob1)
                
                if min_rob1 == float('inf'):
                    min_rob1 = 0  # No prior points
                
                rob = min(rob2, min_rob1)
                max_rob = max(max_rob, rob)
        
        return max_rob


class STLMonitor:
    """
    Online STL monitor for runtime verification.
    
    Monitors multiple STL formulas and tracks their satisfaction
    over streaming signal data.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize STL monitor.
        
        Args:
            buffer_size: Maximum signal buffer size
        """
        self.buffer_size = buffer_size
        
        # Registered formulas
        self._formulas: Dict[str, STLFormula] = {}
        
        # Signal buffer
        self._timestamps: List[float] = []
        self._values: List[np.ndarray] = []
        
        # Monitoring results
        self._results: Dict[str, List[RobustnessResult]] = {}
        
        # Alarm callbacks
        self._alarm_callbacks: List[Callable[[str, RobustnessResult], None]] = []
        
        # Alarm thresholds
        self._alarm_thresholds: Dict[str, float] = {}
    
    def register_formula(
        self,
        name: str,
        formula: STLFormula,
        alarm_threshold: float = 0.0
    ) -> None:
        """
        Register an STL formula for monitoring.
        
        Args:
            name: Formula name/identifier
            formula: STL formula to monitor
            alarm_threshold: Robustness threshold for alarms (default: 0)
        """
        self._formulas[name] = formula
        self._results[name] = []
        self._alarm_thresholds[name] = alarm_threshold
        
        logger.info(f"Registered STL formula: {name}")
    
    def unregister_formula(self, name: str) -> None:
        """Unregister a formula."""
        self._formulas.pop(name, None)
        self._results.pop(name, None)
        self._alarm_thresholds.pop(name, None)
    
    def register_alarm_callback(
        self,
        callback: Callable[[str, RobustnessResult], None]
    ) -> None:
        """Register callback for alarm events."""
        self._alarm_callbacks.append(callback)
    
    def update(self, timestamp: float, value: np.ndarray) -> Dict[str, RobustnessResult]:
        """
        Update monitor with new data point.
        
        Args:
            timestamp: Current time
            value: Signal value (numpy array)
            
        Returns:
            Dictionary of formula_name -> robustness result
        """
        # Add to buffer
        self._timestamps.append(timestamp)
        self._values.append(value)
        
        # Trim buffer if needed
        if len(self._timestamps) > self.buffer_size:
            self._timestamps = self._timestamps[-self.buffer_size:]
            self._values = self._values[-self.buffer_size:]
        
        # Create signal
        signal = Signal(
            timestamps=np.array(self._timestamps),
            values=np.array(self._values)
        )
        
        # Evaluate all formulas
        results = {}
        
        for name, formula in self._formulas.items():
            result = formula.get_robustness_result(signal, timestamp)
            result.formula_name = name
            
            self._results[name].append(result)
            results[name] = result
            
            # Check alarm
            threshold = self._alarm_thresholds.get(name, 0.0)
            if result.value < threshold:
                self._trigger_alarm(name, result)
        
        return results
    
    def _trigger_alarm(self, formula_name: str, result: RobustnessResult) -> None:
        """Trigger alarm callbacks."""
        for callback in self._alarm_callbacks:
            callback(formula_name, result)
    
    def get_signal(self) -> Signal:
        """Get current signal buffer."""
        return Signal(
            timestamps=np.array(self._timestamps),
            values=np.array(self._values)
        )
    
    def get_results(self, formula_name: str) -> List[RobustnessResult]:
        """Get monitoring results for a formula."""
        return self._results.get(formula_name, [])
    
    def get_latest_result(self, formula_name: str) -> Optional[RobustnessResult]:
        """Get most recent result for a formula."""
        results = self._results.get(formula_name, [])
        return results[-1] if results else None
    
    def get_satisfaction_ratio(self, formula_name: str) -> float:
        """Get satisfaction ratio over monitored period."""
        results = self._results.get(formula_name, [])
        if not results:
            return 0.0
        
        satisfied = sum(1 for r in results if r.satisfied)
        return satisfied / len(results)
    
    def get_min_robustness(self, formula_name: str) -> float:
        """Get minimum robustness over monitored period."""
        results = self._results.get(formula_name, [])
        if not results:
            return float('inf')
        
        return min(r.value for r in results)
    
    def clear_buffer(self) -> None:
        """Clear signal buffer and results."""
        self._timestamps = []
        self._values = []
        for name in self._results:
            self._results[name] = []
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get monitoring statistics for all formulas."""
        stats = {}
        
        for name in self._formulas:
            results = self._results.get(name, [])
            
            if results:
                robustness_values = [r.value for r in results]
                stats[name] = {
                    "num_evaluations": len(results),
                    "satisfaction_ratio": self.get_satisfaction_ratio(name),
                    "min_robustness": min(robustness_values),
                    "max_robustness": max(robustness_values),
                    "mean_robustness": np.mean(robustness_values),
                    "std_robustness": np.std(robustness_values),
                }
            else:
                stats[name] = {"num_evaluations": 0}
        
        return stats


# Common safety formulas for heterogeneous fleets

def create_separation_formula(
    agent_index: int,
    min_distance: float,
    name: str = ""
) -> STLPredicate:
    """
    Create formula for minimum separation distance.
    
    Assumes signal has distance to nearest neighbor at given index.
    """
    return STLPredicate.greater_than(
        agent_index,
        min_distance,
        name or f"separation >= {min_distance}"
    )


def create_altitude_bounds_formula(
    altitude_index: int,
    min_alt: float,
    max_alt: float,
    name: str = ""
) -> STLPredicate:
    """Create formula for altitude bounds."""
    return STLPredicate.in_range(
        altitude_index,
        min_alt,
        max_alt,
        name or f"altitude in [{min_alt}, {max_alt}]"
    )


def create_geofence_formula(
    x_index: int,
    y_index: int,
    bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
    name: str = ""
) -> STLAnd:
    """Create formula for geofence bounds."""
    x_min, x_max, y_min, y_max = bounds
    
    return STLAnd(
        STLPredicate.in_range(x_index, x_min, x_max, f"x in bounds"),
        STLPredicate.in_range(y_index, y_min, y_max, f"y in bounds")
    )


def create_velocity_limit_formula(
    speed_index: int,
    max_speed: float,
    name: str = ""
) -> STLPredicate:
    """Create formula for velocity limit."""
    return STLPredicate.less_than(
        speed_index,
        max_speed,
        name or f"speed <= {max_speed}"
    )


def create_always_safe_formula(
    safety_formula: STLFormula,
    horizon: float,
    name: str = ""
) -> STLAlways:
    """Create formula for always maintaining safety over horizon."""
    return STLAlways(
        safety_formula,
        interval=(0, horizon)
    )
