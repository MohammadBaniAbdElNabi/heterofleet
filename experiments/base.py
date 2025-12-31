"""
Base Experiment Class for HeteroFleet.

Provides common infrastructure for experiments.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger

from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.digital_twin.fleet_twin import FleetTwin


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    
    name: str = "experiment"
    description: str = ""
    
    # Repetitions
    num_runs: int = 10
    
    # Duration
    duration: float = 60.0  # seconds
    
    # Random seed
    seed: Optional[int] = None
    
    # Output
    output_dir: str = "./results"
    save_trajectories: bool = True


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    
    run_id: int
    config: Dict[str, Any]
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Raw data
    trajectory_data: Optional[Dict[str, Any]] = None
    time_series: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "duration": self.duration,
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ExperimentSummary:
    """Summary statistics for experiment."""
    
    name: str
    num_runs: int
    successful_runs: int
    
    # Aggregated metrics
    metrics_mean: Dict[str, float] = field(default_factory=dict)
    metrics_std: Dict[str, float] = field(default_factory=dict)
    metrics_min: Dict[str, float] = field(default_factory=dict)
    metrics_max: Dict[str, float] = field(default_factory=dict)
    
    total_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_runs": self.num_runs,
            "successful_runs": self.successful_runs,
            "metrics_mean": self.metrics_mean,
            "metrics_std": self.metrics_std,
            "metrics_min": self.metrics_min,
            "metrics_max": self.metrics_max,
            "total_duration": self.total_duration,
        }


class ExperimentBase(ABC):
    """
    Base class for experiments.
    
    Provides common infrastructure for running and analyzing experiments.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self._results: List[ExperimentResult] = []
        self._summary: Optional[ExperimentSummary] = None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def setup(self, run_id: int) -> SimulationEngine:
        """
        Set up experiment for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Configured simulation engine
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, engine: SimulationEngine) -> Dict[str, float]:
        """
        Compute experiment-specific metrics.
        
        Args:
            engine: Simulation engine after run
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def run_single(self, run_id: int) -> ExperimentResult:
        """Run a single experiment trial."""
        logger.info(f"Starting run {run_id} of {self.config.name}")
        
        result = ExperimentResult(
            run_id=run_id,
            config=self._get_run_config(run_id),
        )
        
        try:
            # Set random seed
            if self.config.seed is not None:
                np.random.seed(self.config.seed + run_id)
            
            # Setup
            result.start_time = time.time()
            engine = self.setup(run_id)
            
            # Run simulation
            engine.run(duration=self.config.duration)
            
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            
            # Compute metrics
            result.metrics = self.compute_metrics(engine)
            
            # Save trajectories
            if self.config.save_trajectories:
                result.trajectory_data = engine.get_all_trajectories()
            
            result.success = True
            logger.info(f"Run {run_id} completed in {result.duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            result.success = False
            result.error = str(e)
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
        
        return result
    
    def run_all(self) -> ExperimentSummary:
        """Run all experiment trials."""
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Configuration: {self.config.num_runs} runs, {self.config.duration}s each")
        
        self._results = []
        
        for run_id in range(self.config.num_runs):
            result = self.run_single(run_id)
            self._results.append(result)
        
        # Compute summary
        self._summary = self._compute_summary()
        
        # Save results
        self._save_results()
        
        logger.info(f"Experiment completed: {self._summary.successful_runs}/{self._summary.num_runs} successful")
        
        return self._summary
    
    def _get_run_config(self, run_id: int) -> Dict[str, Any]:
        """Get configuration for a specific run."""
        return {
            "name": self.config.name,
            "run_id": run_id,
            "duration": self.config.duration,
            "seed": self.config.seed + run_id if self.config.seed else None,
        }
    
    def _compute_summary(self) -> ExperimentSummary:
        """Compute summary statistics from results."""
        successful_results = [r for r in self._results if r.success]
        
        summary = ExperimentSummary(
            name=self.config.name,
            num_runs=len(self._results),
            successful_runs=len(successful_results),
            total_duration=sum(r.duration for r in self._results),
        )
        
        if not successful_results:
            return summary
        
        # Aggregate metrics
        all_metrics: Dict[str, List[float]] = {}
        
        for result in successful_results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        for key, values in all_metrics.items():
            summary.metrics_mean[key] = np.mean(values)
            summary.metrics_std[key] = np.std(values)
            summary.metrics_min[key] = np.min(values)
            summary.metrics_max[key] = np.max(values)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to files."""
        output_dir = Path(self.config.output_dir)
        
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        # Save summary
        summary_file = output_dir / f"{self.config.name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy(self._summary.to_dict()), f, indent=2)
        
        # Save individual results
        results_file = output_dir / f"{self.config.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump([convert_numpy(r.to_dict()) for r in self._results], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_results(self) -> List[ExperimentResult]:
        """Get all results."""
        return self._results
    
    def get_summary(self) -> Optional[ExperimentSummary]:
        """Get summary."""
        return self._summary
