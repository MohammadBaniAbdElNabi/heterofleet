# HeteroFleet

**Heterogeneous Autonomous Vehicle Swarm Coordination Framework**

A comprehensive Python framework for coordinating heterogeneous fleets of autonomous vehicles (UAVs + UGVs) with multi-objective optimization, safety guarantees, and digital twin integration.

## Features

### Core Capabilities
- **Multi-Platform Support**: Aerial (Crazyflie, DJI) and ground vehicles (MentorPi, Jackal)
- **HAIM Coordination**: Heterogeneous Agent Interaction Model for collision-free navigation
- **Multi-Objective Planning**: NSGA-III optimizer with MOPOTA task allocation
- **Safety Guarantees**: STL monitoring, CBF-based safety filters, collision avoidance
- **Digital Twins**: Hierarchical twin architecture (agent → fleet → mission)
- **Real-Time Simulation**: High-fidelity physics with coordination integration

### Architecture
```
heterofleet/
├── core/           # Platform specs, state, messaging
├── coordination/   # HAIM interaction model
├── planning/       # NSGA-III, MOPOTA, scheduling
├── safety/         # STL monitor, collision avoidance, certificates
├── digital_twin/   # Agent, fleet, mission twins
├── simulation/     # Environment, agent dynamics, engine
├── ai/             # LLM interpreter, GNN coordinator
├── communication/  # Mesh networking, adaptive routing
├── hardware/       # Crazyflie, MentorPi interfaces
└── visualization/  # 3D viewer, dashboard
```

## Quick Start

```python
from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

# Create simulation
config = SimulationConfig(
    time_step=0.05,
    num_uavs=3,
    num_ugvs=1,
    enable_coordination=True,
)

engine = SimulationEngine(config)
engine.setup_default_scenario()

# Set targets
engine.set_agent_target("uav_0", Vector3(10, 0, 2))
engine.set_agent_target("ugv_0", Vector3(5, -5, 0))

# Run simulation
engine.run(duration=30.0)

# Get results
for aid, agent in engine._agents.items():
    print(f"{aid}: {agent.position}")
```

## Demo

```bash
python main.py demo
```

## Module Summary

| Module | Description | Lines |
|--------|-------------|-------|
| Core | Platform specs, state, messaging | ~3,000 |
| Coordination | HAIM interaction model | ~700 |
| Planning | NSGA-III, MOPOTA, scheduler | ~2,500 |
| Safety | STL, collision, certificates | ~2,500 |
| Digital Twin | Agent/fleet/mission twins | ~2,500 |
| Simulation | Environment, dynamics, engine | ~2,000 |
| AI | LLM, GNN, anomaly detection | ~1,500 |
| Communication | Protocol, routing | ~1,200 |
| Hardware | Crazyflie, MentorPi | ~800 |
| Visualization | Dashboard, viewer | ~800 |
| Experiments | 5 evaluation scenarios | ~800 |

**Total: ~20,000+ lines**

## Key Components

### HAIM Coordination
The Heterogeneous Agent Interaction Model provides:
- Repulsion forces for collision avoidance
- Friction forces for velocity alignment
- Priority-based conflict resolution
- Network and energy constraints

### Multi-Objective Planning
- **NSGA-III**: Non-dominated sorting genetic algorithm
- **MOPOTA**: Multi-objective parallel optimal task allocation
- **Objectives**: Time, energy, workload balance, safety margins

### Safety Layer
- **STL Monitoring**: Signal Temporal Logic specifications
- **CBF Filters**: Control Barrier Function safety constraints
- **Collision Avoidance**: CPA-based prediction, velocity obstacles
- **Certificates**: Safety guarantee management

### Digital Twin Architecture
- **Agent Twin**: Individual agent state, prediction, anomaly detection
- **Fleet Twin**: Aggregated metrics, formation tracking
- **Mission Twin**: Task/objective tracking, lifecycle management
- **Synchronizer**: Hierarchical state synchronization

## Experiments

Five validation experiments from the architecture:

1. **Scalability**: Performance vs fleet size (5-100 agents)
2. **Formation**: Formation acquisition and maintenance
3. **Task Allocation**: MOPOTA optimization evaluation
4. **Communication**: Mesh network resilience
5. **Emergency**: Response to agent failures

## Hardware Support

### Crazyflie (Simulated/Real)
- Lighthouse/Flow deck positioning
- Multi-ranger obstacle sensing
- High-level commander interface

### MentorPi (Simulated/Real)
- Differential/mecanum drive
- TCP command interface
- Sensor integration

## Requirements

- Python 3.10+
- NumPy
- Pydantic
- Loguru

Optional:
- cflib (Crazyflie hardware)
- matplotlib (visualization)

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{heterofleet2025,
  title={HeteroFleet: Heterogeneous Autonomous Vehicle Swarm Coordination},
  author={UCF Modeling & Simulation},
  year={2025},
  url={https://github.com/ucf-ms/heterofleet}
}
```
