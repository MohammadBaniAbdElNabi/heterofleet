# HeteroFleet Complete Beginner's Guide

## A Step-by-Step Tutorial for Heterogeneous Autonomous Vehicle Swarm Coordination

**Version 1.0 | December 2024**

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Project Structure](#4-project-structure)
5. [Core Concepts](#5-core-concepts)
6. [Your First Simulation](#6-your-first-simulation)
7. [Understanding Each Module](#7-understanding-each-module)
8. [Hardware Setup: Crazyflie Drones](#8-hardware-setup-crazyflie-drones)
9. [Hardware Setup: MentorPi Ground Vehicles](#9-hardware-setup-mentorpi-ground-vehicles)
10. [Running Mixed Fleet Operations](#10-running-mixed-fleet-operations)
11. [Experiments and Evaluation](#11-experiments-and-evaluation)
12. [Advanced Topics](#12-advanced-topics)
13. [Troubleshooting](#13-troubleshooting)
14. [API Reference](#14-api-reference)

---

# 1. Introduction

## What is HeteroFleet?

HeteroFleet is a Python framework for coordinating **heterogeneous fleets** of autonomous vehicles. "Heterogeneous" means the fleet contains different types of robots working together:

- **UAVs (Unmanned Aerial Vehicles)**: Drones like the Crazyflie that fly
- **UGVs (Unmanned Ground Vehicles)**: Robots like MentorPi that drive on the ground

## Why HeteroFleet?

Traditional swarm systems assume all robots are identical. Real-world missions need different capabilities:

| Scenario | UAV Role | UGV Role |
|----------|----------|----------|
| Search & Rescue | Aerial survey, victim detection | Ground access, supply delivery |
| Warehouse | Inventory scanning | Heavy item transport |
| Agriculture | Crop monitoring | Soil sampling, spraying |
| Security | Perimeter patrol | Access control |

## Key Capabilities

1. **Multi-Agent Coordination**: Robots avoid collisions and work together
2. **Task Allocation**: Automatically assign tasks to the best robot
3. **Safety Guarantees**: Mathematically verified safety constraints
4. **Digital Twins**: Virtual copies of robots for monitoring and prediction
5. **Real Hardware Support**: Works with actual Crazyflie drones and MentorPi robots

---

# 2. Prerequisites

## Required Knowledge

You should be comfortable with:
- Basic Python programming (variables, functions, classes)
- Command line / terminal usage
- Basic understanding of 3D coordinates (x, y, z)

## Hardware Requirements

### For Simulation Only
- Computer with Python 3.10+
- 4GB RAM minimum
- Any operating system (Windows, macOS, Linux)

### For Real Hardware
- **Crazyflie 2.1 drone(s)** (~$180 each)
- **Crazyradio PA USB dongle** (~$30)
- **Lighthouse V2 base stations** (2x, ~$150 each) OR Flow deck V2 (~$50)
- **MentorPi robot(s)** (varies by model)
- Computer with USB ports
- Linux recommended for hardware (Ubuntu 22.04+)

## Software Requirements

- Python 3.10 or newer
- pip (Python package manager)
- Git (optional, for version control)

---

# 3. Installation

## Step 1: Create Project Directory

Open your terminal and run:

```bash
# Create a workspace
mkdir ~/heterofleet_workspace
cd ~/heterofleet_workspace
```

## Step 2: Set Up Python Virtual Environment

Virtual environments keep your project dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate

# You should see (venv) in your prompt
```

## Step 3: Extract HeteroFleet

```bash
# Assuming you downloaded heterofleet.zip
unzip heterofleet.zip -d heterofleet
cd heterofleet
```

## Step 4: Install Dependencies

```bash
# Install required packages
pip install numpy pydantic loguru

# For visualization (optional)
pip install matplotlib pillow

# For Crazyflie hardware (optional)
pip install cflib

# For development
pip install pytest black
```

## Step 5: Verify Installation

```bash
# Run the test suite
python main.py test
```

You should see:
```
Testing HeteroFleet module imports...
  ✓ Core Platform
  ✓ Core State
  ✓ Core Message
  ... (all modules should pass)

Results: 18 passed, 0 failed
```

## Step 6: Run Demo

```bash
python main.py demo
```

You should see a simulation run with 4 robots navigating to targets.

---

# 4. Project Structure

Understanding the project layout helps you navigate and modify the code:

```
heterofleet/
│
├── main.py                 # Entry point for demos and tests
├── README.md               # Project overview
├── pyproject.toml          # Package configuration
│
├── heterofleet/            # Main package
│   │
│   ├── core/               # Fundamental building blocks
│   │   ├── platform.py     # Robot specifications (size, speed, etc.)
│   │   ├── state.py        # Robot state (position, velocity, battery)
│   │   ├── message.py      # Communication messages
│   │   └── agent.py        # Base agent class
│   │
│   ├── coordination/       # How robots work together
│   │   ├── haim.py         # Main coordination algorithm
│   │   ├── repulsion.py    # Collision avoidance forces
│   │   ├── friction.py     # Velocity alignment
│   │   └── priority.py     # Who goes first in conflicts
│   │
│   ├── planning/           # Task assignment and scheduling
│   │   ├── task.py         # Task definitions
│   │   ├── nsga3.py        # Multi-objective optimizer
│   │   ├── mopota.py       # Task allocator
│   │   └── scheduler.py    # Temporal scheduling
│   │
│   ├── safety/             # Safety guarantees
│   │   ├── collision.py    # Collision detection/avoidance
│   │   ├── stl_monitor.py  # Safety specification monitoring
│   │   └── certificates.py # Safety certificates
│   │
│   ├── digital_twin/       # Virtual robot copies
│   │   ├── agent_twin.py   # Individual robot twin
│   │   ├── fleet_twin.py   # Fleet-level twin
│   │   └── mission_twin.py # Mission tracking
│   │
│   ├── simulation/         # Simulated environment
│   │   ├── environment.py  # World with obstacles
│   │   ├── agent_sim.py    # Simulated robot physics
│   │   └── engine.py       # Simulation controller
│   │
│   ├── hardware/           # Real robot interfaces
│   │   ├── crazyflie.py    # Crazyflie drone interface
│   │   └── mentorpi.py     # MentorPi robot interface
│   │
│   ├── ai/                 # Intelligent components
│   │   ├── llm_interpreter.py  # Natural language commands
│   │   └── gnn_coordinator.py  # Graph neural network
│   │
│   ├── communication/      # Robot-to-robot communication
│   │   ├── protocol.py     # Message protocols
│   │   └── routing.py      # Mesh networking
│   │
│   └── visualization/      # Display and monitoring
│       ├── viewer.py       # 3D visualization
│       └── dashboard.py    # Status dashboard
│
└── experiments/            # Evaluation experiments
    ├── scalability.py      # Test with many robots
    ├── formation.py        # Formation control
    └── ...
```

---

# 5. Core Concepts

Before diving into code, let's understand the key concepts:

## 5.1 Platforms (Robot Types)

A **Platform** defines a type of robot with its physical properties:

```python
from heterofleet.core.platform import PlatformType, PlatformSpecification

# Available platform types
PlatformType.MICRO_UAV    # Tiny drones like Crazyflie (~30g)
PlatformType.SMALL_UAV    # Small drones like DJI Mini (~250g)
PlatformType.MEDIUM_UAV   # Medium drones (~3kg)
PlatformType.SMALL_UGV    # Small ground robots like MentorPi
PlatformType.MEDIUM_UGV   # Medium ground robots
```

Each platform has specifications:
- **Mass**: How heavy (affects dynamics)
- **Dimensions**: Physical size
- **Max Velocity**: How fast it can move
- **Battery Capacity**: Energy storage
- **Sensor Range**: How far it can "see"

## 5.2 Agents (Individual Robots)

An **Agent** is a specific robot instance:

```python
# Each agent has:
# - Unique ID: "uav_0", "ugv_1", etc.
# - Platform type: What kind of robot
# - State: Current position, velocity, battery, etc.
```

## 5.3 State

**State** represents everything about a robot at a moment in time:

```python
# Position: Where is it? (x, y, z in meters)
position = (5.0, 3.0, 1.0)  # 5m right, 3m forward, 1m up

# Velocity: How fast is it moving? (m/s)
velocity = (0.5, 0.0, 0.0)  # Moving right at 0.5 m/s

# Battery: How much energy left? (0-100%)
battery_level = 0.85  # 85% remaining

# Mode: What is it doing?
# IDLE, NAVIGATING, EXECUTING, EMERGENCY, etc.
```

## 5.4 Coordinate System

HeteroFleet uses a **right-handed coordinate system**:

```
        Z (up)
        │
        │
        │
        └───────── Y (forward)
       /
      /
     X (right)
```

- **X**: Positive = right, Negative = left
- **Y**: Positive = forward, Negative = backward  
- **Z**: Positive = up, Negative = down
- **Units**: Meters (m) for position, meters/second (m/s) for velocity

## 5.5 Vector3

The `Vector3` class represents 3D points and directions:

```python
from heterofleet.core.platform import Vector3

# Create a position
pos = Vector3(x=5.0, y=3.0, z=1.0)

# Access components
print(pos.x)  # 5.0
print(pos.y)  # 3.0
print(pos.z)  # 1.0

# Vector math
pos2 = Vector3(1.0, 1.0, 0.0)
diff = pos - pos2  # Vector3(4.0, 2.0, 1.0)
distance = diff.norm()  # Length of vector

# Normalize (make length = 1)
direction = diff.normalize()
```

---

# 6. Your First Simulation

Let's build up from simple to complex examples.

## 6.1 Hello HeteroFleet

Create a file `tutorial_01_hello.py`:

```python
"""
Tutorial 1: Hello HeteroFleet
Create a simple simulation with one drone.
"""

# Import the modules we need
from heterofleet.core.platform import Vector3, PlatformType, PlatformSpecification
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

# Step 1: Create simulation configuration
config = SimulationConfig(
    time_step=0.05,          # Simulate in 50ms steps
    real_time_factor=1.0,    # Run at real-time speed (set to inf for fast)
    max_duration=30.0,       # Maximum 30 seconds
    num_uavs=1,              # One drone
    num_ugvs=0,              # No ground robots
    enable_coordination=False,  # Simple mode first
)

# Step 2: Create the simulation engine
engine = SimulationEngine(config)

# Step 3: Set up the default scenario (creates robots in a pattern)
engine.setup_default_scenario()

# Step 4: See what we created
print("Created agents:")
for agent_id, agent in engine._agents.items():
    pos = agent.position
    print(f"  {agent_id}: position=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

# Step 5: Give the drone a target
target = Vector3(10.0, 5.0, 2.0)  # 10m right, 5m forward, 2m up
engine.set_agent_target("uav_0", target)
print(f"\nTarget set to: ({target.x}, {target.y}, {target.z})")

# Step 6: Run the simulation
print("\nRunning simulation...")
engine.run(duration=10.0)

# Step 7: Check the results
print("\nSimulation complete!")
final_pos = engine._agents["uav_0"].position
print(f"Final position: ({final_pos.x:.2f}, {final_pos.y:.2f}, {final_pos.z:.2f})")

# Calculate distance to target
distance = ((final_pos.x - target.x)**2 + 
            (final_pos.y - target.y)**2 + 
            (final_pos.z - target.z)**2) ** 0.5
print(f"Distance to target: {distance:.2f}m")
```

Run it:
```bash
python tutorial_01_hello.py
```

## 6.2 Multiple Robots

Create `tutorial_02_multi_robot.py`:

```python
"""
Tutorial 2: Multiple Robots
Coordinate a mixed fleet of drones and ground robots.
"""

from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

# Create a mixed fleet
config = SimulationConfig(
    time_step=0.05,
    real_time_factor=float('inf'),  # Run as fast as possible
    max_duration=60.0,
    num_uavs=3,                      # Three drones
    num_ugvs=2,                      # Two ground robots
    enable_coordination=True,        # Enable collision avoidance
)

engine = SimulationEngine(config)
engine.setup_default_scenario()

# Print initial positions
print("Initial fleet configuration:")
print("-" * 50)
for agent_id, agent in engine._agents.items():
    ptype = agent.platform_spec.platform_type.name
    pos = agent.position
    print(f"  {agent_id} ({ptype}): ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")

# Assign targets to each robot
targets = {
    "uav_0": Vector3(15, 0, 3),      # Drone 0 goes right and up
    "uav_1": Vector3(-10, 10, 2),    # Drone 1 goes left-forward
    "uav_2": Vector3(0, -15, 4),     # Drone 2 goes backward and high
    "ugv_0": Vector3(8, 8, 0),       # Ground robot 0 goes diagonal
    "ugv_1": Vector3(-8, -8, 0),     # Ground robot 1 goes opposite
}

print("\nAssigning targets:")
for agent_id, target in targets.items():
    if agent_id in engine._agents:
        engine.set_agent_target(agent_id, target)
        print(f"  {agent_id} -> ({target.x}, {target.y}, {target.z})")

# Run simulation
print("\nSimulating for 30 seconds...")
engine.run(duration=30.0)

# Results
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

state = engine.state
print(f"Simulation time: {state.sim_time:.1f}s")
print(f"Computation time: {state.wall_time:.2f}s")
print(f"Speed: {state.sim_time/state.wall_time:.1f}x real-time")

print("\nFinal positions and distances to targets:")
for agent_id, agent in engine._agents.items():
    pos = agent.position
    target = targets.get(agent_id, Vector3(0, 0, 0))
    dist = ((pos.x - target.x)**2 + (pos.y - target.y)**2 + (pos.z - target.z)**2) ** 0.5
    battery = agent.battery_level * 100
    
    status = "✓ Reached" if dist < 1.0 else f"✗ {dist:.1f}m away"
    print(f"  {agent_id}: {status}, Battery: {battery:.0f}%")

# Fleet metrics
metrics = engine.fleet_twin.metrics
print(f"\nFleet Statistics:")
print(f"  Total agents: {metrics.total_agents}")
print(f"  Average battery: {metrics.avg_battery_level:.1%}")
print(f"  Minimum battery: {metrics.min_battery_level:.1%}")
```

## 6.3 Adding Obstacles

Create `tutorial_03_obstacles.py`:

```python
"""
Tutorial 3: Navigation with Obstacles
Robots must avoid obstacles while reaching their targets.
"""

from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.simulation.environment import Obstacle, ObstacleType

# Setup
config = SimulationConfig(
    time_step=0.05,
    real_time_factor=float('inf'),
    num_uavs=2,
    num_ugvs=1,
    enable_coordination=True,
    enable_collision_avoidance=True,
)

engine = SimulationEngine(config)
engine.setup_default_scenario()

# Add obstacles to the environment
obstacles = [
    # Building in the center
    Obstacle(
        obstacle_id="building_1",
        obstacle_type=ObstacleType.BUILDING,
        position=Vector3(0, 5, 2.5),
        dimensions=Vector3(4, 4, 5),  # 4x4x5 meter building
    ),
    # Another obstacle
    Obstacle(
        obstacle_id="wall_1",
        obstacle_type=ObstacleType.STATIC,
        position=Vector3(5, 0, 1),
        dimensions=Vector3(1, 10, 2),  # Long wall
    ),
]

for obs in obstacles:
    engine.environment.add_obstacle(obs)
    print(f"Added obstacle: {obs.obstacle_id} at {obs.position}")

# Set targets that require navigation around obstacles
targets = {
    "uav_0": Vector3(0, 12, 2),   # Behind the building
    "uav_1": Vector3(10, 5, 3),   # Past the wall
    "ugv_0": Vector3(8, 8, 0),    # Around obstacles
}

for agent_id, target in targets.items():
    if agent_id in engine._agents:
        engine.set_agent_target(agent_id, target)

# Run
print("\nSimulating navigation around obstacles...")
engine.run(duration=30.0)

# Check results
print("\nResults:")
for agent_id, agent in engine._agents.items():
    pos = agent.position
    target = targets.get(agent_id, Vector3(0, 0, 0))
    dist = ((pos.x - target.x)**2 + (pos.y - target.y)**2 + (pos.z - target.z)**2) ** 0.5
    print(f"  {agent_id}: Final pos ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}), "
          f"Distance to goal: {dist:.1f}m")
```

## 6.4 Task Allocation

Create `tutorial_04_tasks.py`:

```python
"""
Tutorial 4: Automatic Task Allocation
Let the system decide which robot should do which task.
"""

from heterofleet.core.platform import Vector3, PlatformType
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.planning.task import Task, TaskType, TaskPriority
from heterofleet.planning.mopota import MOPOTAAllocator, AgentCapabilities, ObjectiveWeights

# Setup fleet
config = SimulationConfig(
    time_step=0.05,
    real_time_factor=float('inf'),
    num_uavs=3,
    num_ugvs=2,
)

engine = SimulationEngine(config)
engine.setup_default_scenario()

# Define tasks that need to be done
tasks = [
    Task(
        task_id="survey_north",
        task_type=TaskType.SURVEILLANCE,
        location=Vector3(0, 20, 5),    # High up - needs drone
        duration=30.0,
        priority=TaskPriority.HIGH,
        required_capabilities=["aerial", "camera"],
    ),
    Task(
        task_id="survey_east",
        task_type=TaskType.SURVEILLANCE,
        location=Vector3(15, 0, 3),
        duration=20.0,
        priority=TaskPriority.MEDIUM,
        required_capabilities=["camera"],
    ),
    Task(
        task_id="deliver_package",
        task_type=TaskType.DELIVERY,
        location=Vector3(-10, 10, 0),   # Ground level - needs UGV
        duration=60.0,
        priority=TaskPriority.HIGH,
        required_capabilities=["cargo"],
    ),
    Task(
        task_id="inspect_area",
        task_type=TaskType.INSPECTION,
        location=Vector3(5, -10, 0),
        duration=45.0,
        priority=TaskPriority.LOW,
        required_capabilities=["camera"],
    ),
]

print("Tasks to allocate:")
for task in tasks:
    print(f"  {task.task_id}: {task.task_type.name} at {task.location}, "
          f"Priority: {task.priority.name}")

# Define what each agent can do
capabilities = {}

for agent_id, agent in engine._agents.items():
    pos = agent.position
    ptype = agent.platform_spec.platform_type
    
    if "uav" in agent_id:
        caps = AgentCapabilities(
            agent_id=agent_id,
            platform_type=ptype,
            position=pos,
            max_speed=3.0,
            payload_capacity=0.1,  # Small payload
            sensor_range=10.0,
            energy_level=1.0,
            available_sensors=["camera", "aerial", "surveillance"],
        )
    else:  # UGV
        caps = AgentCapabilities(
            agent_id=agent_id,
            platform_type=ptype,
            position=pos,
            max_speed=0.5,
            payload_capacity=2.0,  # Can carry more
            sensor_range=5.0,
            energy_level=1.0,
            available_sensors=["camera", "cargo", "inspection"],
        )
    
    capabilities[agent_id] = caps

# Run the task allocator
print("\nRunning MOPOTA task allocation...")
allocator = MOPOTAAllocator(
    num_generations=30,
    population_size=30,
)

# Optimize for multiple objectives
weights = ObjectiveWeights(
    completion_rate=1.0,    # Maximize tasks completed
    mission_time=0.8,       # Minimize total time
    energy_consumption=0.5, # Minimize energy use
    workload_balance=0.6,   # Distribute work evenly
)

result = allocator.allocate(tasks, capabilities, weights)

# Show results
print("\nAllocation Results:")
print("-" * 50)
for task_id, agent_id in result.assignments.items():
    if agent_id:
        task = next(t for t in tasks if t.task_id == task_id)
        print(f"  {task_id} -> {agent_id}")
        print(f"    Location: {task.location}")
        print(f"    Type: {task.task_type.name}")
    else:
        print(f"  {task_id} -> UNASSIGNED")

print(f"\nObjectives achieved:")
print(f"  Completion rate: {result.objectives['completion_rate']:.1%}")
print(f"  Mission time: {result.objectives['mission_time']:.1f}s")
print(f"  Energy consumption: {result.objectives['energy_consumption']:.2f}")
print(f"  Workload balance: {result.objectives['workload_balance']:.2f}")
```

---

# 7. Understanding Each Module

## 7.1 Core Module

The foundation of HeteroFleet.

### Platform Specifications

```python
from heterofleet.core.platform import (
    PlatformType,
    PlatformSpecification,
    Vector3
)

# Create a custom platform specification
my_drone = PlatformSpecification.from_platform_type(
    PlatformType.SMALL_UAV,
    platform_id="my_custom_drone"
)

# Access properties
print(f"Max velocity: {my_drone.max_velocity} m/s")
print(f"Battery: {my_drone.battery_capacity_wh} Wh")
print(f"Collision radius: {my_drone.collision_envelope.semi_axes}")
```

### State Management

```python
from heterofleet.core.state import AgentState, AgentMode

# States are typically created by the simulation, but you can inspect them:
state = engine._agents["uav_0"].get_state()

print(f"Agent: {state.agent_id}")
print(f"Position: {state.position}")
print(f"Velocity: {state.velocity}")
print(f"Energy: {state.energy_level:.1%}")
print(f"Mode: {state.mode}")
```

### Messaging

```python
from heterofleet.core.message import Message, MessageType, MessagePriority

# Create a message
msg = Message(
    message_id="cmd_001",
    message_type=MessageType.COMMAND,
    sender_id="ground_station",
    receiver_id="uav_0",
    payload={"action": "return_home"},
    priority=MessagePriority.HIGH,
)

print(f"Message: {msg.message_id} from {msg.sender_id} to {msg.receiver_id}")
```

## 7.2 Coordination Module (HAIM)

The **Heterogeneous Agent Interaction Model** controls how robots interact.

### Key Concepts

1. **Repulsion**: Robots push away from each other to avoid collisions
2. **Friction**: Robots align velocities when close (smooth traffic flow)
3. **Self-Drive**: Robots move toward their goals
4. **Priority**: Higher-priority robots have "right of way"

```python
from heterofleet.coordination.haim import HAIMCoordinator, HAIMParameters

# Create coordinator with custom parameters
params = HAIMParameters(
    repulsion_strength=2.0,     # How strongly robots repel
    repulsion_range=5.0,        # Distance at which repulsion starts
    friction_strength=0.5,      # Velocity alignment strength
    self_drive_strength=1.0,    # Goal-seeking strength
    enable_priority=True,       # Use priority system
)

coordinator = HAIMCoordinator(params)

# The simulation engine uses this automatically when enabled
```

### Visualizing Forces

```python
"""
Understanding the forces:

Robot A wants to go right →
Robot B is in the way

Forces on Robot A:
  Self-drive: →→→ (toward goal)
  Repulsion:  ↑   (away from B)
  Result:     ↗   (diagonal to avoid B)
"""
```

## 7.3 Planning Module

### Tasks

```python
from heterofleet.planning.task import Task, TaskType, TaskStatus, TaskPriority

# Task types available:
# SURVEILLANCE - Monitor an area
# INSPECTION - Detailed examination
# DELIVERY - Transport something
# SEARCH - Find something
# PATROL - Regular monitoring route

task = Task(
    task_id="task_001",
    task_type=TaskType.SURVEILLANCE,
    location=Vector3(10, 10, 5),
    duration=60.0,                    # Takes 60 seconds
    priority=TaskPriority.HIGH,
    deadline=300.0,                   # Must complete within 300s
    required_capabilities=["camera"],
)

# Task lifecycle:
# PENDING -> ASSIGNED -> IN_PROGRESS -> COMPLETED
#                    -> FAILED (if something goes wrong)
```

### NSGA-III Optimizer

Multi-objective optimization using NSGA-III algorithm:

```python
from heterofleet.planning.nsga3 import NSGA3Optimizer

# The optimizer finds solutions that balance multiple objectives
optimizer = NSGA3Optimizer(
    num_objectives=4,
    num_variables=10,
    population_size=100,
    num_generations=50,
)

# Objectives we optimize:
# 1. Minimize mission time
# 2. Minimize energy consumption  
# 3. Maximize task completion
# 4. Maximize workload balance
```

### Scheduler

```python
from heterofleet.planning.scheduler import TemporalScheduler

scheduler = TemporalScheduler()

# Add tasks with time constraints
scheduler.add_task(task1, earliest_start=0.0, latest_end=100.0)
scheduler.add_task(task2, earliest_start=50.0, latest_end=200.0)

# Create schedule
schedule = scheduler.create_schedule(agent_ids=["uav_0", "uav_1"])

# Get schedule for specific agent
agent_schedule = scheduler.get_agent_schedule("uav_0")
for entry in agent_schedule:
    print(f"  {entry.task_id}: {entry.start_time} - {entry.end_time}")
```

## 7.4 Safety Module

### Collision Detection and Avoidance

```python
from heterofleet.safety.collision import (
    CollisionChecker,
    CollisionAvoidance,
    CollisionPrediction
)

# Create checker
checker = CollisionChecker(safety_margin=0.5)  # 0.5m buffer

# Check for collision between two agents
prediction = checker.predict_collision(
    pos_a=Vector3(0, 0, 1),
    vel_a=Vector3(1, 0, 0),    # Moving right
    radius_a=0.2,
    pos_b=Vector3(5, 0, 1),
    vel_b=Vector3(-1, 0, 0),   # Moving left (toward A!)
    radius_b=0.2,
    time_horizon=10.0,
)

if prediction.time_to_collision < float('inf'):
    print(f"⚠️ Collision predicted in {prediction.time_to_collision:.1f}s!")
    print(f"   Severity: {prediction.severity.name}")
else:
    print("✓ No collision predicted")
```

### STL Monitor (Safety Specifications)

**Signal Temporal Logic (STL)** lets you write formal safety rules:

```python
from heterofleet.safety.stl_monitor import STLMonitor, STLSpecification

# Create monitor
monitor = STLMonitor()

# Add safety specifications
# "Always stay above 0.5m altitude"
spec1 = STLSpecification(
    spec_id="min_altitude",
    formula="G(altitude > 0.5)",  # G = "globally" (always)
    variables=["altitude"],
)
monitor.add_specification(spec1)

# "Eventually reach the goal within 60 seconds"
spec2 = STLSpecification(
    spec_id="reach_goal",
    formula="F[0,60](distance_to_goal < 1.0)",  # F = "finally" (eventually)
    variables=["distance_to_goal"],
)
monitor.add_specification(spec2)

# Check specifications during simulation
result = monitor.evaluate(
    spec_id="min_altitude",
    signal={"altitude": [1.0, 0.8, 0.6, 0.4, 0.3]},  # Time series
    time_points=[0, 1, 2, 3, 4],
)

print(f"Specification satisfied: {result.satisfied}")
print(f"Robustness: {result.robustness:.2f}")  # How much margin?
```

## 7.5 Digital Twin Module

Digital twins are virtual copies of real robots, used for monitoring and prediction.

### Agent Twin

```python
from heterofleet.digital_twin.agent_twin import AgentTwin, AgentTwinState

# Twins are created automatically by the simulation engine
twin = engine.fleet_twin.get_agent_twin("uav_0")

# Current state
print(f"Position: {twin.state.position}")
print(f"Battery: {twin.state.battery_level:.1%}")
print(f"Status: {twin.status.name}")

# Predict future state
future_state = twin.predict_state(dt=5.0)  # 5 seconds ahead
print(f"Predicted position in 5s: {future_state.position}")

# Predict trajectory
trajectory = twin.predict_trajectory(horizon=30.0, dt=1.0)
for i, state in enumerate(trajectory):
    print(f"  t={i}s: pos=({state.position.x:.1f}, {state.position.y:.1f})")

# Estimate remaining energy
remaining_energy = twin.estimate_remaining_energy()
remaining_time = twin.estimate_remaining_time()
print(f"Estimated flight time remaining: {remaining_time:.0f}s")
```

### Fleet Twin

```python
# Fleet-level monitoring
fleet_twin = engine.fleet_twin

# Get metrics
metrics = fleet_twin.metrics
print(f"Total agents: {metrics.total_agents}")
print(f"Active agents: {metrics.active_agents}")
print(f"Average battery: {metrics.avg_battery_level:.1%}")
print(f"Minimum battery: {metrics.min_battery_level:.1%}")
print(f"Fleet centroid: {metrics.centroid}")
print(f"Fleet spread: {metrics.spread_radius:.1f}m")

# Find agents
available = fleet_twin.get_available_agents()  # Idle agents
nearby = fleet_twin.get_agents_in_radius(Vector3(0, 0, 0), radius=10.0)
uavs = fleet_twin.get_agents_by_type(PlatformType.SMALL_UAV)
```

### Mission Twin

```python
from heterofleet.digital_twin.mission_twin import MissionTwin, MissionObjective

# Create mission
mission = MissionTwin(
    mission_id="search_rescue_01",
    mission_name="Area Search and Rescue",
)

# Add objectives
mission.add_objective(MissionObjective(
    objective_id="coverage",
    description="Search 90% of area",
    target_value=0.9,
))

# Add tasks
mission.add_task("search_zone_a")
mission.add_task("search_zone_b")

# Assign agents
mission.assign_agent("uav_0")
mission.assign_agent("ugv_0")

# Start mission
mission.start()

# Update progress
mission.update_objective("coverage", current_value=0.45)
mission.update_task_status("search_zone_a", TaskStatus.COMPLETED)

# Check status
print(f"Mission status: {mission.state.status.name}")
print(f"Progress: {mission.state.metrics.completion_rate:.1%}")
```

## 7.6 Simulation Module

### Environment

```python
from heterofleet.simulation.environment import (
    SimulationEnvironment,
    EnvironmentConfig,
    Obstacle,
    ObstacleType,
)

# Create environment
env_config = EnvironmentConfig(
    world_bounds_min=Vector3(-50, -50, 0),
    world_bounds_max=Vector3(50, 50, 20),
    wind_velocity=Vector3(2, 0, 0),  # 2 m/s wind from west
)

environment = SimulationEnvironment(env_config)

# Add obstacles
building = Obstacle(
    obstacle_id="tower",
    obstacle_type=ObstacleType.BUILDING,
    position=Vector3(0, 0, 10),
    dimensions=Vector3(5, 5, 20),
)
environment.add_obstacle(building)

# Query environment
is_valid = environment.is_position_valid(Vector3(0, 0, 5))  # Inside building?
nearby = environment.get_obstacles_near(Vector3(0, 0, 0), radius=20.0)
wind = environment.get_wind_at(Vector3(0, 0, 10))  # Wind varies with height
```

### Agent Dynamics

```python
from heterofleet.simulation.agent_sim import SimulatedAgent, AgentDynamics

# The simulation uses realistic dynamics:
# - Mass and inertia
# - Thrust limits
# - Drag forces
# - Wind effects (for aerial vehicles)
# - Energy consumption

# Dynamics are automatically set based on platform type
# but can be customized:
dynamics = AgentDynamics(
    mass=0.5,           # kg
    max_thrust=15.0,    # Newtons
    drag_coefficient=0.1,
    motor_time_constant=0.1,  # Response time
)
```

## 7.7 Visualization Module

### Dashboard

```python
from heterofleet.visualization.dashboard import Dashboard, DashboardWidget

# Create dashboard
dashboard = Dashboard(title="HeteroFleet Mission Control")

# Connect to fleet twin
dashboard.set_fleet_twin(engine.fleet_twin)

# Update (call periodically)
dashboard.update()

# Print text version
print(dashboard.render_text())

# Get alerts
alerts = dashboard.get_alerts(severity="critical")
for alert in alerts:
    print(f"ALERT: {alert.message}")

# Export to dict (for web UI)
data = dashboard.to_dict()
```

### 3D Viewer

```python
from heterofleet.visualization.viewer import FleetViewer, ViewerConfig

# Create viewer
viewer_config = ViewerConfig(
    width=1280,
    height=720,
    show_grid=True,
    show_trajectories=True,
)

viewer = FleetViewer(viewer_config)

# Add agents
for agent_id, agent in engine._agents.items():
    viewer.add_agent(
        agent_id,
        agent.platform_spec.platform_type,
        agent.position,
    )

# Update from fleet twin
viewer.update_from_fleet_twin(engine.fleet_twin)

# Save frame
viewer.save_frame("fleet_view.png")
```

---

# 8. Hardware Setup: Crazyflie Drones

## 8.1 Hardware Overview

### What You Need

| Item | Purpose | Approximate Cost |
|------|---------|------------------|
| Crazyflie 2.1 | The drone itself | $180 |
| Crazyradio PA | USB radio for communication | $30 |
| Lighthouse Deck | Position tracking (recommended) | $40 |
| Lighthouse V2 Base Stations (2) | Position reference | $300 total |
| Multi-ranger Deck | Obstacle detection | $35 |
| USB charging cable | Charging | Included |
| Spare propellers | Replacements | $5 |

### Alternative: Flow Deck

If Lighthouse is too expensive:
- **Flow Deck V2** ($50): Uses optical flow + height sensor
- Less accurate but works in smaller spaces
- No external setup required

## 8.2 Physical Assembly

### Step 1: Attach Lighthouse Deck

1. Power off the Crazyflie
2. Align the Lighthouse deck with the expansion header
3. Gently press down until it clicks
4. The deck goes on TOP of the Crazyflie

```
     Lighthouse Deck
    ┌──────────────┐
    │  ○      ○    │  <- Sensors
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │   Crazyflie  │
    │   ○      ○   │  <- Motors
    └──────────────┘
```

### Step 2: Attach Multi-ranger Deck (Optional)

1. The Multi-ranger goes BELOW the Crazyflie
2. Use the expansion header on the bottom
3. Provides 5-direction obstacle sensing

### Step 3: Install Propellers

1. Note the two propeller types: CW (clockwise) and CCW
2. CW propellers go on motors M1 and M3 (diagonal)
3. CCW propellers go on motors M2 and M4
4. Push firmly until they click

```
      CCW         CW
       M2 ─────── M1
        │         │
        │    +    │
        │         │
       M3 ─────── M4
      CW          CCW
```

## 8.3 Software Installation

### Install cflib

```bash
# Make sure you're in your virtual environment
pip install cflib
```

### Linux USB Permissions

On Linux, you need to allow access to the Crazyradio:

```bash
# Create udev rules file
sudo nano /etc/udev/rules.d/99-crazyradio.rules
```

Add these lines:
```
# Crazyradio PA
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"

# Crazyflie bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
```

Apply:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add yourself to plugdev group
sudo usermod -a -G plugdev $USER

# Log out and back in for changes to take effect
```

### Verify Connection

```python
# test_crazyflie_connection.py
import cflib.crtp

# Initialize drivers
cflib.crtp.init_drivers()

# Scan for Crazyflies
print("Scanning for Crazyflies...")
available = cflib.crtp.scan_interfaces()

for uri in available:
    print(f"  Found: {uri[0]}")
```

Run:
```bash
python test_crazyflie_connection.py
```

You should see something like:
```
Scanning for Crazyflies...
  Found: radio://0/80/2M/E7E7E7E7E7
```

## 8.4 Lighthouse Setup

### Physical Setup

1. **Mount Base Stations**: Place two Lighthouse V2 base stations diagonally opposite each other
2. **Height**: Mount them 2-3 meters high, angled down ~30°
3. **Power**: Connect to power (they don't need PC connection)
4. **Channel**: Set different channels (A and B) using the mode button

```
Room Layout (top view):

   ┌───────────────────────┐
   │    [BS-A]             │
   │       ↘               │
   │                       │
   │        ✈ Flight       │
   │         Area          │
   │                       │
   │              ↙        │
   │             [BS-B]    │
   └───────────────────────┘
```

### Calibrate Geometry

Use the Crazyflie client to calibrate:

```bash
# Install Crazyflie client
pip install cfclient

# Run it
cfclient
```

1. Connect to your Crazyflie
2. Go to the "Lighthouse" tab
3. Click "Auto-calibrate"
4. Save the geometry

### Alternative: Manual Configuration

Create a geometry file `lighthouse_geometry.yaml`:

```yaml
lighthouse_config:
  - bs_id: 0
    origin: [0.0, 0.0, 0.0]
    rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  - bs_id: 1
    origin: [3.0, 0.0, 3.0]
    rotation: [[0.866, 0, -0.5], [0, 1, 0], [0.5, 0, 0.866]]
```

## 8.5 Using Crazyflie with HeteroFleet

### Basic Connection

```python
"""
Connect to a real Crazyflie drone.
"""
import asyncio
from heterofleet.hardware.crazyflie import CrazyflieInterface, CrazyflieConfig

async def main():
    # Configuration
    config = CrazyflieConfig(
        uri="radio://0/80/2M/E7E7E7E7E7",  # Your drone's URI
        default_height=0.5,  # Hover at 0.5m
        use_lighthouse=True,
        use_multiranger=True,
    )
    
    # Create interface
    cf = CrazyflieInterface("uav_0", config)
    
    # Connect
    print("Connecting to Crazyflie...")
    if await cf.connect():
        print("Connected!")
        
        # Get telemetry
        telemetry = await cf.get_telemetry()
        print(f"Position: {telemetry.position}")
        print(f"Battery: {telemetry.battery_percentage:.0f}%")
        
        # Arm
        await cf.arm()
        
        # Take off
        print("Taking off...")
        await cf.takeoff(altitude=0.5)
        
        # Wait
        await asyncio.sleep(3.0)
        
        # Land
        print("Landing...")
        await cf.land()
        
        # Disarm
        await cf.disarm()
        
        # Disconnect
        await cf.disconnect()
    else:
        print("Failed to connect!")

asyncio.run(main())
```

### Multiple Drones (Swarm)

```python
"""
Control multiple Crazyflie drones.
"""
import asyncio
from heterofleet.hardware.crazyflie import CrazyflieSwarm

async def main():
    # Create swarm manager
    swarm = CrazyflieSwarm()
    
    # Add drones (use your actual URIs)
    swarm.add_drone("uav_0", "radio://0/80/2M/E7E7E7E701")
    swarm.add_drone("uav_1", "radio://0/80/2M/E7E7E7E702")
    swarm.add_drone("uav_2", "radio://0/80/2M/E7E7E7E703")
    
    # Connect to all
    print("Connecting to swarm...")
    results = await swarm.connect_all()
    
    for drone_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {drone_id}: {status}")
    
    # Arm all
    for drone in swarm.get_all_drones():
        await drone.arm()
    
    # Take off all
    print("Swarm takeoff...")
    await swarm.takeoff_all(altitude=0.5)
    
    # Wait
    await asyncio.sleep(5.0)
    
    # Land all
    print("Swarm landing...")
    await swarm.land_all()
    
    # Disconnect
    await swarm.disconnect_all()

asyncio.run(main())
```

### Integration with Simulation Engine

For testing, you can run the simulation engine with real hardware:

```python
"""
HeteroFleet with real Crazyflie (hybrid mode).
"""
import asyncio
from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.hardware.crazyflie import CrazyflieInterface, CrazyflieConfig

async def run_hybrid():
    # Create simulation for coordination logic
    config = SimulationConfig(
        time_step=0.05,
        num_uavs=1,
        num_ugvs=0,
        enable_coordination=True,
    )
    engine = SimulationEngine(config)
    
    # Connect to real drone
    cf_config = CrazyflieConfig(uri="radio://0/80/2M/E7E7E7E7E7")
    cf = CrazyflieInterface("uav_0", cf_config)
    
    if not await cf.connect():
        print("Failed to connect!")
        return
    
    # Main loop
    await cf.arm()
    await cf.takeoff(0.5)
    
    target = Vector3(1.0, 0.0, 0.5)
    
    for _ in range(100):  # 100 iterations
        # Get real telemetry
        telemetry = await cf.get_telemetry()
        
        # Update simulation with real position
        # (This would update the digital twin)
        
        # Compute desired velocity (simple proportional control)
        error = target - telemetry.position
        velocity = Vector3(
            min(0.3, error.x * 0.5),
            min(0.3, error.y * 0.5),
            min(0.2, error.z * 0.5),
        )
        
        # Send to real drone
        await cf.set_velocity(velocity)
        
        await asyncio.sleep(0.05)
    
    await cf.land()
    await cf.disconnect()

asyncio.run(run_hybrid())
```

## 8.6 Safety Considerations

### Pre-Flight Checklist

- [ ] Battery fully charged (4.2V)
- [ ] Propellers secure and undamaged
- [ ] Lighthouse base stations powered and visible
- [ ] Flying area clear of obstacles and people
- [ ] Emergency stop accessible (keyboard 'e' or physical switch)
- [ ] Fire extinguisher nearby (LiPo batteries!)

### Emergency Procedures

```python
# Always have emergency stop ready
async def emergency_handler():
    try:
        # Your flight code here
        pass
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        # ALWAYS land and disarm
        await cf.emergency_stop()
        print("EMERGENCY STOP ACTIVATED")
```

### Safe Flying Area

- Minimum: 2m x 2m x 2m
- Recommended: 4m x 4m x 3m
- Keep 0.5m buffer from walls/ceiling
- No loose objects that could be hit

---

# 9. Hardware Setup: MentorPi Ground Vehicles

## 9.1 Hardware Overview

### MentorPi Components

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 4 | Main computer |
| Motor Driver | Controls wheels |
| Camera | Vision |
| LiDAR (optional) | Distance sensing |
| Battery Pack | Power |
| Mecanum Wheels | Omnidirectional movement |

## 9.2 MentorPi Software Setup

### On the MentorPi (Raspberry Pi)

```bash
# SSH into your MentorPi
ssh pi@192.168.1.100  # Use your MentorPi's IP

# Install dependencies
sudo apt update
sudo apt install python3-pip

# Install motor control library
pip3 install adafruit-circuitpython-motorkit

# Create control server
mkdir ~/heterofleet_agent
cd ~/heterofleet_agent
```

### Control Server

Create `mentorpi_server.py` on the MentorPi:

```python
"""
MentorPi Control Server
Runs on the Raspberry Pi and accepts commands from HeteroFleet.
"""

import socket
import threading
import time
from adafruit_motorkit import MotorKit

# Initialize motor controller
kit = MotorKit()

# Motor mapping for mecanum wheels
# Front-Left: M1, Front-Right: M2
# Rear-Left: M3, Rear-Right: M4

class MentorPiServer:
    def __init__(self, port=8080):
        self.port = port
        self.running = False
        
        # Current state
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.battery = 100.0
        
    def start(self):
        self.running = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('0.0.0.0', self.port))
        self.server.listen(5)
        
        print(f"MentorPi server listening on port {self.port}")
        
        while self.running:
            try:
                client, addr = self.server.accept()
                print(f"Connection from {addr}")
                threading.Thread(target=self.handle_client, args=(client,)).start()
            except:
                pass
    
    def handle_client(self, client):
        while self.running:
            try:
                data = client.recv(1024).decode().strip()
                if not data:
                    break
                
                response = self.process_command(data)
                client.send(f"{response}\n".encode())
            except:
                break
        
        client.close()
    
    def process_command(self, command):
        parts = command.split()
        cmd = parts[0].upper()
        
        if cmd == "ARM":
            return "OK"
        
        elif cmd == "STOP":
            self.stop_motors()
            return "OK"
        
        elif cmd == "VELOCITY":
            # VELOCITY <v_left> <v_right>
            v_left = float(parts[1])
            v_right = float(parts[2])
            self.set_velocity(v_left, v_right)
            return "OK"
        
        elif cmd == "MECANUM":
            # MECANUM <vx> <vy> <omega>
            vx = float(parts[1])
            vy = float(parts[2])
            omega = float(parts[3])
            self.set_mecanum(vx, vy, omega)
            return "OK"
        
        elif cmd == "TELEMETRY":
            return f"{self.position[0]} {self.position[1]} {self.velocity[0]} {self.velocity[1]} 0 {self.battery}"
        
        elif cmd == "EMERGENCY_STOP":
            self.emergency_stop()
            return "OK"
        
        return "ERROR"
    
    def set_velocity(self, v_left, v_right):
        """Differential drive mode."""
        # Scale to -1 to 1
        kit.motor1.throttle = max(-1, min(1, v_left))
        kit.motor2.throttle = max(-1, min(1, v_right))
        kit.motor3.throttle = max(-1, min(1, v_left))
        kit.motor4.throttle = max(-1, min(1, v_right))
    
    def set_mecanum(self, vx, vy, omega):
        """Mecanum drive mode (omnidirectional)."""
        # Mecanum wheel kinematics
        fl = vx + vy + omega  # Front-left
        fr = vx - vy - omega  # Front-right
        rl = vx - vy + omega  # Rear-left
        rr = vx + vy - omega  # Rear-right
        
        # Normalize if needed
        max_val = max(abs(fl), abs(fr), abs(rl), abs(rr), 1.0)
        
        kit.motor1.throttle = fl / max_val
        kit.motor2.throttle = fr / max_val
        kit.motor3.throttle = rl / max_val
        kit.motor4.throttle = rr / max_val
    
    def stop_motors(self):
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0
    
    def emergency_stop(self):
        self.stop_motors()
        # Release motors
        kit.motor1.throttle = None
        kit.motor2.throttle = None
        kit.motor3.throttle = None
        kit.motor4.throttle = None

if __name__ == "__main__":
    server = MentorPiServer(port=8080)
    try:
        server.start()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.running = False
        server.stop_motors()
```

Start the server on boot:
```bash
# Add to /etc/rc.local or create a systemd service
python3 /home/pi/heterofleet_agent/mentorpi_server.py &
```

## 9.3 Using MentorPi with HeteroFleet

### Basic Connection

```python
"""
Connect to a real MentorPi robot.
"""
import asyncio
from heterofleet.hardware.mentorpi import MentorPiInterface, MentorPiConfig

async def main():
    # Configuration
    config = MentorPiConfig(
        host="192.168.1.100",  # Your MentorPi's IP
        port=8080,
        max_speed=0.5,
    )
    
    # Create interface
    robot = MentorPiInterface("ugv_0", config)
    
    # Connect
    print("Connecting to MentorPi...")
    if await robot.connect():
        print("Connected!")
        
        # Arm
        await robot.arm()
        
        # Move forward
        print("Moving forward...")
        from heterofleet.core.platform import Vector3
        await robot.set_velocity(Vector3(0.3, 0, 0))
        await asyncio.sleep(2.0)
        
        # Turn
        print("Turning...")
        await robot.set_velocity(Vector3(0, 0.2, 0))
        await asyncio.sleep(1.0)
        
        # Stop
        await robot.set_velocity(Vector3(0, 0, 0))
        
        # Disarm
        await robot.disarm()
        
        # Disconnect
        await robot.disconnect()
    else:
        print("Failed to connect!")

asyncio.run(main())
```

### Mecanum Drive (Omnidirectional)

```python
"""
Use mecanum wheels for omnidirectional movement.
"""
import asyncio
from heterofleet.hardware.mentorpi import MentorPiInterface, MentorPiConfig

async def main():
    config = MentorPiConfig(host="192.168.1.100", port=8080)
    robot = MentorPiInterface("ugv_0", config)
    
    if await robot.connect():
        await robot.arm()
        
        # Strafe left (no rotation!)
        print("Strafing left...")
        await robot.set_mecanum_velocity(
            vx=0.0,    # No forward/back
            vy=-0.3,   # Left
            omega=0.0  # No rotation
        )
        await asyncio.sleep(2.0)
        
        # Diagonal movement
        print("Moving diagonally...")
        await robot.set_mecanum_velocity(
            vx=0.3,    # Forward
            vy=0.3,    # Right
            omega=0.0
        )
        await asyncio.sleep(2.0)
        
        # Rotate in place
        print("Spinning...")
        await robot.set_mecanum_velocity(
            vx=0.0,
            vy=0.0,
            omega=1.0  # Rotate
        )
        await asyncio.sleep(2.0)
        
        # Stop
        await robot.set_mecanum_velocity(0, 0, 0)
        await robot.disarm()
        await robot.disconnect()

asyncio.run(main())
```

## 9.4 Position Tracking for MentorPi

MentorPi doesn't have built-in position tracking. Options:

### Option 1: Wheel Odometry (Built-in)

Add encoders to wheels and track position:

```python
# On MentorPi server, add odometry tracking
class OdometryTracker:
    def __init__(self, wheel_radius=0.03, wheel_base=0.15):
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
    
    def update(self, left_ticks, right_ticks, ticks_per_rev=360):
        # Convert ticks to distance
        d_left = (left_ticks / ticks_per_rev) * 2 * 3.14159 * self.wheel_radius
        d_right = (right_ticks / ticks_per_rev) * 2 * 3.14159 * self.wheel_radius
        
        # Calculate movement
        d_center = (d_left + d_right) / 2
        d_theta = (d_right - d_left) / self.wheel_base
        
        # Update pose
        self.x += d_center * math.cos(self.theta)
        self.y += d_center * math.sin(self.theta)
        self.theta += d_theta
```

### Option 2: External Tracking (Lighthouse)

Use same Lighthouse system as Crazyflie:

```python
# Add a Lighthouse receiver deck to MentorPi
# Connect via I2C to Raspberry Pi
# Read position same as Crazyflie
```

### Option 3: Camera-Based (AprilTags)

Use ceiling-mounted camera and AprilTag markers:

```python
# apriltag_tracker.py
import cv2
import apriltag

detector = apriltag.Detector()

def get_robot_pose(frame, tag_id=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    
    for r in results:
        if r.tag_id == tag_id:
            # Get center
            center = r.center
            # Get rotation
            corners = r.corners
            # Convert to world coordinates
            # (requires camera calibration)
            return center, corners
    
    return None
```

---

# 10. Running Mixed Fleet Operations

Now let's combine drones and ground robots!

## 10.1 Coordinated Mission Example

```python
"""
Complete mixed fleet mission:
- Drones survey area from above
- Ground robots inspect points of interest
"""
import asyncio
from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.hardware.crazyflie import CrazyflieSwarm
from heterofleet.hardware.mentorpi import MentorPiInterface, MentorPiConfig
from heterofleet.planning.task import Task, TaskType
from heterofleet.planning.mopota import MOPOTAAllocator

# Set to True to use real hardware
USE_REAL_HARDWARE = False

async def run_mission():
    # ===== SETUP =====
    
    # Create coordination engine (always simulated for coordination logic)
    config = SimulationConfig(
        time_step=0.05,
        num_uavs=2,
        num_ugvs=1,
        enable_coordination=True,
    )
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    # Hardware interfaces (real or simulated)
    if USE_REAL_HARDWARE:
        # Real drones
        swarm = CrazyflieSwarm()
        swarm.add_drone("uav_0", "radio://0/80/2M/E7E7E7E701")
        swarm.add_drone("uav_1", "radio://0/80/2M/E7E7E7E702")
        await swarm.connect_all()
        
        # Real ground robot
        ugv_config = MentorPiConfig(host="192.168.1.100", port=8080)
        ugv = MentorPiInterface("ugv_0", ugv_config)
        await ugv.connect()
        
        hardware = {"swarm": swarm, "ugv": ugv}
    else:
        hardware = None
        print("Running in simulation mode")
    
    # ===== MISSION PLANNING =====
    
    # Define survey points
    survey_points = [
        Vector3(10, 10, 3),
        Vector3(-10, 10, 3),
        Vector3(10, -10, 3),
        Vector3(-10, -10, 3),
    ]
    
    # Define inspection points (discovered by drones)
    inspection_points = [
        Vector3(5, 5, 0),
        Vector3(-5, 5, 0),
    ]
    
    # ===== PHASE 1: Aerial Survey =====
    print("\n" + "="*50)
    print("PHASE 1: Aerial Survey")
    print("="*50)
    
    if USE_REAL_HARDWARE:
        # Arm and takeoff
        for drone in swarm.get_all_drones():
            await drone.arm()
        await swarm.takeoff_all(altitude=1.0)
        await asyncio.sleep(2.0)
    
    # Assign survey points to drones
    drone_assignments = {
        "uav_0": [survey_points[0], survey_points[2]],
        "uav_1": [survey_points[1], survey_points[3]],
    }
    
    for drone_id, points in drone_assignments.items():
        print(f"\n{drone_id} surveying {len(points)} points...")
        
        for i, point in enumerate(points):
            print(f"  Point {i+1}: ({point.x}, {point.y}, {point.z})")
            
            if USE_REAL_HARDWARE:
                drone = swarm.get_drone(drone_id)
                await drone.set_position(point)
                await asyncio.sleep(5.0)  # Wait to reach and survey
            else:
                engine.set_agent_target(drone_id, point)
                engine.run(duration=5.0)
    
    print("\nSurvey complete!")
    
    # ===== PHASE 2: Ground Inspection =====
    print("\n" + "="*50)
    print("PHASE 2: Ground Inspection")
    print("="*50)
    
    if USE_REAL_HARDWARE:
        await ugv.arm()
    
    for i, point in enumerate(inspection_points):
        print(f"\nInspecting point {i+1}: ({point.x}, {point.y})")
        
        if USE_REAL_HARDWARE:
            await ugv.set_position(point)
            await asyncio.sleep(10.0)  # Time to reach and inspect
        else:
            engine.set_agent_target("ugv_0", point)
            engine.run(duration=10.0)
        
        print(f"  Inspection complete!")
    
    # ===== PHASE 3: Return Home =====
    print("\n" + "="*50)
    print("PHASE 3: Return to Base")
    print("="*50)
    
    home = Vector3(0, 0, 0)
    
    if USE_REAL_HARDWARE:
        # Land drones
        print("Landing drones...")
        await swarm.land_all()
        
        # Return ground robot
        print("Returning ground robot...")
        await ugv.set_position(home)
        await asyncio.sleep(15.0)
        await ugv.disarm()
        
        # Disconnect
        await swarm.disconnect_all()
        await ugv.disconnect()
    else:
        for agent_id in engine._agents:
            target = Vector3(0, 0, 1 if "uav" in agent_id else 0)
            engine.set_agent_target(agent_id, target)
        engine.run(duration=15.0)
    
    # ===== RESULTS =====
    print("\n" + "="*50)
    print("MISSION COMPLETE")
    print("="*50)
    
    metrics = engine.fleet_twin.metrics
    print(f"\nFleet Status:")
    print(f"  Total agents: {metrics.total_agents}")
    print(f"  Average battery: {metrics.avg_battery_level:.0%}")

if __name__ == "__main__":
    asyncio.run(run_mission())
```

## 10.2 Real-Time Coordination

For true real-time coordination between real robots:

```python
"""
Real-time coordination loop for hardware.
"""
import asyncio
from heterofleet.core.platform import Vector3
from heterofleet.coordination.haim import HAIMCoordinator, HAIMParameters
from heterofleet.hardware.crazyflie import CrazyflieSwarm

async def realtime_coordination():
    # Setup HAIM coordinator
    params = HAIMParameters(
        repulsion_strength=2.0,
        repulsion_range=1.0,  # 1 meter for real hardware
    )
    coordinator = HAIMCoordinator(params)
    
    # Connect to hardware
    swarm = CrazyflieSwarm()
    swarm.add_drone("uav_0", "radio://0/80/2M/E7E7E7E701")
    swarm.add_drone("uav_1", "radio://0/80/2M/E7E7E7E702")
    await swarm.connect_all()
    
    # Targets
    targets = {
        "uav_0": Vector3(1.0, 0.0, 0.8),
        "uav_1": Vector3(-1.0, 0.0, 0.8),
    }
    
    # Takeoff
    for drone in swarm.get_all_drones():
        await drone.arm()
    await swarm.takeoff_all(0.5)
    await asyncio.sleep(2.0)
    
    # Real-time control loop (50 Hz)
    dt = 0.02  # 20ms
    
    for step in range(500):  # 10 seconds
        # 1. Get telemetry from all drones
        telemetry = swarm.get_telemetry_all()
        
        # 2. Update coordinator with current states
        for drone_id, telem in telemetry.items():
            drone = swarm.get_drone(drone_id)
            state = drone.to_agent_state()
            coordinator.neighbor_tracker.update_agent(
                drone_id, state, drone.platform_spec
            )
        
        coordinator.neighbor_tracker.compute_neighbors()
        
        # 3. Compute coordinated velocities
        for drone_id in telemetry:
            target = targets[drone_id]
            
            interaction = coordinator.compute_interaction(
                drone_id,
                target,
                dt=dt
            )
            
            # 4. Send velocity command to drone
            drone = swarm.get_drone(drone_id)
            await drone.set_velocity(interaction.desired_velocity)
        
        await asyncio.sleep(dt)
    
    # Land
    await swarm.land_all()
    await swarm.disconnect_all()

if __name__ == "__main__":
    asyncio.run(realtime_coordination())
```

---

# 11. Experiments and Evaluation

HeteroFleet includes 5 experiments for systematic evaluation.

## 11.1 Experiment 1: Scalability

Test how the system performs as fleet size increases.

```python
"""
Scalability Experiment
"""
from experiments.scalability import ScalabilityExperiment, ScalabilityConfig

# Configure
config = ScalabilityConfig(
    num_runs=3,              # 3 runs per fleet size
    duration=30.0,           # 30 second simulations
    fleet_sizes=[5, 10, 20, 50],  # Test these sizes
    output_dir="./results/scalability",
)

# Run
experiment = ScalabilityExperiment(config)
summary = experiment.run_all()

# Results
print("Scalability Results:")
for size in config.fleet_sizes:
    results = [r for r in experiment.get_results() 
               if r.metrics.get('fleet_size') == size and r.success]
    if results:
        avg_step_time = sum(r.metrics['avg_step_time_ms'] for r in results) / len(results)
        print(f"  {size} agents: {avg_step_time:.2f}ms/step")
```

## 11.2 Experiment 2: Formation Control

Test formation acquisition and maintenance.

```python
"""
Formation Control Experiment
"""
from experiments.formation import FormationExperiment, FormationConfig

# Configure
config = FormationConfig(
    num_runs=5,
    duration=30.0,
    formation_type="v",      # V-formation (like birds)
    num_agents=5,
    formation_spacing=2.0,   # 2m between agents
    output_dir="./results/formation",
)

# Run
experiment = FormationExperiment(config)
summary = experiment.run_all()

# Results
print("Formation Results:")
print(f"  Success rate: {summary.metrics_mean['formation_achieved']*100:.0f}%")
print(f"  Time to form: {summary.metrics_mean['time_to_formation']:.1f}s")
print(f"  Final error: {summary.metrics_mean['final_error']:.2f}m")
```

## 11.3 Experiment 3: Task Allocation

Evaluate MOPOTA performance.

```python
"""
Task Allocation Experiment
"""
from experiments.task_allocation import TaskAllocationExperiment, TaskAllocationConfig

# Configure
config = TaskAllocationConfig(
    num_runs=10,
    duration=60.0,
    num_uavs=5,
    num_ugvs=2,
    num_tasks=15,
    output_dir="./results/task_allocation",
)

# Run
experiment = TaskAllocationExperiment(config)
summary = experiment.run_all()

# Results
print("Task Allocation Results:")
print(f"  Completion rate: {summary.metrics_mean['completion_rate']*100:.1f}%")
print(f"  Mission time: {summary.metrics_mean['mission_time']:.1f}s")
print(f"  Workload balance: {summary.metrics_mean['workload_balance']:.2f}")
```

## 11.4 Running All Experiments

```python
"""
Run all experiments and generate report.
"""
from experiments import (
    ScalabilityExperiment, ScalabilityConfig,
    FormationExperiment, FormationConfig,
    TaskAllocationExperiment, TaskAllocationConfig,
    CommunicationExperiment, CommunicationConfig,
    EmergencyExperiment, EmergencyConfig,
)

def run_all_experiments():
    results = {}
    
    # 1. Scalability
    print("\n" + "="*50)
    print("Running Scalability Experiment...")
    exp = ScalabilityExperiment(ScalabilityConfig(num_runs=3))
    results["scalability"] = exp.run_all()
    
    # 2. Formation
    print("\n" + "="*50)
    print("Running Formation Experiment...")
    exp = FormationExperiment(FormationConfig(num_runs=5))
    results["formation"] = exp.run_all()
    
    # 3. Task Allocation
    print("\n" + "="*50)
    print("Running Task Allocation Experiment...")
    exp = TaskAllocationExperiment(TaskAllocationConfig(num_runs=5))
    results["task_allocation"] = exp.run_all()
    
    # 4. Communication
    print("\n" + "="*50)
    print("Running Communication Experiment...")
    exp = CommunicationExperiment(CommunicationConfig(num_runs=5))
    results["communication"] = exp.run_all()
    
    # 5. Emergency Response
    print("\n" + "="*50)
    print("Running Emergency Response Experiment...")
    exp = EmergencyExperiment(EmergencyConfig(num_runs=5))
    results["emergency"] = exp.run_all()
    
    # Generate summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for name, summary in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Runs: {summary.num_runs}")
        print(f"  Success: {summary.successful_runs}/{summary.num_runs}")
        for metric, value in summary.metrics_mean.items():
            print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    run_all_experiments()
```

---

# 12. Advanced Topics

## 12.1 Custom Platform Types

Add your own robot type:

```python
from heterofleet.core.platform import PlatformType, PlatformSpecification

# Create custom spec
my_robot = PlatformSpecification(
    platform_id="custom_heavy_lifter",
    platform_type=PlatformType.MEDIUM_UAV,  # Closest match
    physical={
        "mass": 5.0,
        "dimensions": {"length": 0.8, "width": 0.8, "height": 0.3},
    },
    dynamic={
        "max_velocity": [5.0, 5.0, 3.0],
        "max_acceleration": [3.0, 3.0, 2.0],
    },
    energy={
        "battery_capacity_wh": 200.0,
        "cruise_power": 100.0,
    },
    capabilities={
        "can_carry_payload": True,
        "max_payload_kg": 2.0,
    },
)

# Register it
from heterofleet.core.platform import PlatformRegistry
registry = PlatformRegistry()
registry.register(my_robot)
```

## 12.2 Custom Safety Specifications

```python
from heterofleet.safety.stl_monitor import STLMonitor, STLSpecification

monitor = STLMonitor()

# Complex safety rule:
# "If battery < 20%, return home within 60 seconds"
spec = STLSpecification(
    spec_id="low_battery_return",
    formula="G((battery < 0.2) -> F[0,60](distance_to_home < 1.0))",
    variables=["battery", "distance_to_home"],
)
monitor.add_specification(spec)
```

## 12.3 Neural Network Integration

Use the GNN coordinator for learning-based coordination:

```python
from heterofleet.ai.gnn_coordinator import GNNCoordinator, GNNConfig

# Create GNN coordinator
gnn_config = GNNConfig(
    hidden_dim=64,
    num_layers=3,
    communication_rounds=2,
)

gnn = GNNCoordinator(gnn_config)

# Build graph from fleet
graph = gnn.build_graph(
    positions={"uav_0": Vector3(0, 0, 1), "uav_1": Vector3(5, 0, 1)},
    velocities={"uav_0": Vector3(1, 0, 0), "uav_1": Vector3(-1, 0, 0)},
    platform_types={"uav_0": PlatformType.SMALL_UAV, "uav_1": PlatformType.SMALL_UAV},
    comm_range=10.0,
)

# Get coordination output
output = gnn.forward(graph)
```

## 12.4 Natural Language Commands

```python
from heterofleet.ai.llm_interpreter import LLMInterpreter

interpreter = LLMInterpreter()

# Parse natural language
command = interpreter.parse("Send drone 1 to survey the north building")

print(f"Intent: {command.intent}")
print(f"Target agents: {command.target_agents}")
print(f"Parameters: {command.parameters}")
```

---

# 13. Troubleshooting

## 13.1 Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'heterofleet'`

**Solution**:
```bash
# Make sure you're in the project directory
cd ~/heterofleet_workspace/heterofleet

# Add to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or install in development mode
pip install -e .
```

### Crazyflie Connection Failed

**Problem**: Can't find Crazyflie via radio

**Solutions**:
1. Check USB connection to Crazyradio
2. Ensure drone is powered on
3. Linux: Check udev rules (Section 8.3)
4. Try different radio channel

```python
# Scan all channels
import cflib.crtp
cflib.crtp.init_drivers()

for channel in [80, 90, 100, 110, 120]:
    uri = f"radio://0/{channel}/2M"
    available = cflib.crtp.scan_interfaces(uri)
    if available:
        print(f"Found on channel {channel}: {available}")
```

### Lighthouse Position Not Working

**Problem**: Position reads as (0, 0, 0)

**Solutions**:
1. Verify base stations are powered (green LED)
2. Check base stations are in different modes (button)
3. Ensure Lighthouse deck is properly mounted
4. Re-run geometry calibration
5. Check base station visibility (no obstructions)

### Simulation Too Slow

**Problem**: Simulation runs slower than real-time

**Solutions**:
```python
# Use infinite speed (as fast as possible)
config = SimulationConfig(
    real_time_factor=float('inf'),
    # ...
)

# Increase time step (less accurate but faster)
config = SimulationConfig(
    time_step=0.1,  # 100ms instead of 50ms
    # ...
)

# Disable unused features
config = SimulationConfig(
    enable_coordination=False,  # If not needed
    record_trajectory=False,    # Saves memory
    # ...
)
```

### Out of Memory

**Problem**: Python crashes with memory error

**Solutions**:
```python
# Reduce trajectory recording
config = SimulationConfig(
    record_trajectory=False,
    # ...
)

# Clear history periodically
engine.fleet_twin.clear_history()

# Reduce history size
# In digital_twin/agent_twin.py, modify MAX_HISTORY_SIZE
```

## 13.2 Debug Mode

Enable detailed logging:

```python
from loguru import logger
import sys

# Enable debug logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Now run your code - you'll see detailed logs
```

## 13.3 Getting Help

1. **Check the logs**: Most errors are logged with details
2. **Simplify**: Test with minimal code first
3. **Hardware test**: Use cfclient to verify Crazyflie works
4. **Simulation first**: Always test in simulation before hardware

---

# 14. API Reference

## Core Classes

### Vector3
```python
Vector3(x: float, y: float, z: float)
    .norm() -> float           # Length
    .normalize() -> Vector3    # Unit vector
    .dot(other) -> float       # Dot product
    .cross(other) -> Vector3   # Cross product
    + - * / operators          # Vector math
```

### PlatformSpecification
```python
PlatformSpecification.from_platform_type(type, id) -> PlatformSpecification
    .platform_type             # PlatformType enum
    .max_velocity             # float (m/s)
    .battery_capacity_wh      # float (Wh)
    .collision_envelope       # CollisionEnvelope
```

### SimulationEngine
```python
SimulationEngine(config: SimulationConfig)
    .setup_default_scenario()
    .add_agent(id, spec, position)
    .remove_agent(id)
    .set_agent_target(id, position)
    .set_agent_velocity(id, velocity)
    .start()
    .step() -> Dict[str, AgentState]
    .run(duration: float)
    .pause()
    .stop()
    .state -> SimulationState
    .fleet_twin -> FleetTwin
```

### CrazyflieInterface
```python
CrazyflieInterface(agent_id, config: CrazyflieConfig)
    async .connect() -> bool
    async .disconnect()
    async .arm() -> bool
    async .disarm() -> bool
    async .takeoff(altitude) -> bool
    async .land() -> bool
    async .set_velocity(velocity: Vector3) -> bool
    async .set_position(position: Vector3) -> bool
    async .emergency_stop() -> bool
    async .get_telemetry() -> TelemetryData
```

### MentorPiInterface
```python
MentorPiInterface(agent_id, config: MentorPiConfig)
    async .connect() -> bool
    async .disconnect()
    async .arm() -> bool
    async .disarm() -> bool
    async .set_velocity(velocity: Vector3) -> bool
    async .set_position(position: Vector3) -> bool
    async .set_mecanum_velocity(vx, vy, omega) -> bool
    async .emergency_stop() -> bool
```

---

# Appendix A: Quick Reference Card

## Coordinate System
```
X: Right (+) / Left (-)
Y: Forward (+) / Backward (-)
Z: Up (+) / Down (-)
Units: Meters, Meters/second
```

## Platform Types
```
MICRO_UAV  - Crazyflie (~30g)
SMALL_UAV  - DJI Mini (~250g)
MEDIUM_UAV - DJI Matrice (~3kg)
SMALL_UGV  - MentorPi (~1.5kg)
MEDIUM_UGV - Jackal (~17kg)
```

## Simulation Quick Start
```python
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.core.platform import Vector3

engine = SimulationEngine(SimulationConfig(num_uavs=3, num_ugvs=1))
engine.setup_default_scenario()
engine.set_agent_target("uav_0", Vector3(10, 0, 2))
engine.run(duration=30.0)
```

## Crazyflie Quick Start
```python
import asyncio
from heterofleet.hardware.crazyflie import CrazyflieInterface, CrazyflieConfig

async def fly():
    cf = CrazyflieInterface("uav", CrazyflieConfig(uri="radio://0/80/2M/E7E7E7E7E7"))
    await cf.connect()
    await cf.arm()
    await cf.takeoff(0.5)
    await asyncio.sleep(3)
    await cf.land()
    await cf.disconnect()

asyncio.run(fly())
```

---

**Congratulations!** You've completed the HeteroFleet beginner's guide. You now have the knowledge to:

- ✅ Set up and run simulations
- ✅ Understand all major modules
- ✅ Connect to real Crazyflie drones
- ✅ Connect to MentorPi ground robots
- ✅ Run coordinated mixed-fleet missions
- ✅ Evaluate system performance with experiments

Happy flying and driving! 🚁🤖
