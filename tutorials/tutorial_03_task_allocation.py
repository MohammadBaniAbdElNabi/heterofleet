#!/usr/bin/env python
"""
Tutorial 3: Task Allocation
===========================
Automatically assign tasks to the best robots.

Run: python tutorials/tutorial_03_task_allocation.py
"""

import sys
sys.path.insert(0, '.')

from heterofleet.core.platform import Vector3, PlatformType
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.planning.task import Task, TaskType, TaskPriority
from heterofleet.planning.mopota import MOPOTAAllocator, AgentCapabilities, ObjectiveWeights

def main():
    print("="*60)
    print("Tutorial 3: Task Allocation")
    print("="*60)
    print()
    
    # Setup fleet
    print("Setting up fleet...")
    config = SimulationConfig(
        time_step=0.05,
        real_time_factor=float('inf'),
        num_uavs=3,
        num_ugvs=2,
    )
    
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    print(f"  Created {len(engine._agents)} agents")
    print()
    
    # Define tasks
    print("Defining tasks...")
    from heterofleet.planning.task import TaskLocation, TaskConstraints
    
    tasks = [
        Task(
            task_id="survey_north",
            task_type=TaskType.SURVEILLANCE,
            location=TaskLocation(position=Vector3(0, 20, 5)),
            constraints=TaskConstraints(max_duration=30.0),
            priority=TaskPriority.HIGH,
        ),
        Task(
            task_id="survey_east", 
            task_type=TaskType.SURVEILLANCE,
            location=TaskLocation(position=Vector3(15, 0, 3)),
            constraints=TaskConstraints(max_duration=20.0),
            priority=TaskPriority.NORMAL,
        ),
        Task(
            task_id="deliver_package",
            task_type=TaskType.DELIVERY,
            location=TaskLocation(position=Vector3(-10, 10, 0)),
            constraints=TaskConstraints(max_duration=60.0),
            priority=TaskPriority.HIGH,
        ),
        Task(
            task_id="inspect_building",
            task_type=TaskType.INSPECTION,
            location=TaskLocation(position=Vector3(5, -10, 2)),
            constraints=TaskConstraints(max_duration=45.0),
            priority=TaskPriority.LOW,
        ),
        Task(
            task_id="ground_patrol",
            task_type=TaskType.CUSTOM,
            location=TaskLocation(position=Vector3(-5, -15, 0)),
            constraints=TaskConstraints(max_duration=40.0),
            priority=TaskPriority.NORMAL,
        ),
    ]
    
    print("\nTasks to allocate:")
    print("-"*50)
    for task in tasks:
        loc = task.location.position
        print(f"  {task.task_id}:")
        print(f"    Type: {task.task_type.name}")
        print(f"    Location: ({loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f})")
        print(f"    Priority: {task.priority.name}")
    
    # Define agent capabilities
    print("\nDefining agent capabilities...")
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
                payload_capacity=0.1,
                sensor_range=10.0,
                energy_level=1.0,
                energy_capacity=20.0,
                capabilities=["camera", "aerial", "surveillance"],
            )
            print(f"  {agent_id}: Drone with camera, aerial capability")
        else:
            caps = AgentCapabilities(
                agent_id=agent_id,
                platform_type=ptype,
                position=pos,
                max_speed=0.5,
                payload_capacity=2.0,
                sensor_range=5.0,
                energy_level=1.0,
                energy_capacity=30.0,
                capabilities=["camera", "cargo", "ground"],
            )
            print(f"  {agent_id}: Ground robot with cargo, ground capability")
        
        capabilities[agent_id] = caps
    
    # Run MOPOTA
    print("\n" + "="*50)
    print("Running Task Allocator...")
    print("="*50)
    print()
    print("Using greedy allocation (fast mode)")
    print()
    
    allocator = MOPOTAAllocator(
        population_size=30,
        max_generations=30,
    )
    
    # Set tasks and agents
    allocator.set_tasks(tasks)
    allocator.set_agents(list(capabilities.values()))
    
    # Use greedy allocation (simpler and faster)
    result = allocator.allocate_greedy()
    
    # Show results
    print("="*60)
    print("ALLOCATION RESULTS")
    print("="*60)
    print()
    
    print("Task Assignments:")
    print("-"*50)
    for task_id, agent_id in result.assignments.items():
        task = next((t for t in tasks if t.task_id == task_id), None)
        if agent_id and task:
            agent_type = "Drone" if "uav" in agent_id else "Ground Robot"
            print(f"  {task_id}")
            print(f"    Assigned to: {agent_id} ({agent_type})")
            print(f"    Task type: {task.task_type.name}")
    
    if result.num_tasks_unassigned > 0:
        print(f"\n  {result.num_tasks_unassigned} task(s) unassigned")
    
    print()
    print("Optimization Objectives:")
    print("-"*50)
    print(f"  Completion rate: {result.completion_rate:.1%}")
    print(f"  Mission time: {result.mission_time:.1f}s")
    print(f"  Energy consumption: {result.energy_consumption:.2f}")
    print(f"  Workload balance: {result.workload_balance:.2f}")
    print(f"  (lower is better for workload balance)")
    
    if result.pareto_front_size > 0:
        print(f"\nPareto-optimal solutions found: {result.pareto_front_size}")
    
    print()
    print("="*60)
    print("Task allocation complete!")
    print("="*60)
    print()
    print("Key insight: MOPOTA assigns tasks based on:")
    print("  1. Agent capabilities (can they do it?)")
    print("  2. Distance to task (who's closest?)")
    print("  3. Agent type suitability (drone vs ground)")
    print("  4. Workload balancing (don't overload one agent)")

if __name__ == "__main__":
    main()
