#!/usr/bin/env python
"""
Tutorial 2: Multiple Robots
===========================
Coordinate a mixed fleet of drones and ground robots.

Run: python tutorials/tutorial_02_multi_robot.py
"""

import sys
sys.path.insert(0, '.')

from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

def main():
    print("="*60)
    print("Tutorial 2: Multiple Robots")
    print("="*60)
    print()
    
    # Create a mixed fleet
    print("Creating mixed fleet...")
    config = SimulationConfig(
        time_step=0.05,
        real_time_factor=float('inf'),
        max_duration=60.0,
        num_uavs=3,              # Three drones
        num_ugvs=2,              # Two ground robots
        enable_coordination=True, # Enable collision avoidance!
    )
    
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    # Print initial positions
    print("\nInitial Fleet Configuration:")
    print("-"*50)
    for agent_id, agent in engine._agents.items():
        ptype = agent.platform_spec.platform_type.name
        pos = agent.position
        print(f"  {agent_id} ({ptype}):")
        print(f"    Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
    
    # Assign targets
    targets = {
        "uav_0": Vector3(15, 0, 3),      # Drone 0 goes right and up
        "uav_1": Vector3(-10, 10, 2),    # Drone 1 goes left-forward
        "uav_2": Vector3(0, -15, 4),     # Drone 2 goes backward and high
        "ugv_0": Vector3(8, 8, 0),       # Ground robot 0 goes diagonal
        "ugv_1": Vector3(-8, -8, 0),     # Ground robot 1 goes opposite
    }
    
    print("\nAssigning Targets:")
    print("-"*50)
    for agent_id, target in targets.items():
        if agent_id in engine._agents:
            engine.set_agent_target(agent_id, target)
            print(f"  {agent_id} -> ({target.x:.0f}, {target.y:.0f}, {target.z:.0f})")
    
    # Run simulation
    print("\n" + "="*50)
    print("Running simulation for 30 seconds...")
    print("(With collision avoidance enabled)")
    print("="*50)
    
    engine.run(duration=30.0)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    state = engine.state
    print(f"\nSimulation Statistics:")
    print(f"  Simulation time: {state.sim_time:.1f}s")
    print(f"  Computation time: {state.wall_time:.2f}s")
    print(f"  Speed: {state.sim_time/state.wall_time:.1f}x real-time")
    print(f"  Total steps: {state.step_count}")
    
    print(f"\nFinal Positions and Status:")
    print("-"*50)
    
    all_reached = True
    for agent_id, agent in engine._agents.items():
        pos = agent.position
        target = targets.get(agent_id, Vector3(0, 0, 0))
        dist = ((pos.x - target.x)**2 + 
                (pos.y - target.y)**2 + 
                (pos.z - target.z)**2) ** 0.5
        battery = agent.battery_level * 100
        
        status = "✓ Reached" if dist < 1.0 else f"✗ {dist:.1f}m away"
        if dist >= 1.0:
            all_reached = False
            
        print(f"  {agent_id}: {status}")
        print(f"    Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
        print(f"    Battery: {battery:.0f}%")
    
    # Fleet metrics
    print(f"\nFleet Metrics:")
    print("-"*50)
    metrics = engine.fleet_twin.metrics
    print(f"  Total agents: {metrics.total_agents}")
    print(f"  Average battery: {metrics.avg_battery_level:.1%}")
    print(f"  Minimum battery: {metrics.min_battery_level:.1%}")
    print(f"  Fleet spread: {metrics.spread_radius:.1f}m")
    
    print("\n" + "="*60)
    if all_reached:
        print("SUCCESS! All robots reached their targets.")
    else:
        print("Some robots are still en route.")
        print("(Ground robots are slower than drones)")
    print("="*60)

if __name__ == "__main__":
    main()
