#!/usr/bin/env python
"""
Tutorial 1: Hello HeteroFleet
=============================
Your first simulation with a single drone.

Run: python tutorials/tutorial_01_hello.py
"""

import sys
sys.path.insert(0, '.')

from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

def main():
    print("="*60)
    print("Tutorial 1: Hello HeteroFleet")
    print("="*60)
    print()
    
    # Step 1: Create simulation configuration
    print("Step 1: Creating simulation configuration...")
    config = SimulationConfig(
        time_step=0.05,          # Simulate in 50ms steps
        real_time_factor=float('inf'),  # Run as fast as possible
        max_duration=30.0,       # Maximum 30 seconds
        num_uavs=1,              # One drone
        num_ugvs=0,              # No ground robots
        enable_coordination=False,  # Simple mode
    )
    print(f"  Time step: {config.time_step}s")
    print(f"  UAVs: {config.num_uavs}")
    print()
    
    # Step 2: Create the simulation engine
    print("Step 2: Creating simulation engine...")
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    print("  Engine created!")
    print()
    
    # Step 3: See what we created
    print("Step 3: Initial state")
    for agent_id, agent in engine._agents.items():
        pos = agent.position
        print(f"  {agent_id}:")
        print(f"    Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        print(f"    Platform: {agent.platform_spec.platform_type.name}")
        print(f"    Battery: {agent.battery_level:.0%}")
    print()
    
    # Step 4: Give the drone a target
    print("Step 4: Setting target...")
    target = Vector3(10.0, 5.0, 2.0)
    engine.set_agent_target("uav_0", target)
    print(f"  Target: ({target.x}, {target.y}, {target.z})")
    print()
    
    # Step 5: Run the simulation
    print("Step 5: Running simulation for 15 seconds...")
    engine.run(duration=15.0)
    print("  Done!")
    print()
    
    # Step 6: Check the results
    print("Step 6: Results")
    print("-"*40)
    
    state = engine.state
    print(f"  Simulation time: {state.sim_time:.2f}s")
    print(f"  Computation time: {state.wall_time:.2f}s")
    print(f"  Steps executed: {state.step_count}")
    print(f"  Speed: {state.sim_time/state.wall_time:.1f}x real-time")
    print()
    
    final_pos = engine._agents["uav_0"].position
    print(f"  Final position: ({final_pos.x:.2f}, {final_pos.y:.2f}, {final_pos.z:.2f})")
    
    distance = ((final_pos.x - target.x)**2 + 
                (final_pos.y - target.y)**2 + 
                (final_pos.z - target.z)**2) ** 0.5
    print(f"  Distance to target: {distance:.2f}m")
    
    battery = engine._agents["uav_0"].battery_level
    print(f"  Battery remaining: {battery:.1%}")
    
    print()
    print("="*60)
    if distance < 1.0:
        print("SUCCESS! The drone reached its target.")
    else:
        print("The drone is still moving toward the target.")
    print("="*60)

if __name__ == "__main__":
    main()
