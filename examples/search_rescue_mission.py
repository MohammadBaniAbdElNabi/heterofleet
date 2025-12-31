#!/usr/bin/env python
"""
Example: Complete Search and Rescue Mission
============================================

This example demonstrates a realistic multi-phase mission:
1. Phase 1: Aerial reconnaissance (drones survey the area)
2. Phase 2: Target identification (find points of interest)
3. Phase 3: Ground response (ground robots inspect targets)
4. Phase 4: Return to base

Run: python examples/search_rescue_mission.py
"""

import sys
sys.path.insert(0, '.')

import time
from heterofleet.core.platform import Vector3, PlatformType
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
from heterofleet.digital_twin.mission_twin import MissionTwin, MissionObjective, MissionStatus


def print_phase(phase_num, title):
    """Print a phase header."""
    print()
    print("=" * 60)
    print(f"PHASE {phase_num}: {title}")
    print("=" * 60)
    print()


def print_status(engine, targets=None):
    """Print current fleet status."""
    print("Fleet Status:")
    print("-" * 40)
    for agent_id, agent in engine._agents.items():
        pos = agent.position
        battery = agent.battery_level * 100
        
        status = ""
        if targets and agent_id in targets:
            target = targets[agent_id]
            dist = ((pos.x - target.x)**2 + (pos.y - target.y)**2 + (pos.z - target.z)**2) ** 0.5
            status = f" [→ target: {dist:.1f}m]"
        
        print(f"  {agent_id}: ({pos.x:6.1f}, {pos.y:6.1f}, {pos.z:5.1f}) Battery: {battery:5.1f}%{status}")
    print()


def main():
    print("=" * 60)
    print("SEARCH AND RESCUE MISSION")
    print("=" * 60)
    print()
    print("Mission briefing:")
    print("  - Area to search: 40m x 40m")
    print("  - Assets: 3 drones (aerial survey) + 2 ground robots (response)")
    print("  - Objective: Survey area, identify targets, dispatch responders")
    print()
    
    # ========== SETUP ==========
    
    print("Initializing fleet...")
    config = SimulationConfig(
        time_step=0.05,
        real_time_factor=float('inf'),  # Fast simulation
        max_duration=300.0,
        num_uavs=3,
        num_ugvs=2,
        enable_coordination=True,
    )
    
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    # Create mission tracker
    mission = MissionTwin(
        mission_id="sar_001",
        name="Area Search and Rescue"
    )
    
    # Add objectives
    mission.add_objective(MissionObjective(
        objective_id="area_coverage",
        description="Survey 100% of target area",
        target_value=1.0,
    ))
    
    mission.start()
    
    print_status(engine)
    
    # ========== PHASE 1: AERIAL RECONNAISSANCE ==========
    
    print_phase(1, "AERIAL RECONNAISSANCE")
    print("Drones will survey the search area in a grid pattern.")
    print()
    
    # Define survey waypoints (grid pattern covering 40x40m area)
    survey_grid = [
        # Drone 0: Right side
        [Vector3(15, -15, 5), Vector3(15, 0, 5), Vector3(15, 15, 5)],
        # Drone 1: Center
        [Vector3(0, -15, 5), Vector3(0, 0, 5), Vector3(0, 15, 5)],
        # Drone 2: Left side
        [Vector3(-15, -15, 5), Vector3(-15, 0, 5), Vector3(-15, 15, 5)],
    ]
    
    print("Survey assignments:")
    for i, waypoints in enumerate(survey_grid):
        print(f"  uav_{i}: {len(waypoints)} waypoints")
    
    # Execute survey
    for wp_idx in range(3):  # 3 waypoints per drone
        print(f"\nWaypoint {wp_idx + 1}/3...")
        
        targets = {}
        for drone_idx in range(3):
            drone_id = f"uav_{drone_idx}"
            target = survey_grid[drone_idx][wp_idx]
            targets[drone_id] = target
            engine.set_agent_target(drone_id, target)
        
        # Run until drones reach waypoints (or timeout)
        engine.run(duration=8.0)
        print_status(engine, targets)
    
    print("✓ Aerial survey complete!")
    
    # Simulate finding targets during survey
    discovered_targets = [
        Vector3(12, 8, 0),    # Target 1: East area
        Vector3(-8, -12, 0),  # Target 2: Southwest area
    ]
    
    print(f"\nTargets discovered: {len(discovered_targets)}")
    for i, target in enumerate(discovered_targets):
        print(f"  Target {i+1}: ({target.x:.0f}, {target.y:.0f})")
    
    # ========== PHASE 2: DRONES HOVER FOR OVERWATCH ==========
    
    print_phase(2, "ESTABLISH OVERWATCH")
    print("Drones move to overwatch positions above discovered targets.")
    print()
    
    # Position drones for overwatch
    overwatch_positions = {
        "uav_0": Vector3(12, 8, 8),    # Over target 1
        "uav_1": Vector3(-8, -12, 8),  # Over target 2
        "uav_2": Vector3(0, 0, 10),    # Central high position
    }
    
    print("Overwatch assignments:")
    for drone_id, pos in overwatch_positions.items():
        print(f"  {drone_id}: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f})")
        engine.set_agent_target(drone_id, pos)
    
    engine.run(duration=10.0)
    print_status(engine, overwatch_positions)
    print("✓ Overwatch established!")
    
    # ========== PHASE 3: GROUND RESPONSE ==========
    
    print_phase(3, "GROUND RESPONSE")
    print("Ground robots dispatched to inspect discovered targets.")
    print()
    
    # Assign ground robots to targets
    ground_assignments = {
        "ugv_0": discovered_targets[0],  # To target 1
        "ugv_1": discovered_targets[1],  # To target 2
    }
    
    print("Response assignments:")
    for robot_id, target in ground_assignments.items():
        print(f"  {robot_id} -> Target at ({target.x:.0f}, {target.y:.0f})")
        engine.set_agent_target(robot_id, target)
    
    # Ground robots are slower, run longer
    print("\nDispatching ground units...")
    for step in range(4):
        engine.run(duration=15.0)
        
        # Check progress
        all_arrived = True
        for robot_id, target in ground_assignments.items():
            pos = engine._agents[robot_id].position
            dist = ((pos.x - target.x)**2 + (pos.y - target.y)**2) ** 0.5
            if dist > 1.0:
                all_arrived = False
        
        elapsed = (step + 1) * 15
        print(f"  Progress check at {elapsed}s...")
        
        if all_arrived:
            break
    
    print_status(engine, ground_assignments)
    
    # Simulate inspection
    print("Ground units inspecting targets...")
    time.sleep(0.5)  # Simulate inspection time
    print("✓ Inspection complete!")
    print()
    print("Inspection results:")
    print("  Target 1: Debris field - marked for cleanup")
    print("  Target 2: Survivor located - medical response requested")
    
    # ========== PHASE 4: RETURN TO BASE ==========
    
    print_phase(4, "RETURN TO BASE")
    print("All units returning to launch point.")
    print()
    
    # Define return positions (staggered for safety)
    return_positions = {
        "uav_0": Vector3(2, 0, 1),
        "uav_1": Vector3(-2, 0, 1),
        "uav_2": Vector3(0, 2, 1),
        "ugv_0": Vector3(2, -2, 0),
        "ugv_1": Vector3(-2, -2, 0),
    }
    
    for agent_id, pos in return_positions.items():
        engine.set_agent_target(agent_id, pos)
    
    print("Return waypoints assigned...")
    engine.run(duration=30.0)
    
    print_status(engine, return_positions)
    
    # ========== MISSION SUMMARY ==========
    
    print()
    print("=" * 60)
    print("MISSION SUMMARY")
    print("=" * 60)
    print()
    
    state = engine.state
    print(f"Mission Duration: {state.sim_time:.1f}s (simulated)")
    print(f"Computation Time: {state.wall_time:.2f}s")
    print()
    
    metrics = engine.fleet_twin.metrics
    print("Fleet Status:")
    print(f"  Total agents: {metrics.total_agents}")
    print(f"  Average battery: {metrics.avg_battery_level:.1%}")
    print(f"  Minimum battery: {metrics.min_battery_level:.1%}")
    print()
    
    print("Mission Accomplishments:")
    print("  ✓ Area surveyed: 40m x 40m")
    print(f"  ✓ Targets discovered: {len(discovered_targets)}")
    print("  ✓ Ground inspections completed: 2")
    print("  ✓ All units returned safely")
    print()
    
    print("=" * 60)
    print("MISSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
