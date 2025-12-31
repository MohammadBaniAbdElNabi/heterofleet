#!/usr/bin/env python
"""
Tutorial 4: Digital Twins
=========================
Monitor and predict robot behavior using digital twins.

Run: python tutorials/tutorial_04_digital_twins.py
"""

import sys
sys.path.insert(0, '.')

from heterofleet.core.platform import Vector3
from heterofleet.simulation.engine import SimulationEngine, SimulationConfig

def main():
    print("="*60)
    print("Tutorial 4: Digital Twins")
    print("="*60)
    print()
    print("Digital twins are virtual copies of real robots that:")
    print("  - Track current state")
    print("  - Predict future behavior")
    print("  - Detect anomalies")
    print("  - Aggregate fleet-level metrics")
    print()
    
    # Setup
    config = SimulationConfig(
        time_step=0.05,
        real_time_factor=float('inf'),
        num_uavs=2,
        num_ugvs=1,
        enable_coordination=True,
    )
    
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    # Set targets
    targets = {
        "uav_0": Vector3(10, 0, 2),
        "uav_1": Vector3(-10, 0, 2),
        "ugv_0": Vector3(0, 10, 0),
    }
    
    for agent_id, target in targets.items():
        engine.set_agent_target(agent_id, target)
    
    # Run a bit to get some state
    print("Running simulation to generate state history...")
    engine.run(duration=5.0)
    print("Done!\n")
    
    # Access digital twins
    print("="*60)
    print("AGENT DIGITAL TWINS")
    print("="*60)
    
    for agent_id in engine._agents:
        twin = engine.fleet_twin.get_agent_twin(agent_id)
        
        print(f"\n{agent_id} Digital Twin:")
        print("-"*40)
        
        # Current state
        state = twin.state
        print(f"  Current State:")
        print(f"    Position: ({state.position.x:.2f}, {state.position.y:.2f}, {state.position.z:.2f})")
        print(f"    Velocity: ({state.velocity.x:.2f}, {state.velocity.y:.2f}, {state.velocity.z:.2f})")
        print(f"    Battery: {state.battery_level:.1%}")
        print(f"    Mode: {state.mode}")
        print(f"    Status: {twin.status.name}")
        
        # Prediction
        print(f"\n  Future Prediction (5 seconds ahead):")
        future = twin.predict_state(dt=5.0)
        print(f"    Predicted position: ({future.position.x:.2f}, {future.position.y:.2f}, {future.position.z:.2f})")
        print(f"    Predicted battery: {future.battery_level:.1%}")
        
        # Trajectory prediction
        print(f"\n  Trajectory Prediction (next 10 seconds):")
        trajectory = twin.predict_trajectory(horizon=10.0, dt=2.0)
        for i, traj_state in enumerate(trajectory):
            t = (i + 1) * 2
            print(f"    t+{t}s: ({traj_state.position.x:.1f}, {traj_state.position.y:.1f}, {traj_state.position.z:.1f})")
        
        # Resource estimation
        print(f"\n  Resource Estimation:")
        remaining_energy = twin.estimate_remaining_energy()
        remaining_range = twin.estimate_range()
        print(f"    Remaining energy: {remaining_energy:.2f} Wh")
        print(f"    Estimated range: {remaining_range:.1f}m")
    
    # Fleet-level twin
    print("\n" + "="*60)
    print("FLEET DIGITAL TWIN")
    print("="*60)
    
    fleet_twin = engine.fleet_twin
    metrics = fleet_twin.metrics
    
    print(f"\nFleet Metrics:")
    print("-"*40)
    print(f"  Total agents: {metrics.total_agents}")
    print(f"  Active agents: {metrics.active_agents}")
    print(f"  Idle agents: {metrics.idle_agents}")
    
    print(f"\nEnergy Status:")
    print(f"  Average battery: {metrics.avg_battery_level:.1%}")
    print(f"  Minimum battery: {metrics.min_battery_level:.1%}")
    print(f"  Total remaining energy: {metrics.total_remaining_energy_wh:.2f} Wh")
    
    print(f"\nSpatial Distribution:")
    centroid = metrics.centroid
    print(f"  Fleet centroid: ({centroid.x:.2f}, {centroid.y:.2f}, {centroid.z:.2f})")
    print(f"  Fleet spread radius: {metrics.spread_radius:.2f}m")
    
    if metrics.bounding_box:
        bb = metrics.bounding_box
        print(f"  Bounding box: ({bb[0].x:.1f},{bb[0].y:.1f},{bb[0].z:.1f}) to ({bb[1].x:.1f},{bb[1].y:.1f},{bb[1].z:.1f})")
    
    print(f"\nBy Platform Type:")
    for ptype, count in metrics.agents_by_type.items():
        print(f"  {ptype}: {count}")
    
    # Fleet queries
    print(f"\nFleet Queries:")
    print("-"*40)
    
    # Find agents near a point
    nearby = fleet_twin.get_agents_in_radius(Vector3(0, 0, 0), radius=15.0)
    print(f"  Agents within 15m of origin: {nearby}")
    
    # Find nearest agent
    nearest = fleet_twin.get_nearest_agent(Vector3(5, 5, 1))
    if nearest:
        print(f"  Nearest agent to (5,5,1): {nearest}")
    
    print("\n" + "="*60)
    print("Digital Twin Summary")
    print("="*60)
    print()
    print("Digital twins provide:")
    print("  ✓ Real-time state monitoring")
    print("  ✓ Future state prediction")
    print("  ✓ Resource estimation (battery, range)")
    print("  ✓ Fleet-wide metrics aggregation")
    print("  ✓ Spatial queries (find nearby, available agents)")
    print()
    print("In real deployments, twins stay synchronized with")
    print("actual hardware via the synchronizer module.")

if __name__ == "__main__":
    main()
