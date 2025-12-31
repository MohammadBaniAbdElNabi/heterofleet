#!/usr/bin/env python
"""
HeteroFleet - Heterogeneous Autonomous Vehicle Swarm Coordination Framework

Main entry point for running demonstrations and tests.
"""

import sys
import argparse
from loguru import logger


def run_demo():
    """Run a quick demonstration of HeteroFleet capabilities."""
    from heterofleet.core.platform import Vector3
    from heterofleet.simulation.engine import SimulationEngine, SimulationConfig
    
    print("=" * 60)
    print("HeteroFleet Demonstration")
    print("=" * 60)
    print()
    
    # Create simulation
    config = SimulationConfig(
        time_step=0.05,
        real_time_factor=float('inf'),
        max_duration=10.0,
        num_uavs=3,
        num_ugvs=1,
        enable_coordination=True,
    )
    
    engine = SimulationEngine(config)
    engine.setup_default_scenario()
    
    print(f"Created fleet with {len(engine._agents)} agents:")
    for aid, agent in engine._agents.items():
        print(f"  - {aid}: {agent.platform_spec.platform_type.name}")
    
    # Set targets
    targets = {
        "uav_0": Vector3(10, 0, 2),
        "uav_1": Vector3(-5, 10, 2),
        "uav_2": Vector3(-5, -10, 2),
        "ugv_0": Vector3(5, -5, 0),
    }
    
    print("\nAssigning targets...")
    for aid, target in targets.items():
        if aid in engine._agents:
            engine.set_agent_target(aid, target)
            print(f"  {aid} -> ({target.x}, {target.y}, {target.z})")
    
    print("\nRunning simulation...")
    engine.run(duration=10.0)
    
    state = engine.state
    print(f"\nSimulation completed:")
    print(f"  Time: {state.sim_time:.2f}s")
    print(f"  Steps: {state.step_count}")
    print(f"  Avg step time: {state.avg_step_time_ms:.2f}ms")
    
    print("\nFinal agent positions:")
    for aid, agent in engine._agents.items():
        target = targets.get(aid, Vector3(0, 0, 0))
        dist = ((agent.position.x - target.x)**2 + 
                (agent.position.y - target.y)**2 + 
                (agent.position.z - target.z)**2) ** 0.5
        print(f"  {aid}: ({agent.position.x:.2f}, {agent.position.y:.2f}, {agent.position.z:.2f}) "
              f"[dist to target: {dist:.2f}m]")
    
    metrics = engine.fleet_twin.metrics
    print(f"\nFleet metrics:")
    print(f"  Total agents: {metrics.total_agents}")
    print(f"  Avg battery: {metrics.avg_battery_level:.1%}")
    print(f"  Fleet spread: {metrics.spread_radius:.2f}m")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


def run_tests():
    """Run module import tests."""
    print("Testing HeteroFleet module imports...")
    
    modules = [
        ("Core Platform", "heterofleet.core.platform"),
        ("Core State", "heterofleet.core.state"),
        ("Core Message", "heterofleet.core.message"),
        ("Simulation Environment", "heterofleet.simulation.environment"),
        ("Simulation Agent", "heterofleet.simulation.agent_sim"),
        ("Simulation Engine", "heterofleet.simulation.engine"),
        ("Planning Task", "heterofleet.planning.task"),
        ("Planning NSGA-III", "heterofleet.planning.nsga3"),
        ("Planning MOPOTA", "heterofleet.planning.mopota"),
        ("Safety Collision", "heterofleet.safety.collision"),
        ("Safety STL Monitor", "heterofleet.safety.stl_monitor"),
        ("Digital Twin Agent", "heterofleet.digital_twin.agent_twin"),
        ("Digital Twin Fleet", "heterofleet.digital_twin.fleet_twin"),
        ("Coordination HAIM", "heterofleet.coordination.haim"),
        ("Hardware Crazyflie", "heterofleet.hardware.crazyflie"),
        ("Hardware MentorPi", "heterofleet.hardware.mentorpi"),
        ("Visualization", "heterofleet.visualization.dashboard"),
        ("Communication", "heterofleet.communication.routing"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="HeteroFleet Framework")
    parser.add_argument("command", choices=["demo", "test"], 
                       help="Command to run")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    if args.command == "demo":
        run_demo()
    elif args.command == "test":
        success = run_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
