#!/usr/bin/env python
"""
Tutorial 5: Hardware Quickstart
===============================
Quick reference for connecting to real hardware.

NOTE: This tutorial requires actual hardware!
- Crazyflie 2.1 with Crazyradio PA
- MentorPi or similar ground robot

Run: python tutorials/tutorial_05_hardware.py
"""

import sys
sys.path.insert(0, '.')

import asyncio

# Check if hardware libraries are available
try:
    import cflib
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

async def crazyflie_demo():
    """Demonstrate Crazyflie connection (requires hardware)."""
    from heterofleet.hardware.crazyflie import CrazyflieInterface, CrazyflieConfig
    from heterofleet.core.platform import Vector3
    
    print_header("CRAZYFLIE DEMO")
    
    # Configuration - UPDATE THIS TO YOUR DRONE'S URI
    config = CrazyflieConfig(
        uri="radio://0/80/2M/E7E7E7E7E7",  # Change this!
        default_height=0.5,
        use_lighthouse=True,
        use_multiranger=True,
    )
    
    print(f"Connecting to: {config.uri}")
    print("(Change the URI in the script to match your drone)")
    print()
    
    cf = CrazyflieInterface("uav_0", config)
    
    if await cf.connect():
        print("✓ Connected!")
        
        # Get telemetry
        telemetry = await cf.get_telemetry()
        print(f"\nTelemetry:")
        print(f"  Position: ({telemetry.position.x:.2f}, {telemetry.position.y:.2f}, {telemetry.position.z:.2f})")
        print(f"  Battery: {telemetry.battery_percentage:.0f}%")
        
        # Safety check
        if telemetry.battery_percentage < 20:
            print("\n⚠️ Battery too low for flight!")
            await cf.disconnect()
            return
        
        # Confirm before flying
        print("\n" + "-"*40)
        response = input("Ready to fly? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            await cf.disconnect()
            return
        
        try:
            # Arm
            await cf.arm()
            print("Armed!")
            
            # Takeoff
            print("Taking off...")
            await cf.takeoff(altitude=0.5)
            await asyncio.sleep(2.0)
            
            # Hover
            print("Hovering for 3 seconds...")
            await asyncio.sleep(3.0)
            
            # Land
            print("Landing...")
            await cf.land()
            
            # Disarm
            await cf.disarm()
            print("✓ Flight complete!")
            
        except Exception as e:
            print(f"Error during flight: {e}")
            await cf.emergency_stop()
        
        await cf.disconnect()
    else:
        print("✗ Failed to connect!")
        print("\nTroubleshooting:")
        print("  1. Is the Crazyradio PA plugged in?")
        print("  2. Is the Crazyflie powered on?")
        print("  3. Check the URI matches your drone")
        print("  4. On Linux, check USB permissions")

async def mentorpi_demo():
    """Demonstrate MentorPi connection (requires hardware)."""
    from heterofleet.hardware.mentorpi import MentorPiInterface, MentorPiConfig
    from heterofleet.core.platform import Vector3
    
    print_header("MENTORPI DEMO")
    
    # Configuration - UPDATE THIS TO YOUR ROBOT'S IP
    config = MentorPiConfig(
        host="192.168.1.100",  # Change this!
        port=8080,
        max_speed=0.3,
    )
    
    print(f"Connecting to: {config.host}:{config.port}")
    print("(Change the IP in the script to match your robot)")
    print()
    
    robot = MentorPiInterface("ugv_0", config)
    
    if await robot.connect():
        print("✓ Connected!")
        
        # Get telemetry
        telemetry = await robot.get_telemetry()
        print(f"\nTelemetry:")
        print(f"  Position: ({telemetry.position.x:.2f}, {telemetry.position.y:.2f})")
        print(f"  Battery: {telemetry.battery_percentage:.0f}%")
        
        # Confirm before moving
        print("\n" + "-"*40)
        response = input("Ready to move? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            await robot.disconnect()
            return
        
        try:
            await robot.arm()
            print("Armed!")
            
            # Move forward
            print("Moving forward...")
            await robot.set_velocity(Vector3(0.2, 0, 0))
            await asyncio.sleep(2.0)
            
            # Stop
            await robot.set_velocity(Vector3(0, 0, 0))
            print("Stopped!")
            
            await robot.disarm()
            print("✓ Movement complete!")
            
        except Exception as e:
            print(f"Error: {e}")
            await robot.emergency_stop()
        
        await robot.disconnect()
    else:
        print("✗ Failed to connect!")
        print("\nTroubleshooting:")
        print("  1. Is the MentorPi powered on?")
        print("  2. Is the server running on the Pi?")
        print("  3. Check network connectivity")
        print("  4. Verify IP address and port")

def main():
    print("="*60)
    print("Tutorial 5: Hardware Quickstart")
    print("="*60)
    print()
    print("This tutorial demonstrates real hardware connections.")
    print()
    print("IMPORTANT: Only run this with actual hardware!")
    print("           Make sure you have a safe flying/driving area!")
    print()
    
    # Check cflib
    if CFLIB_AVAILABLE:
        print("✓ cflib is installed (Crazyflie support available)")
    else:
        print("✗ cflib not installed")
        print("  Install with: pip install cflib")
    print()
    
    print("Available demos:")
    print("  1. Crazyflie drone")
    print("  2. MentorPi ground robot")
    print("  3. Exit")
    print()
    
    choice = input("Select demo (1/2/3): ")
    
    if choice == "1":
        if not CFLIB_AVAILABLE:
            print("\ncflib required! Install with: pip install cflib")
            return
        asyncio.run(crazyflie_demo())
    elif choice == "2":
        asyncio.run(mentorpi_demo())
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
