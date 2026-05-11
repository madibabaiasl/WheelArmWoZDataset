#!/usr/bin/env python3
"""
Quick sanity‑check for DexArmControl.

Run in a ROS 2 environment where the Kinova Gen3 driver (or a simulator)
is already publishing /joint_states and the Cartesian pose topic you
configured in DexArmControl (default: /tf/end_effector2base).

$ chmod +x test_dex_arm.py
$ ./test_dex_arm.py
"""
import time
import sys

try:
    from kinova_gen3_control import DexArmControl          # adjust import path if needed
except ImportError:
    # If dex_arm_control.py is in the same folder:
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from dex_arm_control import DexArmControl


def main():
    arm = DexArmControl(robot_type="kinova_gen3")

    print("Waiting for first joint & cartesian messages …")
    timeout_s = 10.0
    start = time.time()
    while (arm.get_arm_joint_state() is None or
           arm.get_arm_cartesian_state() is None):
        if time.time() - start > timeout_s:
            print(f"Timed‑out (> {timeout_s}s) — did the driver start?")
            arm.shutdown()
            sys.exit(1)
        time.sleep(0.05)

    print("\n✅  Messages received!")
    print("Joint state:")
    print(arm.get_arm_joint_state())       # dictionary with pos/vel/effort/timestamp
    print("\nCartesian state:")
    print(arm.get_arm_cartesian_state())   # dictionary with position/orientation

    arm.shutdown()


if __name__ == "__main__":
    main()
