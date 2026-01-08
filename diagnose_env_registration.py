#!/usr/bin/env python3
"""Diagnostic script to verify environment registration chain."""

import sys
import os

# Debug: Print Python path
print("=" * 80)
print("DEBUG: Python Path")
print("=" * 80)
for i, path in enumerate(sys.path[:5]):
    print(f"  [{i}] {path}")

# Step 1: Import bipedal_locomotion
print("\n" + "=" * 80)
print("STEP 1: Importing bipedal_locomotion...")
print("=" * 80)
try:
    import bipedal_locomotion
    print(f"✓ Successfully imported from: {bipedal_locomotion.__file__}")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Step 2: Import tasks submodule
print("\n" + "=" * 80)
print("STEP 2: Importing bipedal_locomotion.tasks...")
print("=" * 80)
try:
    import bipedal_locomotion.tasks
    print("✓ Successfully imported bipedal_locomotion.tasks")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Step 3: Import locomotion submodule
print("\n" + "=" * 80)
print("STEP 3: Importing bipedal_locomotion.tasks.locomotion...")
print("=" * 80)
try:
    import bipedal_locomotion.tasks.locomotion
    print("✓ Successfully imported bipedal_locomotion.tasks.locomotion")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Step 4: Import robots submodule (this should trigger registration)
print("\n" + "=" * 80)
print("STEP 4: Importing bipedal_locomotion.tasks.locomotion.robots...")
print("=" * 80)
try:
    import bipedal_locomotion.tasks.locomotion.robots
    print("✓ Successfully imported bipedal_locomotion.tasks.locomotion.robots")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Step 5: Check gymnasium registry
print("\n" + "=" * 80)
print("STEP 5: Checking Gymnasium Registry")
print("=" * 80)
try:
    import gymnasium as gym
    
    # List all Isaac-Limx environments
    all_envs = list(gym.envs.registry.keys())
    isaac_limx_envs = [env for env in all_envs if "Isaac-Limx" in env]
    
    print(f"Total environments in registry: {len(all_envs)}")
    print(f"\nFound {len(isaac_limx_envs)} Isaac-Limx environments:")
    for env in sorted(isaac_limx_envs):
        print(f"  ✓ {env}")
    
    # Check specifically for Pronk environments
    print("\n" + "-" * 80)
    if "Isaac-Limx-PF-Pronk-v0" in all_envs:
        print("✓ Isaac-Limx-PF-Pronk-v0 is registered!")
        spec = gym.spec("Isaac-Limx-PF-Pronk-v0")
        print(f"  Entry point: {spec.entry_point}")
        print(f"  Env cfg: {spec.kwargs.get('env_cfg_entry_point')}")
        print(f"  Agent cfg: {spec.kwargs.get('rsl_rl_cfg_entry_point')}")
    else:
        print("✗ Isaac-Limx-PF-Pronk-v0 NOT found in registry!")
        print("This suggests the environment registration failed.")
        
    if "Isaac-Limx-PF-Pronk-Play-v0" in all_envs:
        print("\n✓ Isaac-Limx-PF-Pronk-Play-v0 is registered!")
    else:
        print("\n✗ Isaac-Limx-PF-Pronk-Play-v0 NOT found in registry!")
        
except Exception as e:
    print(f"✗ Error checking registry: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
