#!/usr/bin/env python3
"""Wrapper for the canonical IMU rotate node in xjrobot_localization."""

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parents[3]
    / "src/nav/xjrobot_localization/scripts/imu_rotate_to_lidar_frame.py"
)

if not TARGET.is_file():
    raise FileNotFoundError(f"Canonical IMU rotate script not found: {TARGET}")

runpy.run_path(str(TARGET), run_name="__main__")
