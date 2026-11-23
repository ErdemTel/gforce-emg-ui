"""Bounds shared with the robot's independent direct-gain validation (no ROS dependency)."""

import math
import numbers


def validate_pair(kp, kd):
    for name, value in (("Kp", kp), ("Kd", kd)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} multiplier must be finite and between 0 and 1")
    return float(kp), float(kd)
