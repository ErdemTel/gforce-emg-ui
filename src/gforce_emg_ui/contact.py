"""Electrode contact quality from per-channel impedance (Ohm)."""

from typing import Optional

from .config import CONTACT_GOOD_KOHM, CONTACT_MARGINAL_KOHM

GOOD, MARGINAL, POOR, UNKNOWN = "good", "marginal", "poor", "unknown"

LEVEL_COLORS = {
    GOOD: "#2e7d32",
    MARGINAL: "#ef6c00",
    POOR: "#c62828",
    UNKNOWN: "#757575",
}


def contact_level(impedance_ohm: Optional[float]) -> str:
    if impedance_ohm is None or impedance_ohm != impedance_ohm:  # None or NaN
        return UNKNOWN
    kohm = impedance_ohm / 1000.0
    if kohm <= CONTACT_GOOD_KOHM:
        return GOOD
    if kohm <= CONTACT_MARGINAL_KOHM:
        return MARGINAL
    return POOR


def contact_color(impedance_ohm: Optional[float]) -> str:
    return LEVEL_COLORS[contact_level(impedance_ohm)]


def format_kohm(impedance_ohm: Optional[float]) -> str:
    if impedance_ohm is None or impedance_ohm != impedance_ohm:
        return "-- kΩ"
    return f"{impedance_ohm / 1000.0:.0f} kΩ"
