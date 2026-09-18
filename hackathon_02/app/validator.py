"""Final replay validator — the same checks the judge harness performs."""
from __future__ import annotations

from typing import Any, Dict, List

TOL = 0.01


def replay(
    plan: List[Dict[str, Any]],
    hours: List[Dict[str, float]],
    battery: Dict[str, float],
    cons: Dict[str, Any],
) -> List[str]:
    issues: List[str] = []
    cap = battery["capacity_kwh"]
    e0 = battery["initial_energy_kwh"]
    base_min = battery["minimum_energy_kwh"]
    max_c = battery["max_charge_kwh_per_hour"]
    max_d = battery["max_discharge_kwh_per_hour"]

    if len(plan) != 24 or sorted(p["hour"] for p in plan) != list(range(24)):
        return ["hourly_plan must contain hours 0..23 exactly once"]

    energy = e0
    for p in sorted(plan, key=lambda x: x["hour"]):
        h = p["hour"]
        g, s, a, b = p["grid_kwh"], p["solar_used_kwh"], p["battery_action"], p["battery_kwh"]

        if min(g, s, b) < -TOL:
            issues.append(f"h{h}: negative value")
        eff = hours[h]["solar_kwh"] * cons["solar_factor"][h]
        if s > eff + TOL:
            issues.append(f"h{h}: solar_used exceeds effective solar")

        charge = b if a == "charge" else 0.0
        discharge = b if a == "discharge" else 0.0
        if a == "idle" and abs(b) > TOL:
            issues.append(f"h{h}: idle hour with non-zero battery_kwh")
        if charge > max_c + TOL:
            issues.append(f"h{h}: charge rate exceeded")
        if discharge > max_d + TOL:
            issues.append(f"h{h}: discharge rate exceeded")
        if h in cons["no_charge"] and charge > TOL:
            issues.append(f"h{h}: charging inside no_charge_window")
        if h in cons["no_discharge"] and discharge > TOL:
            issues.append(f"h{h}: discharging inside no_discharge_window")
        if h in cons["grid_cap"] and g > cons["grid_cap"][h] + TOL:
            issues.append(f"h{h}: grid import above max_grid_kwh")

        if abs((g + s + discharge) - (hours[h]["demand_kwh"] + charge)) > TOL:
            issues.append(f"h{h}: energy balance broken")

        energy = energy + charge - discharge
        floor = max(base_min, cons["reserve"][h])
        if energy < floor - TOL or energy > cap + TOL:
            issues.append(f"h{h}: battery energy out of bounds")
        if abs(energy - p["battery_energy_after_kwh"]) > TOL:
            issues.append(f"h{h}: battery_energy_after_kwh mismatch")

    if abs(energy - e0) > TOL:
        issues.append("end-of-day battery neutrality broken")
    return issues
