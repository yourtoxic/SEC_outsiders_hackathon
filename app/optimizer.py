"""Cost-minimizing 24-hour schedule, solved as a linear program (PuLP/CBC)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pulp

EPS = 1e-6


def _solve(
    hours: List[Dict[str, float]],
    battery: Dict[str, float],
    cons: Dict[str, Any],
    use_directives: bool,
) -> Tuple[bool, List[Dict[str, Any]]]:
    cap = battery["capacity_kwh"]
    e0 = battery["initial_energy_kwh"]
    base_min = battery["minimum_energy_kwh"]
    max_c = battery["max_charge_kwh_per_hour"]
    max_d = battery["max_discharge_kwh_per_hour"]

    factor = cons["solar_factor"] if use_directives else {h: 1.0 for h in range(24)}
    reserve = cons["reserve"] if use_directives else {h: 0.0 for h in range(24)}
    no_charge = cons["no_charge"] if use_directives else set()
    no_discharge = cons["no_discharge"] if use_directives else set()
    grid_cap = cons["grid_cap"] if use_directives else {}

    eff_solar = {h: hours[h]["solar_kwh"] * factor[h] for h in range(24)}

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)
    g, s, c, d, e = {}, {}, {}, {}, {}
    for h in range(24):
        g[h] = pulp.LpVariable(
            f"g{h}", 0, grid_cap.get(h) if h in grid_cap else None
        )
        s[h] = pulp.LpVariable(f"s{h}", 0, eff_solar[h])
        c[h] = pulp.LpVariable(f"c{h}", 0, 0 if h in no_charge else max_c)
        d[h] = pulp.LpVariable(f"d{h}", 0, 0 if h in no_discharge else max_d)
        lo = max(base_min, reserve[h])
        e[h] = pulp.LpVariable(f"e{h}", lo, cap)

    prob += pulp.lpSum(g[h] * hours[h]["tariff_bdt_per_kwh"] for h in range(24))

    for h in range(24):
        prob += g[h] + s[h] + d[h] == hours[h]["demand_kwh"] + c[h], f"bal{h}"
        prev = e0 if h == 0 else e[h - 1]
        prob += e[h] == prev + c[h] - d[h], f"state{h}"
    prob += e[23] == e0, "neutral"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return False, []

    plan: List[Dict[str, Any]] = []
    energy = e0
    for h in range(24):
        cv, dv = max(c[h].value() or 0.0, 0.0), max(d[h].value() or 0.0, 0.0)
        net = cv - dv  # simultaneous charge+discharge is energy-neutral: net it out
        if net > EPS:
            action, amount = "charge", net
        elif net < -EPS:
            action, amount = "discharge", -net
        else:
            action, amount = "idle", 0.0
        energy = round(energy + net, 6)

        solar_used = min(max(s[h].value() or 0.0, 0.0), eff_solar[h])
        # grid is derived from the balance equation so it always closes exactly
        grid = max(hours[h]["demand_kwh"] + net - solar_used, 0.0)

        plan.append(
            {
                "hour": h,
                "grid_kwh": round(grid, 4),
                "solar_used_kwh": round(solar_used, 4),
                "battery_action": action,
                "battery_kwh": round(amount, 4),
                "battery_energy_after_kwh": round(energy, 4),
            }
        )
    # force exact end-of-day neutrality against rounding drift
    plan[23]["battery_energy_after_kwh"] = round(e0, 4)
    return True, plan


def optimize(
    hours: List[Dict[str, float]], battery: Dict[str, float], cons: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (plan, directives_applied). Falls back to a directive-free plan
    only if the directive-constrained problem is infeasible."""
    ok, plan = _solve(hours, battery, cons, use_directives=True)
    if ok:
        return plan, True
    ok, plan = _solve(hours, battery, cons, use_directives=False)
    if ok:
        return plan, False
    return _idle_plan(hours, battery), False


def _idle_plan(hours, battery) -> List[Dict[str, Any]]:
    """Last-resort physically valid plan: solar first, grid for the rest."""
    e0 = battery["initial_energy_kwh"]
    plan = []
    for h in range(24):
        solar_used = min(hours[h]["solar_kwh"], hours[h]["demand_kwh"])
        plan.append(
            {
                "hour": h,
                "grid_kwh": round(hours[h]["demand_kwh"] - solar_used, 4),
                "solar_used_kwh": round(solar_used, 4),
                "battery_action": "idle",
                "battery_kwh": 0.0,
                "battery_energy_after_kwh": round(e0, 4),
            }
        )
    return plan


def totals(plan: List[Dict[str, Any]], hours: List[Dict[str, float]]):
    total_grid = sum(p["grid_kwh"] for p in plan)
    cost = sum(p["grid_kwh"] * hours[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    peak = max(p["grid_kwh"] for p in plan)
    return round(total_grid, 4), round(cost, 4), round(peak, 4)
