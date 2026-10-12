"""Comfort metrics against synthetic timelines."""

from __future__ import annotations

import pytest
from helpers import scenario

from roommind_sim.analysis.load import RunData
from roommind_sim.analysis.metrics import _room_metrics, _truth_beta_s
from roommind_sim.scenario import load_scenario_dict


def _run(temps: list[float], heat: list[float], hold: list[bool] | None = None, step_s: int = 3600) -> RunData:
    """One sample per step; the first one only opens the interval and is not integrated."""
    scn = load_scenario_dict(scenario({"r": {}}))
    t0 = scn.start
    held = hold or [False] * len(temps)
    samples = [{"t": t0 + step_s * i, "rooms": {"r": {"t_air": temp}}} for i, temp in enumerate(temps)]
    observer = [
        {
            "t": t0 + step_s * i,
            "rooms": {
                "r": {
                    "heat_target": h,
                    "cool_target": None,
                    "mode": "idle" if held[i] else "heating",
                    "compressor_protection_active": held[i],
                }
            },
        }
        for i, h in enumerate(heat)
    ]
    return RunData(None, scn, samples, observer, [], t0 + step_s * len(temps), {})  # type: ignore[arg-type]


def test_heating_above_band_counts_as_overshoot():
    m = _room_metrics(_run([21.0, 22.0, 22.5, 22.0], [21.0] * 4), "r")
    assert m["heat_overshoot_kh"] == pytest.approx(0.5 + 1.0 + 0.5)
    assert m["heat_coast_kh"] == 0.0


def test_passive_cooldown_after_setback_is_coasting_not_overshoot():
    temps = [21.5, 21.5, 21.3, 21.0, 20.6, 20.2, 19.8, 19.4]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == 0.0
    assert m["heat_coast_kh"] == pytest.approx(1.8 + 1.5 + 1.1 + 0.7 + 0.3)
    assert m["heat_above_band_kh"] == pytest.approx(m["heat_coast_kh"])


def test_coast_ends_in_band_so_later_overshoot_counts():
    temps = [21.5, 21.5, 20.0, 19.4, 20.5]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_coast_kh"] == pytest.approx(0.5)
    assert m["heat_overshoot_kh"] == pytest.approx(1.0)


def test_rewarming_during_setback_ends_the_coast():
    temps = [21.5, 21.5, 21.0, 20.6, 21.0, 21.4]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_coast_kh"] == pytest.approx(1.5 + 1.1)
    assert m["heat_overshoot_kh"] == pytest.approx(1.5 + 1.9)


def test_preheat_ahead_of_comfort_block_is_not_overshoot():
    temps = [17.5, 17.5, 17.5, 19.5, 20.5, 21.0, 21.0]
    heat = [17.0, 17.0, 17.0, 17.0, 17.0, 21.0, 21.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == 0.0
    assert m["undershoot_kh"] == 0.0
    assert m["heat_above_band_kh"] == pytest.approx(2.0 + 3.0)


def test_preheat_lookahead_is_limited():
    temps = [18.0, 19.5, 19.5, 19.5, 19.5, 19.5]
    heat = [17.0, 17.0, 17.0, 17.0, 17.0, 21.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == pytest.approx(2.0)


def _with_sun(run: RunData, ghi: list[float], solar_w: list[float]) -> RunData:
    for s, g, w in zip(run.samples, ghi, solar_w, strict=True):
        s["outdoor"] = {"ghi": g}
        s["rooms"]["r"]["solar_w"] = w
    return run


def test_truth_beta_s_in_roommind_units():
    run = _with_sun(_run([20.0] * 3, [20.0] * 3), [0.0, 250.0, 125.0], [0.0, 500.0, 250.0])
    assert _truth_beta_s(run, "r", 3.6e6) == pytest.approx(2.0)
    night = _with_sun(_run([20.0] * 2, [20.0] * 2), [0.0, 0.0], [0.0, 0.0])
    assert _truth_beta_s(night, "r", 3.6e6) is None


def test_hold_overshoot_counts_only_while_the_min_run_holds():
    temps = [20.0, 21.0, 21.8, 22.6, 23.0, 22.0]
    hold = [False, False, True, True, False, False]
    m = _room_metrics(_run(temps, [21.0] * 6, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == pytest.approx(1.6)
    assert m["hold_heat_over_kh"] == pytest.approx((0.3 + 1.1) / 6, abs=1e-3)


def test_hold_running_into_a_setback_is_measured_against_the_comfort_target():
    temps = [20.0, 21.0, 21.4, 21.6]
    heat = [21.0, 21.0, 18.0, 18.0]
    hold = [False, False, True, True]
    m = _room_metrics(_run(temps, heat, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == pytest.approx(0.6)


def test_hold_while_preheating_is_measured_against_the_coming_target():
    temps = [18.0, 19.0, 20.5, 21.2]
    heat = [18.0, 18.0, 18.0, 21.0]
    hold = [False, False, True, False]
    m = _room_metrics(_run(temps, heat, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == 0.0
