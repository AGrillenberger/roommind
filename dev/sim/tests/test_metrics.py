"""Comfort metrics against synthetic timelines."""

from __future__ import annotations

import pytest
from helpers import scenario

from roommind_sim.analysis.load import RunData
from roommind_sim.analysis.metrics import _room_metrics, _truth_beta_s
from roommind_sim.scenario import load_scenario_dict


def _run(temps: list[float], heat: list[float]) -> RunData:
    """One sample per hour; the first one only opens the interval and is not integrated."""
    scn = load_scenario_dict(scenario({"r": {}}))
    t0 = scn.start
    samples = [{"t": t0 + 3600 * i, "rooms": {"r": {"t_air": temp}}} for i, temp in enumerate(temps)]
    observer = [
        {"t": t0 + 3600 * i, "rooms": {"r": {"heat_target": h, "cool_target": None}}} for i, h in enumerate(heat)
    ]
    return RunData(None, scn, samples, observer, [], t0 + 3600 * len(temps), {})  # type: ignore[arg-type]


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
