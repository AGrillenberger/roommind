"""MPC must not toggle a fast system heat/idle every cycle around the target.

Seen in the simulation with split ACs: the air node heats within a minute and
relaxes into the walls right after a stop, so the optimizer (no memory, no
hysteresis) restarted the device on every other 30 s cycle (~45 commands/h).
"""

from __future__ import annotations

import time

import pytest

from custom_components.roommind.const import MIN_POWER_FRACTION, TargetTemps
from custom_components.roommind.control.mpc_controller import (
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    MPCController,
)
from custom_components.roommind.control.mpc_optimizer import MPCOptimizer, MPCPlan
from custom_components.roommind.control.thermal_model import RCModel, RoomModelManager

from .conftest import build_hass, make_room

HEAT_ROOM = {"climate_mode": "heat_only"}
COOL_ROOM = {"acs": ["climate.ac1"], "thermostats": [], "climate_mode": "cool_only"}


def _controller(room_overrides=None, **kwargs) -> MPCController:
    room = make_room(**(room_overrides or HEAT_ROOM))
    return MPCController(
        build_hass(),
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=kwargs.pop("outdoor_temp", 5.0),
        settings={},
        has_external_sensor=True,
        **kwargs,
    )


def _fake_plan(monkeypatch, action: str, pf: float = 0.6, lookahead_blocks: int = 6) -> None:
    plan = MPCPlan(
        actions=[action] * 24,
        temperatures=[21.0] * 25,
        power_fractions=[pf if action != MODE_IDLE else 0.0] * 24,
        lookahead_blocks=lookahead_blocks,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: plan,
    )


def _within_min_run(mode: str) -> dict:
    return {"previous_mode": mode, "mode_on_since": time.time() - 60}


# --- starting -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("temp", "expected"),
    [
        (21.0, MODE_IDLE),
        (20.9, MODE_IDLE),
        (20.899999999999995, MODE_IDLE),
        (20.8, MODE_HEATING),
        (20.800000000000004, MODE_HEATING),
    ],
)
def test_heating_starts_only_past_the_hysteresis(monkeypatch, temp, expected):
    _fake_plan(monkeypatch, MODE_HEATING)
    mode, _ = _controller()._evaluate_mpc(temp, TargetTemps(heat=21.0, cool=24.0))
    assert mode == expected


def test_running_heat_is_not_stopped_by_the_start_hysteresis(monkeypatch):
    _fake_plan(monkeypatch, MODE_HEATING, pf=0.6)
    ctrl = _controller(previous_mode=MODE_HEATING, mode_on_since=time.time() - 3600)
    assert ctrl._evaluate_mpc(20.9, TargetTemps(heat=21.0, cool=24.0)) == (MODE_HEATING, 0.6)


def test_preheat_for_a_rising_target_may_start_early(monkeypatch):
    """A schedule step up inside the guard horizon keeps pre-heating the optimizer's call."""
    _fake_plan(monkeypatch, MODE_HEATING, pf=1.0)
    rise_at = time.time() + 15 * 60
    ctrl = _controller(
        target_resolver=lambda ts: TargetTemps(heat=21.0 if ts >= rise_at else 18.0, cool=24.0),
    )
    mode, pf = ctrl._evaluate_mpc(20.5, TargetTemps(heat=18.0, cool=24.0))
    assert (mode, pf) == (MODE_HEATING, 1.0)


@pytest.mark.parametrize(
    ("temp", "expected"),
    [
        (24.0, MODE_IDLE),
        (24.1, MODE_IDLE),
        (24.100000000000005, MODE_IDLE),
        (24.2, MODE_COOLING),
        (24.199999999999996, MODE_COOLING),
    ],
)
def test_cooling_starts_only_past_the_hysteresis(monkeypatch, temp, expected):
    _fake_plan(monkeypatch, MODE_COOLING)
    mode, _ = _controller(COOL_ROOM, outdoor_temp=30.0)._evaluate_mpc(temp, TargetTemps(heat=None, cool=24.0))
    assert mode == expected


def test_precool_for_a_falling_target_may_start_early(monkeypatch):
    _fake_plan(monkeypatch, MODE_COOLING, pf=1.0)
    drop_at = time.time() + 15 * 60
    ctrl = _controller(
        COOL_ROOM,
        outdoor_temp=30.0,
        target_resolver=lambda ts: TargetTemps(heat=None, cool=24.0 if ts >= drop_at else 27.0),
    )
    assert ctrl._evaluate_mpc(25.0, TargetTemps(heat=None, cool=27.0)) == (MODE_COOLING, 1.0)


# --- stopping -------------------------------------------------------------------


def test_heating_run_survives_an_idle_plan_until_min_run_is_over(monkeypatch):
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(**_within_min_run(MODE_HEATING))
    mode, pf = ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_HEATING
    assert MIN_POWER_FRACTION <= pf < 1.0


def test_gentle_power_follows_the_closed_form_below_target(monkeypatch):
    """Below the target the kept run uses the proportional power, not just the floor."""
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(**_within_min_run(MODE_HEATING))
    _, pf = ctrl._evaluate_mpc(19.0, TargetTemps(heat=21.0, cool=24.0))
    assert pf > MIN_POWER_FRACTION


def test_heating_run_ends_after_min_run(monkeypatch):
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(previous_mode=MODE_HEATING, mode_on_since=time.time() - 3600)
    assert ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=24.0)) == (MODE_IDLE, 0.0)


def test_heating_run_ends_when_heating_is_no_longer_allowed(monkeypatch):
    """Outdoor gate (heating_max 22) closed: the min run does not force heat."""
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(outdoor_temp=25.0, **_within_min_run(MODE_HEATING))
    assert ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=24.0)) == (MODE_IDLE, 0.0)


def test_hard_ceiling_still_ends_a_fast_run_inside_min_run(monkeypatch):
    _fake_plan(monkeypatch, MODE_HEATING, pf=1.0)
    ctrl = _controller(**_within_min_run(MODE_HEATING))
    assert ctrl._evaluate_mpc(22.1, TargetTemps(heat=21.0, cool=24.0)) == (MODE_IDLE, 0.0)


def test_idle_plan_past_the_ceiling_is_not_reenabled(monkeypatch):
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(**_within_min_run(MODE_HEATING))
    assert ctrl._evaluate_mpc(22.1, TargetTemps(heat=21.0, cool=24.0)) == (MODE_IDLE, 0.0)


def test_cooling_run_survives_an_idle_plan_until_min_run_is_over(monkeypatch):
    _fake_plan(monkeypatch, MODE_IDLE)
    ctrl = _controller(COOL_ROOM, outdoor_temp=30.0, **_within_min_run(MODE_COOLING))
    mode, pf = ctrl._evaluate_mpc(24.0, TargetTemps(heat=None, cool=24.0))
    assert mode == MODE_COOLING
    assert MIN_POWER_FRACTION <= pf < 1.0


# --- slow systems keep the optimizer's decision ----------------------------------


def test_slow_system_skips_the_hysteresis(monkeypatch):
    _fake_plan(monkeypatch, MODE_HEATING, pf=0.8, lookahead_blocks=24)
    ctrl = _controller({**HEAT_ROOM, "heating_system_type": "underfloor"}, heating_system_type="underfloor")
    assert ctrl._evaluate_mpc(20.9, TargetTemps(heat=21.0, cool=24.0)) == (MODE_HEATING, 0.8)


# --- optimizer: candidates are judged at the power they would get ------------------


def test_strong_device_below_target_is_not_left_idle():
    """Large learned Q (AC heating the air node): full-power look-ahead overshot the
    cool side of the band, so idle won and the room sat ~0.7 K below target."""
    model = RCModel(C=1.0, U=0.01, Q_heat=32.0, Q_cool=4.0)
    opt = MPCOptimizer(model=model, can_heat=True, can_cool=False, w_comfort=7.0, w_energy=3.0, min_run_blocks=2)
    n = 24
    plan = opt.optimize(
        T_room=20.4,
        T_outdoor_series=[0.0] * n,
        heat_target_series=[21.0] * n,
        cool_target_series=[24.5] * n,
        dt_minutes=5,
    )
    assert plan.actions[0] == MODE_HEATING
    assert plan.power_fractions[0] < 1.0


def test_hypothesis_power_fraction():
    model = RCModel(C=1.0, U=0.05, Q_heat=10.0, Q_cool=10.0)
    opt = MPCOptimizer(model=model, can_heat=True, can_cool=True)
    kw = {"q_solar": 0.0, "q_residual": 0.0, "q_occupancy": 0.0}
    below = opt._hypothesis_power_fraction(MODE_HEATING, 20.8, 5.0, 21.0, 5.0, **kw)
    assert 0.0 < below < 1.0
    # at/above target there is no proportional answer -> full power, as before
    assert opt._hypothesis_power_fraction(MODE_HEATING, 22.0, 5.0, 21.0, 5.0, **kw) == 1.0
    # closed form answers with the other mode -> full power for this candidate
    assert opt._hypothesis_power_fraction(MODE_COOLING, 20.0, 5.0, 21.0, 5.0, **kw) == 1.0


def test_hybrid_ufh_and_ac_room_keeps_ufh_preheat(monkeypatch):
    """UFH + AC (auto): the optimizer keeps the base lookahead because the room can cool,
    but the UFH side is still slow and keeps its drift pre-heat (#131)."""
    _fake_plan(monkeypatch, MODE_HEATING, pf=0.8, lookahead_blocks=6)
    ctrl = _controller(
        {"climate_mode": "auto", "acs": ["climate.ac1"], "heating_system_type": "underfloor"},
        heating_system_type="underfloor",
    )
    assert ctrl._predict_idle_drift(21.0, 30.0) < 20.8
    assert ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=24.0)) == (MODE_HEATING, 0.8)


def test_radiator_counts_as_fast(monkeypatch):
    _fake_plan(monkeypatch, MODE_HEATING)
    ctrl = _controller({**HEAT_ROOM, "heating_system_type": "radiator"}, heating_system_type="radiator")
    assert ctrl._evaluate_mpc(20.9, TargetTemps(heat=21.0, cool=24.0)) == (MODE_IDLE, 0.0)
