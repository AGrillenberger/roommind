"""EKF training label from the delivered power of the driven devices."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import MODE_COOLING, MODE_HEATING
from custom_components.roommind.managers.heat_source_orchestrator import DeviceCommand, HeatSourcePlan

from .conftest import _create_coordinator, _make_store_mock


def _room(*eids: str) -> dict:
    return {"devices": [{"entity_id": e, "type": "ac" if "ac" in e else "trv"} for e in eids]}


def _states(**by_eid):
    def get(eid):
        entry = by_eid.get(eid.replace("climate.", ""))
        if entry is None:
            return None
        state, action = entry
        attrs = {} if action is None else {"hvac_action": action}
        return SimpleNamespace(state=state, attributes=attrs)

    return get


@pytest.fixture
def coordinator(hass, mock_config_entry):
    hass.data = {"roommind": {"store": _make_store_mock({})}}
    return _create_coordinator(hass, mock_config_entry)


def _label(coordinator, room, mode, pf, states, plan=None):
    coordinator.hass.states.get = MagicMock(side_effect=states)
    return coordinator._delivered_training_power(room, mode, pf, plan)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("heating", 1.0),
        ("preheating", 1.0),
        ("idle", 0.0),
        ("cooling", 0.0),
        ("defrosting", 0.0),
        (None, 0.2),
        ("fan", 0.2),
    ],
)
def test_single_device_label(coordinator, action, expected):
    label = _label(coordinator, _room("climate.trv"), MODE_HEATING, 0.2, _states(trv=("heat", action)))
    assert label == expected


def test_mean_over_driven_devices_skips_off_and_unavailable(coordinator):
    room = _room("climate.trv1", "climate.trv2", "climate.trv3", "climate.trv4", "climate.trv5")
    states = _states(
        trv1=("heat", "heating"),
        trv2=("heat", "idle"),
        trv3=("off", "off"),
        trv4=("unavailable", None),
    )
    assert _label(coordinator, room, MODE_HEATING, 0.1, states) == 0.5


def test_cooling_counts_cooling_only(coordinator):
    room = _room("climate.ac", "climate.trv")
    states = _states(ac=("cool", "cooling"), trv=("off", "off"))
    assert _label(coordinator, room, MODE_COOLING, 0.1, states) == 1.0


def test_no_observable_device_keeps_commanded_pf(coordinator):
    assert _label(coordinator, _room("climate.trv"), MODE_HEATING, 0.3, _states()) == 0.3


def test_orchestrated_parked_device_counts_only_while_running(coordinator):
    plan = HeatSourcePlan(
        commands=[
            DeviceCommand("climate.trv", "primary", "thermostat", True, 0.4, "active"),
            DeviceCommand("climate.ac", "secondary", "ac", False, 0.0, "parked"),
        ],
        active_sources="primary",
        reason="test",
    )
    room = _room("climate.trv", "climate.ac")
    held = _states(trv=("heat", None), ac=("heat", "heating"))
    assert _label(coordinator, room, MODE_HEATING, 0.9, held, plan) == pytest.approx(0.7)
    parked = _states(trv=("heat", None), ac=("heat", "idle"))
    assert _label(coordinator, room, MODE_HEATING, 0.9, parked, plan) == 0.4
