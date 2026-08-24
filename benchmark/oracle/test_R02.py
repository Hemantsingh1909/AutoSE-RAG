import pytest

def test_oracle_watchdog_nominal():
    res = validate_sensor(40.0, 0.0, 50.0)
    assert res[0] is True

def test_oracle_watchdog_ftti_violation():
    res = validate_sensor(55.0, 0.0, 50.0)
    assert res[0] is False
    assert "DIAGNOSTIC" in str(res[1]).upper() or "EVENT" in str(res[1]).upper()
