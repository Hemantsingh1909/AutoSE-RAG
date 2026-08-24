import pytest

def test_oracle_nominal_match():
    res = validate_throttle_pedal(2.0, 2.1, max_discrepancy=0.2)
    assert res[0] is True, "Valid inputs must pass validation"

def test_oracle_lower_boundary_violation():
    res = validate_throttle_pedal(0.4, 2.0)
    assert res[0] is False, "Voltage < 0.5V must be rejected"
    assert "ELECTRICAL" in str(res[1]).upper() or "FAULT" in str(res[1]).upper()

def test_oracle_upper_boundary_violation():
    res = validate_throttle_pedal(2.0, 4.6)
    assert res[0] is False, "Voltage > 4.5V must be rejected"

def test_oracle_cross_sensor_discrepancy():
    res = validate_throttle_pedal(1.0, 1.5, max_discrepancy=0.2)
    assert res[0] is False, "Discrepancy > 0.2V must be flagged as plausibility fault"
