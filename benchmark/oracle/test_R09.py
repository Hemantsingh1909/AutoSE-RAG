import pytest

def test_oracle_steering_angle_in_bounds():
    res = validate_sensor(180.0, -540.0, 540.0)
    assert res[0] is True

def test_oracle_steering_angle_overflow():
    res = validate_sensor(999.0, -540.0, 540.0)
    assert res[0] is False
