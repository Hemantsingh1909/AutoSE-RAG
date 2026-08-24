import pytest

def test_oracle_nominal_arrival():
    res = validate_sensor(10.0, 0.0, 15.0)
    assert res[0] is True

def test_oracle_bus_timeout():
    res = validate_sensor(25.0, 0.0, 15.0)
    assert res[0] is False
