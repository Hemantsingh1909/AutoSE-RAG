import pytest

def test_oracle_map_sensor_nominal():
    res = validate_sensor(100.0, 20.0, 250.0)
    assert res[0] is True

def test_oracle_map_sensor_electrical_short():
    res = validate_sensor(5.0, 20.0, 250.0)
    assert res[0] is False
