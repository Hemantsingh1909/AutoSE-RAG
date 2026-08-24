import pytest

def test_oracle_bms_voltage_nominal():
    res = validate_sensor(3.7, 2.5, 4.2)
    assert res[0] is True

def test_oracle_bms_undervoltage():
    res = validate_sensor(2.1, 2.5, 4.2)
    assert res[0] is False

def test_oracle_bms_overvoltage():
    res = validate_sensor(4.5, 2.5, 4.2)
    assert res[0] is False
