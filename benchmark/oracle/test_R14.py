import pytest

def test_oracle_hvil_normal_impedance():
    res = validate_sensor(2.5, 0.0, 10.0)
    assert res[0] is True

def test_oracle_hvil_open_circuit():
    res = validate_sensor(25.0, 0.0, 10.0)
    assert res[0] is False
    assert "DIAGNOSTIC" in res[1]
