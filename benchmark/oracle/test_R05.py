import pytest

def test_oracle_brake_synchronized():
    res = validate_throttle_pedal(3.0, 3.05, max_discrepancy=0.2)
    assert res[0] is True

def test_oracle_brake_discrepancy_backup():
    res = validate_throttle_pedal(2.0, 3.0, max_discrepancy=0.2)
    assert res[0] is False
    assert "PLAUSIBILITY" in res[1]
