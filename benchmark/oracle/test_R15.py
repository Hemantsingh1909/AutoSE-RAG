import pytest

def test_oracle_fota_valid_boot():
    res = validate_sensor(10.0, 0.0, 30.0)
    assert res[0] is True

def test_oracle_fota_boot_watchdog_rollback():
    # Timeout exceeding 30s limit triggers diagnostic rollback event
    res = validate_sensor(35.0, 0.0, 30.0)
    assert res[0] is False
    assert "DIAGNOSTIC" in res[1]
