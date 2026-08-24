import pytest

def test_oracle_misfire_freeze_frame():
    dem = DiagnosticManager(debounce_threshold=1)
    status, _ = dem.monitor_signal("misfire_cyl1", is_valid=False, snapshot={"rpm": 3200, "load": 80})
    assert status == "CONFIRMED"
    assert "DTC_MISFIRE_CYL1_MALFUNCTION" in dem.confirmed_dtcs
