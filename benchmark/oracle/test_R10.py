import pytest

def test_oracle_uds_snapshot_lookup():
    dem = DiagnosticManager(debounce_threshold=1)
    snapshot = {"speed": 100, "voltage": 12.4, "timestamp": 123456}
    status, _ = dem.monitor_signal("uds_fault", is_valid=False, snapshot=snapshot)
    assert status == "CONFIRMED"
    assert "DTC_UDS_FAULT_MALFUNCTION" in dem.confirmed_dtcs
