import pytest

def test_oracle_dtc_healing():
    dem = DiagnosticManager(debounce_threshold=2)
    dem.monitor_signal("oxygen_sensor", is_valid=False)
    dem.monitor_signal("oxygen_sensor", is_valid=False)
    status, _ = dem.monitor_signal("oxygen_sensor", is_valid=True)
    assert status == "PASSED"
