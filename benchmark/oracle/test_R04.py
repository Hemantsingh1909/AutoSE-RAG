import pytest

def test_oracle_dtc_debounce_progression():
    dem = DiagnosticManager(debounce_threshold=3)
    s1, _ = dem.monitor_signal("coolant", is_valid=False)
    assert s1 == "PENDING", "First fault must be PENDING"
    s2, _ = dem.monitor_signal("coolant", is_valid=False)
    assert s2 == "PENDING"
    s3, _ = dem.monitor_signal("coolant", is_valid=False, snapshot={"temp": 140})
    assert s3 == "CONFIRMED", "Third consecutive fault must be CONFIRMED"
    assert "DTC_COOLANT_MALFUNCTION" in dem.confirmed_dtcs or len(dem.confirmed_dtcs) > 0
