def validate_sensor(value, minimum, maximum):
    if value < minimum or value > maximum:
        return False, "diagnostic: invalid sensor value"
    return True, "ok"


def test_valid_value():
    assert validate_sensor(50, 0, 100)[0] is True


def test_lower_boundary():
    assert validate_sensor(0, 0, 100)[0] is True


def test_upper_boundary():
    assert validate_sensor(100, 0, 100)[0] is True


def test_below_range():
    ok, diagnostic = validate_sensor(-1, 0, 100)
    assert ok is False
    assert "diagnostic" in diagnostic


def test_above_range():
    ok, diagnostic = validate_sensor(101, 0, 100)
    assert ok is False
    assert "diagnostic" in diagnostic
