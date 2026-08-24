import zlib
import pytest

def test_oracle_e2e_nominal_success():
    val = E2EFrameValidator(data_id=0x1234)
    payload = b"speed=80"
    header = (0x1234).to_bytes(2, 'big') + (0).to_bytes(1, 'big')
    crc = zlib.crc32(header + payload) & 0xFFFFFFFF
    ok, msg = val.validate_and_unpack(payload, counter=0, received_crc=crc)
    assert ok is True

def test_oracle_e2e_bitflip_crc_error():
    val = E2EFrameValidator(data_id=0x1234)
    payload = b"speed=80"
    ok, msg = val.validate_and_unpack(payload, counter=0, received_crc=0x12345678)
    assert ok is False, "Corrupted CRC must be rejected"

def test_oracle_e2e_alive_counter_skip():
    val = E2EFrameValidator(data_id=0x1234)
    payload = b"speed=80"
    header = (0x1234).to_bytes(2, 'big') + (10).to_bytes(1, 'big')
    crc = zlib.crc32(header + payload) & 0xFFFFFFFF
    ok, msg = val.validate_and_unpack(payload, counter=10, received_crc=crc)
    assert ok is False, "Skipped sequence counter must be rejected"
