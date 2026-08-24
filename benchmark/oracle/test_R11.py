import zlib
import pytest

def test_oracle_alive_counter_stuck():
    val = E2EFrameValidator(data_id=0x55)
    payload = b"can_fd_msg"
    header = (0x55).to_bytes(2, 'big') + (0).to_bytes(1, 'big')
    crc = zlib.crc32(header + payload) & 0xFFFFFFFF
    ok1, _ = val.validate_and_unpack(payload, counter=0, received_crc=crc)
    assert ok1 is True
    ok2, _ = val.validate_and_unpack(payload, counter=0, received_crc=crc)
    assert ok2 is False
