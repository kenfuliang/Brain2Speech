import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from audit_willett_record_counts import record_count


def record(payload):
    return struct.pack('<Q', len(payload))+b'0000'+payload+b'0000'


def test_records_counted(tmp_path):
    path = tmp_path/'records'
    path.write_bytes(record(b'a')+record(b'abc'))
    assert record_count(path) == 2


@pytest.mark.parametrize('value', [b'a', record(b'abc')[:-1], struct.pack('<Q', 100)+b'0000'])
def test_truncation_rejected(tmp_path, value):
    path = tmp_path/'records'
    path.write_bytes(value)
    with pytest.raises(AssertionError):
        record_count(path)
