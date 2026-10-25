"""Read-only TFRecord framing counts bound to sealed file hashes; no TF/GPU."""
import argparse
import json
from pathlib import Path
import struct

import train_pytorch_recipe as recipe


def record_count(path):
    size = path.stat().st_size
    count = 0
    with path.open('rb') as stream:
        while stream.tell() < size:
            header = stream.read(12)
            assert len(header) == 12, 'Truncated TFRecord header'
            length = struct.unpack('<Q', header[:8])[0]
            assert length > 0 and stream.tell()+length+4 <= size, 'Truncated TFRecord payload'
            stream.seek(length+4, 1)
            count += 1
        assert stream.tell() == size
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    identity = json.loads(args.identity.read_text())
    manifest = json.loads((recipe.ROOT/'research/data/manifest.json').read_text())
    by_session = {}
    for relative, expected in identity['tfrecord_hashes'].items():
        path = recipe.ROOT/relative
        assert recipe.sha(path) == expected
        session, split = path.parts[-3:-1]
        by_session.setdefault(session, {}).setdefault(split, 0)
        by_session[session][split] += record_count(path)
    assert len(by_session) == 19
    for session, splits in by_session.items():
        assert set(splits) == {'train', 'test'}
        for split, count in splits.items():
            assert count == len(manifest['sessions'][session][split]['trials'])
    totals = {split: sum(day[split] for day in by_session.values()) for split in ('train', 'test')}
    assert totals == {'train': 6640, 'test': 680}
    result = dict(passed=True, files=len(identity['tfrecord_hashes']), sessions=19, totals=totals,
        by_session=by_session, identity_sha256=recipe.sha(args.identity),
        manifest_sha256=recipe.sha(recipe.ROOT/'research/data/manifest.json'),
        auditor_sha256=recipe.sha(Path(__file__)),
        scope='Hash-bound framing record counts; CRC/feature/label parsing not independently tested here')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
