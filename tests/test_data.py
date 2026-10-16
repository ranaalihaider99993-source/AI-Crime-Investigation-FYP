from pathlib import Path
import csv
import cv2
import numpy as np
from scripts.data import inspect, split_manifest, CLASSES, source_id


def test_inspection_and_split(tmp_path):
    root=tmp_path/'data'; root.mkdir()
    for label in CLASSES:
        for i in range(3):
            folder=root/label/f'{label}{i:03}_x264.mp4'; folder.mkdir(parents=True)
            for j in range(2):
                writer=cv2.VideoWriter(str(folder/f'{j}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),5,(16,16))
                writer.write(np.zeros((16,16,3),np.uint8)); writer.release()
    (root/'labels.csv').write_text('clip,label\n')
    report=inspect(root,tmp_path/'report')
    assert report['readable']==24 and len(report['directories_named_mp4'])==12
    assert report['annotation_candidates']==['labels.csv']
    rows=split_manifest(tmp_path/'report/manifest.csv',tmp_path/'split.csv')
    sets={s:{r['source'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert all(len(v)==4 for v in sets.values())
    assert not sets['train'] & (sets['val'] | sets['test'])
    assert not sets['val'] & sets['test']
    assert source_id(root/'Burglary/Burglary001_x264_1.mp4',root)==source_id(root/'Burglary/Burglary001_x264_2.mp4',root)


def test_unreadable_and_missing_classes(tmp_path):
    import pytest
    (tmp_path/'bad.mp4').write_bytes(b'not a video')
    report=inspect(tmp_path,tmp_path/'out')
    assert report['unreadable']==1
    with pytest.raises(ValueError,match='at least 3'):
        split_manifest(tmp_path/'out/manifest.csv',tmp_path/'split.csv')


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (16, 16))
    assert writer.isOpened()
    # Unique per logical source/clip; mirror copies are still copied byte-for-byte.
    import hashlib
    key = f'{path.parent.name}/{path.name}'.encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')
    writer.write(np.random.default_rng(seed).integers(0, 256, (16, 16, 3), dtype=np.uint8))
    writer.release()


def test_duplicate_dataset_roots(tmp_path):
    import shutil
    from scripts.data import canonical_clip, validate_manifest
    root = tmp_path/'dataset'
    direct = root/'DCSASS Dataset'
    for label in CLASSES + ['Arson']:
        for i in range(3 if label in CLASSES else 1):
            for j in range(2):
                video(direct/label/f'{label}{i:03}_x264.mp4'/f'{j}.avi')
    mirror = root/'dcsass dataset'/'DCSASS Dataset'
    shutil.copytree(direct, mirror)
    report = inspect(root, tmp_path/'report')
    assert report['raw_video_files'] == 52
    assert report['unique_video_clips'] == 26
    assert report['excluded_duplicate_copies'] == 26
    assert report['unique_original_sources'] == 13
    assert report['unmapped_clips'] == 2 and report['unmapped_category_counts'] == {'arson': 2}
    rows = read_rows(tmp_path/'report/manifest.csv')
    excluded = read_rows(tmp_path/'report/excluded_duplicates.csv')
    assert len({r['canonical_clip'] for r in rows}) == len(rows)
    assert all(Path(r['path']).is_relative_to(direct) for r in rows)
    assert all(len(r['sha256']) == 64 for r in excluded)
    a = direct/'Fighting/Fighting000_x264.mp4/0.avi'
    b = mirror/'Fighting/Fighting000_x264.mp4/0.avi'
    assert canonical_clip(a, root) == canonical_clip(b, root)
    assert source_id(a, root) == source_id(b, root) == 'fighting000_x264'
    first = (tmp_path/'report/manifest.csv').read_bytes()
    inspect(root, tmp_path/'report')
    assert (tmp_path/'report/manifest.csv').read_bytes() == first
    assert read_rows(tmp_path/'report/excluded_duplicates.csv') == excluded
    split = split_manifest(tmp_path/'report/manifest.csv', tmp_path/'split.csv', per_class=3)
    validate_manifest(split, require_splits=True, require_classes=True)
    # Input enumeration order does not alter the seeded split or clip selection.
    from scripts.data import write_csv, FIELDS
    write_csv(tmp_path/'reversed.csv', reversed(rows), FIELDS)
    assert split_manifest(tmp_path/'reversed.csv', tmp_path/'split2.csv', per_class=3) == split


def test_conflicting_duplicate_bytes(tmp_path):
    import pytest
    a = tmp_path/'data/DCSASS Dataset/Fighting/Fighting001_x264.mp4/0.avi'
    b = tmp_path/'data/dcsass dataset/DCSASS Dataset/Fighting/Fighting001_x264.mp4/0.avi'
    video(a)
    b.parent.mkdir(parents=True)
    payload = bytearray(a.read_bytes()); payload[-1] ^= 1
    b.write_bytes(payload)
    with pytest.raises(ValueError, match='different video contents'):
        inspect(tmp_path/'data', tmp_path/'out')
    assert (tmp_path/'out/duplicate_conflict.json').is_file()


def test_manifest_guards(tmp_path):
    import pytest
    from scripts.data import validate_manifest
    test_inspection_and_split(tmp_path)
    rows = read_rows(tmp_path/'split.csv')
    with pytest.raises(ValueError, match='Duplicate canonical clip'):
        validate_manifest(rows + [dict(rows[0])], require_splits=True)
    altered = [dict(r) for r in rows]
    # Another distinct clip from the same original video, assigned across splits.
    extra = dict(altered[0])
    original = Path(extra['path']); alternate = original.with_name('1.avi' if original.name == '0.avi' else '0.avi')
    extra['path'] = str(alternate)
    extra['canonical_clip'] = extra['canonical_clip'].rsplit('/', 1)[0] + '/' + alternate.name
    extra['split'] = 'test' if extra['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='Source leakage'):
        validate_manifest(altered + [extra], require_splits=True)
    altered[0]['path'] = str(tmp_path/'missing.avi')
    with pytest.raises(FileNotFoundError, match='Missing video file'):
        validate_manifest(altered)
    with pytest.raises(ValueError, match='class counts'):
        validate_manifest(rows[1:], require_splits=True, require_classes=True)
    legacy = dict(rows[0]); del legacy['canonical_clip']
    with pytest.raises(ValueError, match='regenerate'):
        validate_manifest([legacy])


def test_fighting_eight_source_limit(tmp_path):
    import json
    root = tmp_path/'data'
    available = {'Burglary': 10, 'Stealing': 12, 'Fighting': 8, 'Vandalism': 9}
    for label, count in available.items():
        for i in range(count):
            video(root/label/f'{label}{i:03}_x264.mp4/0.avi')
    inspect(root, tmp_path/'inspection')
    rows = split_manifest(tmp_path/'inspection/manifest.csv', tmp_path/'split.csv', per_class=30)
    report = json.loads((tmp_path/'split.report.json').read_text())
    assert report['selected_sources_per_class'] == 8
    for label in CLASSES:
        assert report['available_readable'][label]['sources'] == available[label]
        assert [report['split_counts'][s][label]['sources'] for s in ['train','val','test']] == [6,1,1]
    assert all(not overlap for overlap in report['source_overlap'].values())
    assert len(rows) == 32
