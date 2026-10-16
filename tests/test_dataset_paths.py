"""Reproduce the reported KaggleHub root mismatch, relocation and escape attempts."""
import json
import shutil
from pathlib import Path
import pytest
from scripts.data import CLASSES, inspect, write_csv, FIELDS, split_manifest, split_signature
from scripts.dataset_paths import resolve_dataset_root, resolve_inspection_paths
from scripts.smoke import prepare_smoke, read_csv
from tests.test_data import video
from tests.test_smoke import CONFIG


def cache_root(tmp_path, version='1'):
    return tmp_path/'root/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions'/version


def build(root, report, legacy=False):
    direct = root/'DCSASS Dataset'
    for label in CLASSES + ['Abuse']:
        for i in range(1, 2 if label == 'Abuse' else 4):
            stem = f'{label}{i:03}_x264'
            path = direct/label/f'{stem}.mp4'/f'{stem}_0.avi'
            video(path)
            # Real video bytes with the exact reported .mp4 directory/clip names.
            path.rename(path.with_suffix('.mp4'))
    shutil.copytree(direct, root/'dcsass dataset/DCSASS Dataset')
    inspect(root, report)
    if legacy:
        p = report/'inspection.json'; meta = json.loads(p.read_text())
        meta.pop('dataset_root'); meta.pop('clip_sha256')
        p.write_text(json.dumps(meta))
    return root/'dcsass dataset/DCSASS Dataset'


def test_exact_reported_abuse_path_with_nested_selected_root(tmp_path):
    root = cache_root(tmp_path)
    nested = build(root, tmp_path/'inspection', legacy=True)
    reported = root/'DCSASS Dataset/Abuse/Abuse001_x264.mp4/Abuse001_x264_0.mp4'
    assert reported.is_file() and not reported.is_relative_to(nested)
    assert resolve_dataset_root(nested) == root
    assert resolve_dataset_root(root/'DCSASS Dataset') == root
    original = (tmp_path/'inspection/manifest.csv').read_bytes()
    resolved, directory, result = resolve_inspection_paths(nested, tmp_path/'inspection', tmp_path/'rebound')
    assert resolved == root and directory == tmp_path/'inspection'
    assert result['relocated_copies'] == 0
    assert (tmp_path/'inspection/manifest.csv').read_bytes() == original
    report = prepare_smoke(nested, tmp_path/'inspection', tmp_path/'prep', CONFIG)
    assert report['automatic_status'] == 'PASS'
    assert report['path_resolution']['verified_dataset_root'] == str(root)
    assert report['human_mapping_review'] == 'PENDING' and report['training_executed'] is False


def test_new_cache_version_rebases_legacy_records_without_old_reads(tmp_path, monkeypatch):
    old, new = cache_root(tmp_path, '1'), cache_root(tmp_path, '2')
    build(old, tmp_path/'inspection', legacy=True)
    originals = {p.name:p.read_bytes() for p in (tmp_path/'inspection').iterdir()}
    before = split_manifest(tmp_path/'inspection/manifest.csv', tmp_path/'before.csv', per_class=3)
    signature = split_signature(tmp_path/'before.csv')
    shutil.copytree(old, new)
    # Keep old data on disk to ensure existing old paths do not bypass rebasing.
    import scripts.dataset_paths as paths
    real_hash = paths.file_hash
    def new_only_hash(path):
        assert Path(path).is_relative_to(new), 'Resolver read old/out-of-bound video bytes'
        return real_hash(path)
    monkeypatch.setattr(paths, 'file_hash', new_only_hash)
    root, directory, result = resolve_inspection_paths(new, tmp_path/'inspection', tmp_path/'rebound')
    assert root == new and directory == tmp_path/'rebound'
    assert result['relocated_copies'] == result['reverified_sha256_files'] == 26
    rebound = read_csv(directory/'manifest.csv')
    assert all(Path(r['path']).is_relative_to(new) for r in rebound)
    assert [r['canonical_clip'] for r in rebound] == [r['canonical_clip'] for r in read_csv(tmp_path/'inspection/manifest.csv')]
    assert [r['source'] for r in rebound] == [r['source'] for r in read_csv(tmp_path/'inspection/manifest.csv')]
    assert {p.name:p.read_bytes() for p in (tmp_path/'inspection').iterdir()} == originals
    split_manifest(directory/'manifest.csv', tmp_path/'after.csv', per_class=3)
    assert split_signature(tmp_path/'after.csv') == signature
    assert len(read_csv(directory/'path_relocations.csv')) == 26


def test_recorded_custom_root_relocation_after_old_cache_deleted(tmp_path):
    old, new = tmp_path/'old_download', tmp_path/'new_download'
    build(old, tmp_path/'inspection')
    shutil.copytree(old, new); shutil.rmtree(old)
    report = prepare_smoke(new, tmp_path/'inspection', tmp_path/'prep', CONFIG)
    assert report['automatic_status'] == 'PASS'
    assert report['path_resolution']['relocated_copies'] == 26
    assert Path(report['inspection_dir']) == tmp_path/'prep/resolved_inspection'
    assert all(Path(r['path']).is_relative_to(new) for r in read_csv(report['split_manifest']))


@pytest.mark.parametrize('layout', ['datasets/mateohervas/dcsass-dataset', 'dcsass-dataset'])
def test_kaggle_mounted_input_root(tmp_path, layout):
    root = tmp_path/'kaggle/input'/layout
    nested = build(root, tmp_path/'inspection')
    assert resolve_dataset_root(nested) == root


def test_root_ascent_is_never_inferred_from_wrapper_name_alone(tmp_path):
    root = tmp_path/'unverified_parent/DCSASS Dataset'
    root.mkdir(parents=True)
    assert resolve_dataset_root(root) == root
    version = cache_root(tmp_path)
    category = version/'DCSASS Dataset/Abuse'; category.mkdir(parents=True)
    with pytest.raises(ValueError, match='category/source subdirectory'):
        resolve_dataset_root(category)
    with pytest.raises(ValueError, match='traversal'):
        resolve_dataset_root(category/'..')


@pytest.mark.parametrize('bad', ['missing', 'changed_bytes', 'arbitrary_old_path', 'wrong_relative_identity', 'missing_hash'])
def test_relocation_fails_closed(tmp_path, bad):
    old, new = cache_root(tmp_path, '1'), cache_root(tmp_path, '2')
    build(old, tmp_path/'inspection', legacy=True)
    shutil.copytree(old, new)
    rows = read_csv(tmp_path/'inspection/manifest.csv')
    relative = Path(rows[0]['path']).relative_to(old)
    if bad == 'missing':
        (new/relative).unlink()
    elif bad == 'changed_bytes':
        path = new/relative; payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
    elif bad == 'arbitrary_old_path':
        rows[0]['path'] = str(tmp_path/'unrelated'/relative)
        write_csv(tmp_path/'inspection/manifest.csv', rows, FIELDS)
    elif bad == 'wrong_relative_identity':
        rows[0]['path'] = rows[1]['path']
        write_csv(tmp_path/'inspection/manifest.csv', rows, FIELDS)
    else:
        # Single-copy historic artifacts without recorded content evidence cannot move.
        audit = read_csv(tmp_path/'inspection/excluded_duplicates.csv')
        selected = rows[0]['canonical_clip']
        record = next(r for r in audit if r['canonical_clip'] == selected)
        (new/Path(record['excluded_path']).relative_to(old)).unlink()
        audit = [r for r in audit if r['canonical_clip'] != selected]
        write_csv(tmp_path/'inspection/excluded_duplicates.csv', audit, list(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_inspection_paths(new, tmp_path/'inspection', tmp_path/'rebound')
    assert not (tmp_path/'rebound/manifest.csv').exists()
    # The notebook records failure without enabling review/training.
    report = prepare_smoke(new, tmp_path/'inspection', tmp_path/'prep', CONFIG)
    assert report['automatic_status'] == 'FAIL'
    assert report['training_executed'] is False


def test_symlink_file_and_wrapper_escapes_rejected(tmp_path):
    root = cache_root(tmp_path)
    build(root, tmp_path/'inspection')
    rows = read_csv(tmp_path/'inspection/manifest.csv')
    path = Path(rows[0]['path'])
    outside = tmp_path/'outside.mp4'; shutil.copyfile(path, outside)
    path.unlink(); path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink escapes'):
        resolve_inspection_paths(root, tmp_path/'inspection', tmp_path/'rebound')
    with pytest.raises(ValueError, match='symlink escapes'):
        inspect(root, tmp_path/'new-inspection')
    other = cache_root(tmp_path, '2'); other.mkdir()
    wrapper = other/'DCSASS Dataset'; wrapper.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink escapes'):
        resolve_dataset_root(wrapper)
