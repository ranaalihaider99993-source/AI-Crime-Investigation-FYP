import json
from pathlib import Path
import pytest
import torch
from scripts.data import inspect, write_csv, FIELDS
from scripts.smoke import prepare_smoke, run_reviewed_smoke, read_csv
from tests.test_data import test_duplicate_dataset_roots as make_mirrors, video

CONFIG = dict(seed=42, sources_per_class=3, epochs=2, batch_size=4,
              image_size=16, frames_per_clip=1, learning_rate=.001, patience=3)


def fixture(tmp_path):
    make_mirrors(tmp_path)
    report = prepare_smoke(tmp_path/'dataset', tmp_path/'report', tmp_path/'prep', CONFIG)
    return report


def test_automatic_report_is_not_review_or_training(tmp_path):
    report = fixture(tmp_path)
    assert report['automatic_status'] == 'PASS'
    assert report['human_mapping_review'] == 'PENDING'
    assert report['training_executed'] is False
    assert len(report['checks']) == 7
    assert all(not v for v in report['source_overlap'].values())
    assert report['available_readable']['Fighting']['sources'] == 3
    assert all(report['split_counts'][s]['Fighting'] == {'clips':1,'sources':1} for s in ['train','val','test'])
    assert (tmp_path/'prep/review_sheet.jpg').is_file()
    assert len(read_csv(tmp_path/'prep/source_review.csv')) == 12
    assert run_reviewed_smoke(tmp_path/'prep/validation.json', tmp_path/'runs', CONFIG) is None
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('failure', ['duplicate_clip', 'missing_file', 'audit_count', 'physical_binding'])
def test_failed_automatic_checks_block_training(tmp_path, failure):
    fixture(tmp_path)
    manifest = tmp_path/'report/manifest.csv'
    rows = read_csv(manifest)
    if failure == 'duplicate_clip':
        write_csv(manifest, rows+[dict(rows[0])], FIELDS)
    elif failure == 'missing_file':
        Path(rows[0]['path']).unlink()
    elif failure == 'physical_binding':
        # Still a real file, but from the wrong canonical clip.
        rows[0]['path'] = rows[1]['path']
        write_csv(manifest, rows, FIELDS)
    else:
        duplicates = tmp_path/'report/excluded_duplicates.csv'
        audit = read_csv(duplicates)
        write_csv(duplicates, audit[:-1], list(audit[0]))
    report = prepare_smoke(tmp_path/'dataset', tmp_path/'report', tmp_path/'prep', CONFIG)
    assert report['automatic_status'] == 'FAIL'
    assert any(c['status'] == 'FAIL' for c in report['checks'])
    assert (tmp_path/'prep/validation.md').is_file()
    with pytest.raises(ValueError, match='validation failed'):
        run_reviewed_smoke(tmp_path/'prep/validation.json', tmp_path/'runs', CONFIG, True, True, require_gpu=False)
    assert not (tmp_path/'runs').exists()


def test_insufficient_fighting_sources_reported(tmp_path):
    for label in ['Burglary','Stealing','Fighting','Vandalism']:
        for i in range(2 if label == 'Fighting' else 3):
            video(tmp_path/'dataset'/label/f'{label}{i:03}_x264.mp4/0.avi')
    inspect(tmp_path/'dataset', tmp_path/'inspection')
    report = prepare_smoke(tmp_path/'dataset', tmp_path/'inspection', tmp_path/'prep', CONFIG)
    assert report['automatic_status'] == 'FAIL'
    assert report['available_readable']['Fighting']['sources'] == 2
    assert any('Need 3' in c['detail'] for c in report['checks'])


def test_review_gpu_and_changed_artifact_gates(tmp_path, monkeypatch):
    fixture(tmp_path)
    args = (tmp_path/'prep/validation.json', tmp_path/'runs', CONFIG)
    with pytest.raises(ValueError, match='Human source-mapping review'):
        run_reviewed_smoke(*args, run_prototype_training=True, require_gpu=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='No CUDA GPU'):
        run_reviewed_smoke(*args, True, True)
    manifest = tmp_path/'report/manifest.csv'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError, match='artifacts changed'):
        run_reviewed_smoke(*args, True, True, require_gpu=False)
    assert not (tmp_path/'runs').exists()


def test_reviewed_cpu_fixture_run_and_unique_completion(tmp_path, monkeypatch):
    # Explicit local test override; the Colab notebook ALWAYS requires GPU.
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    fixture(tmp_path)
    args = (tmp_path/'prep/validation.json', tmp_path/'runs', CONFIG)
    first = run_reviewed_smoke(*args, True, True, require_gpu=False)
    assert first['actual_training_device'] == 'cpu'
    assert first['epochs_executed'] == 2
    assert Path(first['checkpoint']).is_file()
    assert len(first['metrics']['confusion_matrix']) == 4
    assert sum(sum(r) for r in first['metrics']['confusion_matrix']) == 4
    assert len(first['metrics']['per_class']) >= 4
    second = run_reviewed_smoke(*args, True, True, require_gpu=False)
    assert second['run_dir'] != first['run_dir']
    assert Path(first['checkpoint']).is_file()
    assert json.loads((tmp_path/'runs/latest_completed.json').read_text())['run_dir'] == second['run_dir']


def test_expanded_training_config_rejected(tmp_path):
    config = dict(CONFIG, epochs=10)
    report = prepare_smoke(tmp_path/'dataset', tmp_path/'inspection', tmp_path/'prep', config)
    assert report['automatic_status'] == 'FAIL'
    assert 'full training is not authorized' in report['checks'][0]['detail']


def test_identical_selected_bytes_across_sources_block_preflight(tmp_path):
    report = fixture(tmp_path)
    rows = read_csv(report['split_manifest'])
    first = rows[0]
    second = next(r for r in rows if r['split'] != first['split'])
    payload = Path(first['path']).read_bytes()
    Path(second['path']).write_bytes(payload)
    for record in read_csv(tmp_path/'report/excluded_duplicates.csv'):
        if record['canonical_clip'] == second['canonical_clip']:
            Path(record['excluded_path']).write_bytes(payload)
    failed = prepare_smoke(tmp_path/'dataset', tmp_path/'report', tmp_path/'prep', CONFIG)
    assert failed['automatic_status'] == 'FAIL'
    assert any('Identical video bytes' in c['detail'] for c in failed['checks'])


def test_changed_selected_contents_and_partial_runs_do_not_complete(tmp_path, monkeypatch):
    report = fixture(tmp_path)
    args = (tmp_path/'prep/validation.json', tmp_path/'runs', CONFIG)
    # A successfully decoded file can still change after review.
    selected = read_csv(report['split_manifest'])
    payload = Path(selected[0]['path']).read_bytes()
    Path(selected[0]['path']).write_bytes(Path(selected[1]['path']).read_bytes())
    with pytest.raises(ValueError, match='video contents changed'):
        run_reviewed_smoke(*args, True, True, require_gpu=False)
    Path(selected[0]['path']).write_bytes(payload)
    def fail_training(*args, **kwargs):
        raise RuntimeError('simulated runtime disconnection')
    monkeypatch.setattr('scripts.smoke.train', fail_training)
    with pytest.raises(RuntimeError, match='disconnection'):
        run_reviewed_smoke(*args, True, True, require_gpu=False)
    assert not (tmp_path/'runs/latest_completed.json').exists()
