"""Fresh Colab path recovery with exact Kaggle-path, annotation and hash fixtures."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.data import CLASSES,inspect,file_hash
from scripts.smoke import prepare_smoke,artifact_hashes,read_csv
from scripts.context_review import context_plan,save_context_review
from scripts.review import review_selected
from scripts.review_dashboard import consolidate_review,write_dashboard
from scripts.review_interface import save_review_decisions
from scripts.review_launcher import resume_decisions,launch_saved_review
from scripts.source_correction import SOURCE,preview_source_correction,apply_source_correction
from scripts.review_paths import ensure_review_dataset,recover_review_dataset
from scripts import review_paths
from tests.test_review import write_video
from tests.test_review_dashboard import edit_checklist
from tests.test_review_interface import answers
from tests.test_smoke import CONFIG

OLD='/kaggle/input/dcsass-dataset'


@pytest.fixture
def saved_review(tmp_path,monkeypatch):
    root=tmp_path/'original-data';output=tmp_path/'drive'/'canonical-four-class-v2';prep=output/'smoke_preparation'
    for c,label in enumerate(CLASSES):
        for source in ([19,43,1] if label=='Vandalism' else [1,2,3]):
            for index in range(3):
                path=root/'DCSASS Dataset'/label/f'{label}{source:03}_x264.mp4'/f'{label}{source:03}_x264_{index}.avi'
                rng=np.random.default_rng(c*1000+source*10+index)
                write_video(path,[rng.integers(0,256,(32,48,3),dtype=np.uint8) for _ in range(6)])
    shutil.copytree(root/'DCSASS Dataset',root/'dcsass dataset'/'DCSASS Dataset')
    inspect(root,output/'inspection');result=prepare_smoke(root,output/'inspection',prep,CONFIG)
    assert result['automatic_status']=='PASS'
    review_selected(result['split_manifest'],root,prep/'temporal_review')
    directory=save_context_review(context_plan(prep/'validation.json'),prep/'source_context_review')
    fixed=directory.with_name('review-bd132d7b024a');directory.rename(fixed);directory=fixed
    # Model an actual Kaggle-authored saved review using the exact reported root.
    for file in [output/'inspection'/'manifest.csv',output/'inspection'/'excluded_duplicates.csv',
                 output/'inspection'/'inspection.json',prep/'candidate_splits.csv',prep/'temporal_review'/'review.json',directory/'context_metadata.json']:
        file.write_text(file.read_text().replace(str(root),OLD))
    report=json.loads((prep/'validation.json').read_text().replace(str(root),OLD))
    report['artifact_hashes']=artifact_hashes(output/'inspection');report['original_artifact_hashes']=report['artifact_hashes']
    report['split_file_sha256']=file_hash(prep/'candidate_splits.csv')
    (prep/'validation.json').write_text(json.dumps(report))
    metadata=json.loads((directory/'context_metadata.json').read_text());metadata['manifest_sha256']=report['split_file_sha256']
    (directory/'context_metadata.json').write_text(json.dumps(metadata))
    ensure_review_dataset(directory,root)
    def annotate(rows):
        for row in rows:
            row['notes']='Existing human note: '+row['source']
            if row['source']==SOURCE:row.update(selected_label_visible='Confirmed',watched_all='Confirmed',incident_independence='Pending')
    edit_checklist(directory,annotate)
    payload=answers(consolidate_review(directory))
    next(row for row in payload if row['source']==SOURCE).update(label_visible='Confirmed',incident_independence='Pending')
    save_review_decisions(directory,consolidate_review(directory),payload)
    apply_source_correction(directory,preview_source_correction(directory),approved=True)
    home=tmp_path/'colab-home';cache=home/'.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1'
    cache.parent.mkdir(parents=True);shutil.move(root,cache)
    review_paths._BINDINGS.clear()  # A fresh kernel has no previous path bindings.
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:home))
    protected={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    return directory,cache,protected,output


def assert_unchanged(protected):
    assert all(path.read_bytes()==data for path,data in protected.items())


def test_fresh_colab_auto_detects_root_preserves_approved_correction(saved_review):
    directory,cache,protected,output=saved_review
    plan=ensure_review_dataset(directory)
    assert plan['dataset_root']==str(cache)
    assert all(Path(c['path']).is_relative_to(cache) for g in plan['groups'] for c in g['videos'])
    report=consolidate_review(directory);decisions,path=resume_decisions(directory,report)
    assert decisions[SOURCE]['label_visible']=='Uncertain'
    row=next(r for r in report['rows'] if r['source']==SOURCE)
    assert row['selected_label_visible']=='Uncertain' and row['incident_independence']=='Pending'
    assert report['recorded_dataset_root']==OLD and report['runtime_dataset_root']==str(cache)
    assert report['source_mapping_reviewed'] is False and report['run_prototype_training'] is False
    assert_unchanged(protected)
    # Subsequent callbacks validate the same temporary path view, not stale paths.
    assert ensure_review_dataset(directory)['groups']==plan['groups']


def test_fresh_launcher_uses_existing_review_without_preparation(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    for target in ['scripts.data.inspect','scripts.data.split_manifest','scripts.smoke.prepare_smoke','scripts.smoke.run_reviewed_smoke']:
        monkeypatch.setattr(target,lambda *a,**kw:pytest.fail('Forbidden split/inspection/training'))
    monkeypatch.setattr('scripts.review_dashboard.write_dashboard',lambda folder,**kwargs:folder/'local-dashboard-fixture')
    selected,dashboard=launch_saved_review(output,directory)
    assert selected==directory and directory.name=='review-bd132d7b024a'
    assert_unchanged(protected)


def test_missing_files_stop_with_exact_recovery_action(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    shutil.rmtree(cache)
    monkeypatch.setattr(review_paths,'discover_dataset_roots',lambda directory:[])
    with pytest.raises(RuntimeError,match='Restore exact saved-review dataset') as error:
        launch_saved_review(output,directory)
    assert OLD in str(error.value) and 'RECOVERY_VERSION' in str(error.value)
    assert 'Only the 12 exported videos are insufficient' in str(error.value)
    assert_unchanged(protected)


@pytest.mark.parametrize('damage',['changed','missing_copy','symlink','extra_clip'])
def test_invalid_dataset_never_accepted(saved_review,damage,tmp_path):
    directory,cache,protected,output=saved_review
    path=next(cache.rglob('*.avi'))
    if damage=='changed':path.write_bytes(path.read_bytes()+b'changed')
    if damage=='missing_copy':path.unlink()
    if damage=='symlink':
        outside=tmp_path/'outside.avi';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    if damage=='extra_clip':(cache/'extra.avi').write_bytes(path.read_bytes())
    with pytest.raises(FileNotFoundError,match='no relocated root passed verification'):
        ensure_review_dataset(directory,cache)
    assert str(directory.resolve()) not in review_paths._BINDINGS
    assert_unchanged(protected)


def test_unknown_original_version_never_downloads_latest(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    calls=[];monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle)))
    with pytest.raises(ValueError,match='not recorded unambiguously'):
        recover_review_dataset(directory)
    assert calls==[] and not review_paths._BINDINGS
    assert_unchanged(protected)


def test_explicit_exact_version_download_is_verified_before_acceptance(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    calls=[]
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle) or str(cache)))
    accepted=recover_review_dataset(directory,version=1)
    assert accepted==cache and calls==['mateohervas/dcsass-dataset/versions/1']
    assert_unchanged(protected)


def test_original_validation_and_candidate_paths_can_move_with_drive(saved_review):
    directory,cache,protected,output=saved_review
    # Simulate the project having moved after saving metadata. Bindings locate
    # byte-identical artifacts only at bounded known project-relative locations.
    other=output.with_name('moved-drive-project');shutil.move(output,other)
    moved=other/'smoke_preparation/source_context_review'/directory.name
    plan=ensure_review_dataset(moved)
    assert plan['dataset_root']==str(cache)
    restored,_=resume_decisions(moved,consolidate_review(moved))
    assert restored[SOURCE]['label_visible']=='Uncertain'


def test_relocated_dashboard_and_analysis_use_verified_paths(saved_review):
    directory,cache,protected,output=saved_review
    ensure_review_dataset(directory)
    dashboard=write_dashboard(directory,analyze=True)
    report=json.loads((dashboard/'review_report.json').read_text())
    assert report['runtime_dataset_root']==str(cache)
    assert report['automatic_analysis']['canonical_split_signature']==report['canonical_split_signature']
    assert len(report['automatic_analysis']['clips'])>=12
    assert all(r['technical_status']!='FAIL' for r in report['automatic_analysis']['clips'])
    restored,_=resume_decisions(directory,report)
    assert restored[SOURCE]['label_visible']=='Uncertain'
    assert_unchanged(protected)


def test_historical_context_path_escape_is_not_ignored(saved_review):
    directory,cache,protected,output=saved_review
    path=directory/'context_metadata.json';metadata=json.loads(path.read_text())
    metadata['groups'][0]['videos'][0]['path']='/outside/unrelated.avi'
    path.write_text(json.dumps(metadata))
    with pytest.raises(FileNotFoundError,match='outside its recorded dataset root'):
        ensure_review_dataset(directory,cache)


def test_candidate_download_matches_complete_manifest_without_assuming_version(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    monkeypatch.setattr(review_paths,'discover_dataset_roots',lambda directory:[])
    calls=[]
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle) or str(cache)))
    root=recover_review_dataset(directory,allow_candidate=True)
    assert root==cache and calls==['mateohervas/dcsass-dataset']
    restored,_=resume_decisions(directory,consolidate_review(directory))
    assert restored[SOURCE]['label_visible']=='Uncertain'
    receipts=list(directory.glob('dataset-recovery-*/recovery_receipt.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['historical_version_established'] is False
    assert receipt['downloaded_cache_version']=='1'
    assert receipt['verification']=='COMPLETE_SAVED_VIDEO_MANIFEST_MATCH'
    assert receipt['physical_copies']==72 and receipt['unique_clips']==36
    assert_unchanged(protected)


def test_candidate_mismatch_is_rejected_and_never_resets_human_review(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    monkeypatch.setattr(review_paths,'discover_dataset_roots',lambda directory:[])
    next(cache.rglob('*.avi')).write_bytes(b'wrong release bytes')
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:str(cache)))
    with pytest.raises(RuntimeError,match='REJECTED by complete saved-manifest'):
        recover_review_dataset(directory,allow_candidate=True)
    assert not list(directory.glob('dataset-recovery-*'))
    assert str(directory.resolve()) not in review_paths._BINDINGS
    assert_unchanged(protected)


def test_version_in_nested_bound_download_record_is_recovered(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    path=directory.parent.parent/'validation.json';report=json.loads(path.read_text())
    report['path_resolution']={'selected_download_root':'/root/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/7'}
    path.write_text(json.dumps(report))
    calls=[]
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle) or str(cache)))
    recover_review_dataset(directory)
    assert calls==['mateohervas/dcsass-dataset/versions/7']
    receipt=json.loads(next(directory.glob('dataset-recovery-*/recovery_receipt.json')).read_text())
    assert receipt['historical_version_established'] is True
    assert 'validation.json.path_resolution.selected_download_root' in str(receipt['recorded_version_evidence'])


def test_authentication_failure_gives_login_instructions_without_changes(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    monkeypatch.setattr(review_paths,'discover_dataset_roots',lambda directory:[])
    def login_required(handle):raise PermissionError('fixture: authentication needed')
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=login_required))
    with pytest.raises(RuntimeError,match='kagglehub.login') as error:
        recover_review_dataset(directory,allow_candidate=True)
    assert 'no review files changed' in str(error.value)
    assert not list(directory.glob('dataset-recovery-*'))
    assert_unchanged(protected)


def test_incomplete_saved_sha_evidence_blocks_download(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    inspection=output/'inspection';metadata=json.loads((inspection/'inspection.json').read_text())
    missing=next(iter(metadata['clip_sha256']));metadata['clip_sha256'].pop(missing)
    (inspection/'inspection.json').write_text(json.dumps(metadata))
    import csv
    path=inspection/'excluded_duplicates.csv';rows=read_csv(path)
    next(row for row in rows if row['canonical_clip']==missing)['sha256']=''
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path=directory.parent.parent/'validation.json';report=json.loads(path.read_text())
    report['artifact_hashes']=artifact_hashes(inspection);report['original_artifact_hashes']=report['artifact_hashes']
    path.write_text(json.dumps(report))
    calls=[];monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle)))
    with pytest.raises(ValueError,match='Complete original SHA-256 evidence'):
        recover_review_dataset(directory,allow_candidate=True)
    assert calls==[]
    assert (directory/'human_checklist.csv').read_bytes()==protected[directory/'human_checklist.csv']


def test_default_recovery_cell_opts_in_to_verified_candidate_without_training():
    import nbformat
    notebook=nbformat.read('notebooks/02_Crime_Classification_Training.ipynb',as_version=4)
    cell=next(c for c in notebook.cells if c.id=='restore-exact-saved-review-dataset')
    assert 'allow_candidate=True' in cell.source
    assert 'RECOVERY_VERSION = None' in cell.source
    assert 'prepare_smoke(' not in cell.source and 'split_manifest(' not in cell.source
    assert 'SOURCE_MAPPING_REVIEWED = True' not in cell.source and 'RUN_PROTOTYPE_TRAINING = True' not in cell.source


def test_dataset_records_changed_during_download_stop_acceptance(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    monkeypatch.setattr(review_paths,'discover_dataset_roots',lambda directory:[])
    candidate=directory.parent.parent/'candidate_splits.csv'
    original=candidate.read_bytes()
    def concurrent_change(handle):
        candidate.write_bytes(original+b'\n')
        return str(cache)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=concurrent_change))
    with pytest.raises(RuntimeError,match='REJECTED by complete saved-manifest'):
        recover_review_dataset(directory,allow_candidate=True)
    assert not list(directory.glob('dataset-recovery-*'))
    assert (directory/'human_checklist.csv').read_bytes()==protected[directory/'human_checklist.csv']


def test_valid_local_copy_recovery_avoids_download_and_keeps_review(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    calls=[];monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:calls.append(handle)))
    assert recover_review_dataset(directory,allow_candidate=True)==cache
    assert calls==[]
    assert_unchanged(protected)


def test_recovery_rechecks_cached_duplicate_bytes_outside_selected_clips(saved_review,monkeypatch):
    directory,cache,protected,output=saved_review
    ensure_review_dataset(directory,cache)
    # Excluded physical mirror copies are not selected playback paths. Their
    # hashes must still be rechecked by recovery instead of trusting a binding.
    excluded=next((cache/'dcsass dataset').rglob('*.avi'))
    excluded.write_bytes(excluded.read_bytes()+b'changed duplicate')
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(dataset_download=lambda handle:str(cache)))
    with pytest.raises(RuntimeError,match='REJECTED by complete saved-manifest'):
        recover_review_dataset(directory,allow_candidate=True)
    assert not list(directory.glob('dataset-recovery-*'))
    assert_unchanged(protected)
