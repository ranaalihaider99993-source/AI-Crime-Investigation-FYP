import json
import os
from pathlib import Path
import pytest
import nbformat
from scripts.context_review import context_plan,save_context_review
from scripts.review_dashboard import consolidate_review
from scripts.review_interface import save_review_decisions,render_review_interface
from scripts.review_launcher import choose_context,resume_decisions,launch_saved_review
from tests.test_review_interface import answers
from tests.test_original_review import selected


def existing(selected,tmp_path):
    parent=tmp_path/'smoke_preparation'/'source_context_review'
    plan=context_plan(selected)
    directory=save_context_review(plan,parent)
    return plan,directory,consolidate_review(directory)


def test_resume_only_explicit_answers_and_original_notes(selected,tmp_path):
    plan,directory,report=existing(selected,tmp_path)
    assert resume_decisions(directory,report)==({},None)
    payload=answers(report)
    payload[0]['label_visible']='Confirmed'
    receipt=save_review_decisions(directory,report,payload)
    restored,path=resume_decisions(directory,report)
    assert path==receipt['saved_csv']
    assert Path(path).name.startswith('review_decisions_')
    assert restored[payload[0]['source']]['label_visible']=='Confirmed'
    assert restored[payload[0]['source']]['review_notes']=='=note; uncertain label'
    page=render_review_interface(report,plan,'test.callback',restored,path)
    assert 'Restored explicit human answers from:' in page
    assert '<option selected>Confirmed</option>' in page
    assert 'Selected-clip label decisions' in page and 'data-summary' in page
    assert 'summary();' in page
    assert not json.loads((Path(path).parent/'provenance.json').read_text())['source_mapping_reviewed']


def test_activity_selection_preserves_annotated_review(selected,tmp_path):
    plan,directory,report=existing(selected,tmp_path)
    save_review_decisions(directory,report,answers(report))
    newer=save_context_review(plan,directory.parent)
    os.utime(newer/'context_metadata.json',(9999999999,9999999999))
    assert choose_context(tmp_path)==directory
    assert choose_context(tmp_path,explicit=newer)==newer


def test_annotation_selection_without_decisions(selected,tmp_path):
    plan,directory,report=existing(selected,tmp_path)
    from tests.test_review_dashboard import edit_checklist
    edit_checklist(directory,lambda rows:rows[0].update(notes='Keep these human notes'))
    newer=save_context_review(plan,directory.parent)
    os.utime(newer/'context_metadata.json',(9999999999,9999999999))
    assert choose_context(tmp_path)==directory


def test_stale_resume_stops_instead_of_discarding_answers(selected,tmp_path):
    plan,directory,report=existing(selected,tmp_path)
    receipt=save_review_decisions(directory,report,answers(report))
    report['inputs']=dict(report['inputs'],checklist_sha256='changed')
    with pytest.raises(ValueError,match='No decisions were silently discarded'):
        resume_decisions(directory,report)
    assert Path(receipt['saved_csv']).is_file()


def test_missing_dataset_recovery_never_prepares_or_splits(selected,tmp_path,monkeypatch):
    plan,directory,report=existing(selected,tmp_path)
    for module,function in [('scripts.data','inspect'),('scripts.data','split_manifest'),('scripts.smoke','prepare_smoke'),('scripts.smoke','run_reviewed_smoke')]:
        monkeypatch.setattr(module+'.'+function,lambda *a,**kw:pytest.fail('Forbidden preparation/training operation'))
    Path(plan['groups'][0]['videos'][0]['path']).unlink()
    with pytest.raises(RuntimeError,match='Recovery:.*') as error:
        launch_saved_review(tmp_path,explicit=directory)
    assert 'SAME recorded DCSASS version' in str(error.value)
    assert 'Do not Run all' in str(error.value)
    assert plan['dataset_root'] in str(error.value)
    assert not list(directory.glob('decisions-*'))


def test_launcher_uses_existing_snapshot_only(selected,tmp_path,monkeypatch):
    plan,directory,report=existing(selected,tmp_path)
    calls=[]
    from scripts import review_dashboard
    monkeypatch.setattr(review_dashboard,'write_dashboard',lambda folder,**kw: calls.append((folder,kw)) or folder/'dashboard-test')
    chosen,dashboard=launch_saved_review(tmp_path)
    assert chosen==directory
    assert calls==[(directory,dict(interactive=True,resume=True,analyze=True))]


def test_one_cell_launcher_is_first_code_and_disables_training():
    book=nbformat.read('notebooks/02_Crime_Classification_Training.ipynb',as_version=4)
    code=next(c for c in book.cells if c.cell_type=='code')
    assert code.id=='launch-saved-review-dashboard'
    assert 'drive.mount' in code.source and "'fetch', 'origin', 'main'" in code.source
    assert 'SOURCE_MAPPING_REVIEWED = False' in code.source and 'RUN_PROTOTYPE_TRAINING = False' in code.source
    assert 'launch_saved_review(OUTPUT, REVIEW_CONTEXT_FOLDER, dataset_root=REVIEW_DATASET_ROOT)' in code.source
    assert 'prepare_smoke(' not in code.source and 'dataset_download(' not in code.source


def test_resume_prior_fixed_filename_csv(selected,tmp_path):
    plan,directory,report=existing(selected,tmp_path)
    saved=Path(save_review_decisions(directory,report,answers(report))['saved_csv'])
    old=saved.with_name('review_decisions.csv')
    saved.rename(old)
    restored,path=resume_decisions(directory,report)
    assert path==str(old) and len(restored)==12
