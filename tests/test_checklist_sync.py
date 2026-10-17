import csv
import json
import os
from pathlib import Path
import pytest
from scripts.context_review import context_plan,save_context_review
from scripts.review_dashboard import consolidate_review
from scripts.review_interface import save_review_decisions
from scripts.review_launcher import resume_decisions
from scripts.checklist_sync import preview_checklist_sync,apply_checklist_sync,latest_saved_decision
from tests.test_original_review import selected
from tests.test_review_interface import answers
from tests.test_review_dashboard import edit_checklist


def setup(selected,tmp_path):
    directory=save_context_review(context_plan(selected),tmp_path/'context')
    edit_checklist(directory,lambda rows: rows[0].update(notes='Original note must stay',review_notes='Earlier review',reviewer='Earlier reviewer'))
    report=consolidate_review(directory)
    payload=answers(report)
    payload[0].update(label_visible='Confirmed',incident_independence='Uncertain',review_notes='Person seen; crime uncertain')
    payload[1].update(label_visible='Pending',incident_independence='Pending')
    receipt=save_review_decisions(directory,report,payload)
    return directory,Path(receipt['saved_csv']),payload


def test_preview_read_only_and_explicit_approval_required(selected,tmp_path):
    directory,saved,payload=setup(selected,tmp_path)
    before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    preview=preview_checklist_sync(directory)
    assert any(c['before']=='PENDING' and c['after']=='Confirmed' and c['field']=='selected_label_visible' for c in preview['changes'])
    for approval in [False,'True',1,None]:
        with pytest.raises(ValueError,match='Explicit approval'):
            apply_checklist_sync(directory,preview,approved=approval)
    assert before=={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    assert not list(directory.glob('sync-*'))


def test_approved_sync_backup_notes_exact_identity_and_resumption(selected,tmp_path):
    directory,saved,payload=setup(selected,tmp_path)
    original=(directory/'human_checklist.csv').read_bytes();saved_bytes=saved.read_bytes()
    candidate=Path(json.loads(selected.read_text())['split_manifest']);candidate_bytes=candidate.read_bytes()
    preview=preview_checklist_sync(directory)
    result=apply_checklist_sync(directory,preview,approved=True)
    assert Path(result['backup_path']).read_bytes()==original
    assert 'human_checklist.backup.' in Path(result['backup_path']).name
    with (directory/'human_checklist.csv').open() as f:rows=list(csv.DictReader(f))
    row=next(r for r in rows if r['source']==payload[0]['source'])
    assert row['notes']=='Original note must stay'
    assert 'Earlier review' in row['review_notes'] and 'Person seen; crime uncertain' in row['review_notes']
    assert row['reviewer']=='Earlier reviewer; Human reviewer'
    assert row['selected_label_visible']=='Confirmed' and row['incident_independence']=='Uncertain'
    assert next(r for r in rows if r['source']==payload[1]['source'])['selected_label_visible']=='Pending'
    assert saved.read_bytes()==saved_bytes and candidate.read_bytes()==candidate_bytes
    recalculated=consolidate_review(directory)
    assert recalculated['human_approval']=='PENDING'
    assert recalculated['checklist_status']==result['status']
    assert any(r['source']==payload[0]['source'] and any('Uncertain' in reason for reason in r['reasons']) for r in result['status']['remaining_pending_or_uncertain'])
    restored,path=resume_decisions(directory,recalculated)
    assert path==str(saved) and restored[payload[0]['source']]['incident_independence']=='Uncertain'
    assert not recalculated['source_mapping_reviewed'] and not recalculated['run_prototype_training']
    again=preview_checklist_sync(directory)
    assert again['changes']==[]
    assert apply_checklist_sync(directory,again,approved=True)['message'].startswith('No changes')
    assert len(list(directory.glob('sync-*')))==1


@pytest.mark.parametrize('change',['checklist','new_decision','tampered_preview'])
def test_stale_preview_requires_new_approval(selected,tmp_path,change):
    directory,saved,payload=setup(selected,tmp_path)
    preview=preview_checklist_sync(directory)
    if change=='checklist':
        edit_checklist(directory,lambda rows:rows[0].update(notes='Edited after preview'))
    elif change=='new_decision':
        payload[0]['label_visible']='Rejected'
        save_review_decisions(directory,consolidate_review(directory),payload)
    else:
        preview['proposed_rows'][0]['selected_label_visible']='Confirmed FAKE'
    current=(directory/'human_checklist.csv').read_bytes()
    with pytest.raises(ValueError):apply_checklist_sync(directory,preview,approved=True)
    assert (directory/'human_checklist.csv').read_bytes()==current
    assert not list(directory.glob('sync-*'))


@pytest.mark.parametrize('kind',['case_changed','duplicate','missing','split_changed','unknown_option'])
def test_bad_decision_sources_and_identity_rejected(selected,tmp_path,kind):
    directory,saved,payload=setup(selected,tmp_path)
    with saved.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if kind=='case_changed':rows[0]['source']=rows[0]['source'].upper()
    if kind=='duplicate':rows[1]['source']=rows[0]['source']
    if kind=='missing':rows.pop()
    if kind=='split_changed':rows[0]['split']='other'
    if kind=='unknown_option':rows[0]['incident_independence']='auto approved'
    with saved.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):preview_checklist_sync(directory)


def test_latest_by_saved_utc_not_drive_mtime(selected,tmp_path):
    directory,older,payload=setup(selected,tmp_path)
    newer=Path(save_review_decisions(directory,consolidate_review(directory),payload)['saved_csv'])
    os.utime(older.parent/'provenance.json',(9999999999,9999999999))
    assert latest_saved_decision(directory)[0]==newer
    assert preview_checklist_sync(directory)['decision_file']==str(newer)


def test_report_failure_restores_checklist_and_keeps_backup(selected,tmp_path,monkeypatch):
    directory,saved,payload=setup(selected,tmp_path)
    preview=preview_checklist_sync(directory);original=(directory/'human_checklist.csv').read_bytes()
    from scripts import review_dashboard
    original_consolidate=review_dashboard.consolidate_review
    def fail_after_write(folder):
        if (directory/'human_checklist.csv').read_bytes()!=original:
            raise ValueError('fixture post-write validation failure')
        return original_consolidate(folder)
    monkeypatch.setattr(review_dashboard,'consolidate_review',fail_after_write)
    with pytest.raises(ValueError,match='post-write'):
        apply_checklist_sync(directory,preview,approved=True)
    assert (directory/'human_checklist.csv').read_bytes()==original
    backups=list(directory.glob('sync-*/human_checklist.backup.*.csv'))
    assert len(backups)==1 and backups[0].read_bytes()==original
    assert not (directory/'.checklist_sync.lock').exists()


def test_merge_later_checklist_notes_only_via_explicit_preview(selected,tmp_path):
    directory,saved,payload=setup(selected,tmp_path)
    edit_checklist(directory,lambda rows:rows[0].update(notes='New note after decisions were saved'))
    with pytest.raises(ValueError,match='changed review inputs'):
        resume_decisions(directory,consolidate_review(directory))
    preview=preview_checklist_sync(directory)
    assert preview['proposed_rows'][0]['notes']=='New note after decisions were saved'
    apply_checklist_sync(directory,preview,approved=True)
    current=consolidate_review(directory)
    restored,path=resume_decisions(directory,current)
    assert path==str(saved)
    assert current['rows'][0]['review_notes'].startswith('New note after decisions were saved')


def test_colab_preview_token_and_separate_approval(selected,tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    from scripts.checklist_sync import render_sync_controls
    directory,saved,payload=setup(selected,tmp_path)
    callbacks={}
    google=ModuleType('google');colab=ModuleType('google.colab')
    colab.output=type('Output',(),{'register_callback':staticmethod(lambda name,cb:callbacks.update({name:cb}))})
    google.colab=colab
    ipython=ModuleType('IPython');display=ModuleType('IPython.display');display.JSON=lambda result:result
    ipython.display=display
    for name,module in [('google',google),('google.colab',colab),('IPython',ipython),('IPython.display',display)]:
        monkeypatch.setitem(sys.modules,name,module)
    page=render_sync_controls(directory)
    assert 'data-approve disabled' in page and 'data-apply disabled' in page
    callback=next(iter(callbacks.values()));original=(directory/'human_checklist.csv').read_bytes()
    proposed=callback('preview')
    assert (directory/'human_checklist.csv').read_bytes()==original
    with pytest.raises(ValueError,match='Explicit approval'):
        callback('apply',proposed['token'],False)
    with pytest.raises(ValueError,match='Preview first'):
        callback('apply','unknown',True)
    applied=callback('apply',proposed['token'],True)
    assert Path(applied['backup_path']).read_bytes()==original
    with pytest.raises(ValueError,match='Preview first'):
        callback('apply',proposed['token'],True)


@pytest.mark.parametrize('name',['Alice; Bob','=Reviewer; team'])
def test_idempotent_reviewer_history_with_delimiters(selected,tmp_path,name):
    directory,saved,payload=setup(selected,tmp_path)
    for row in payload:row['reviewer']=name
    save_review_decisions(directory,consolidate_review(directory),payload)
    apply_checklist_sync(directory,preview_checklist_sync(directory),approved=True)
    assert preview_checklist_sync(directory)['changes']==[]
