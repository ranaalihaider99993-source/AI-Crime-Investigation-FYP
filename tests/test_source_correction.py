import csv
import io
import json
from pathlib import Path
import pytest
from scripts.context_review import context_plan,save_context_review
from scripts.review_dashboard import consolidate_review
from scripts.review_interface import save_review_decisions
from scripts.review_launcher import resume_decisions
from scripts.checklist_sync import preview_checklist_sync
from scripts.source_correction import SOURCE,NOTE,patch_record,preview_source_correction,apply_source_correction
from tests.test_original_review import selected
from tests.test_review_interface import answers
from tests.test_review_dashboard import edit_checklist


def setup(selected,tmp_path,watched='Confirmed'):
    directory=save_context_review(context_plan(selected),tmp_path/'context')
    def annotate(rows):
        for r in rows:
            r['notes']='Existing notes: '+r['source']
            if r['source']==SOURCE:r.update(selected_label_visible='Confirmed',watched_all=watched,incident_independence='Pending',review_notes='Earlier observation')
    edit_checklist(directory,annotate)
    report=consolidate_review(directory);payload=answers(report)
    for row in payload:
        if row['source']==SOURCE:row.update(label_visible='Confirmed',videos_watched=watched,incident_independence='Pending',review_notes='Original saved observation')
    saved=Path(save_review_decisions(directory,report,payload)['saved_csv'])
    return directory,saved


def read_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def test_patch_only_exact_csv_record_preserves_other_raw_bytes():
    prefix=b'source,label,notes\r\nother,Confirmed,"quoted, note\r\nwith newline"\r\n'
    target=(SOURCE+',Confirmed,"original note"\r\n').encode()
    suffix=b'vandalism019_X264,Pending,"Case-sensitive other row"\r\nlast,Uncertain,"keep quoting"'
    patched,before,after=patch_record(prefix+target+suffix,{'label':'Uncertain','notes':NOTE})
    assert patched.startswith(prefix) and patched.endswith(suffix)
    assert before['label']=='Confirmed' and after['label']=='Uncertain'
    assert patch_record(patched,{})[0]==patched


@pytest.mark.parametrize('watched',['Confirmed','Pending','Uncertain'])
def test_approved_single_source_patch_and_new_decision_preserve_others(selected,tmp_path,watched):
    directory,saved=setup(selected,tmp_path,watched)
    original=(directory/'human_checklist.csv').read_bytes();old_rows=read_rows(directory/'human_checklist.csv')
    saved_bytes=saved.read_bytes();old_saved=read_rows(saved)
    preview=preview_source_correction(directory)
    assert preview['after']['selected_label_visible']=='Uncertain'
    assert preview['after']['watched_all']==watched and preview['after']['incident_independence']=='Pending'
    assert (directory/'human_checklist.csv').read_bytes()==original
    with pytest.raises(ValueError,match='Explicit approval'):apply_source_correction(directory,preview)
    result=apply_source_correction(directory,preview,approved=True)
    assert Path(result['backup_path']).read_bytes()==original and saved.read_bytes()==saved_bytes
    new_rows=read_rows(directory/'human_checklist.csv');new_saved=read_rows(result['decision_file'])
    for before,after in zip(old_rows,new_rows):
        if before['source']!=SOURCE:assert before==after
        else:
            assert after['selected_label_visible']=='Uncertain'
            assert after['watched_all']==before['watched_all'] and after['incident_independence']=='Pending'
            assert before['review_notes'] in after['review_notes'] and NOTE in after['review_notes']
            assert before['notes']==after['notes']
    for before,after in zip(old_saved,new_saved):
        if before['source']!=SOURCE:assert before==after
        else:
            assert after['label_visible']=='Uncertain' and after['videos_watched']==before['videos_watched']
            assert before['review_notes'] in after['review_notes'] and NOTE in after['review_notes']
    report=consolidate_review(directory);restored,path=resume_decisions(directory,report)
    assert path==result['decision_file'] and restored[SOURCE]['label_visible']=='Uncertain'
    assert not report['source_mapping_reviewed'] and not report['run_prototype_training']
    # A future blanket sync must not propose restoring the obsolete Confirmed label.
    assert not any(c['source']==SOURCE and c['field']=='selected_label_visible' and c['after']=='Confirmed' for c in preview_checklist_sync(directory)['changes'])
    files_before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    again=apply_source_correction(directory,preview_source_correction(directory),approved=True)
    assert again['message'].startswith('Correction already present')
    assert files_before=={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_newer_conflicting_state_and_stale_preview_stop(selected,tmp_path):
    directory,saved=setup(selected,tmp_path)
    preview=preview_source_correction(directory)
    edit_checklist(directory,lambda rows:rows[0].update(notes='Changed after preview'))
    with pytest.raises(ValueError,match='stale'):apply_source_correction(directory,preview,approved=True)
    edit_checklist(directory,lambda rows:next(r for r in rows if r['source']==SOURCE).update(incident_independence='Confirmed'))
    with pytest.raises(ValueError,match='not Pending'):preview_source_correction(directory)
    assert not list(directory.glob('correction-*'))


def test_report_failure_rolls_back_only_own_changes(selected,tmp_path,monkeypatch):
    directory,saved=setup(selected,tmp_path)
    preview=preview_source_correction(directory);original=(directory/'human_checklist.csv').read_bytes()
    from scripts import review_dashboard
    real=review_dashboard.consolidate_review
    def broken(path):
        if (directory/'human_checklist.csv').read_bytes()!=original:raise ValueError('fixture report failure')
        return real(path)
    monkeypatch.setattr(review_dashboard,'consolidate_review',broken)
    with pytest.raises(ValueError,match='fixture report'):apply_source_correction(directory,preview,approved=True)
    assert (directory/'human_checklist.csv').read_bytes()==original
    assert len(list(directory.glob('correction-*/human_checklist.backup.*.csv')))==1
    assert not (directory/'.checklist_sync.lock').exists()


def test_future_decision_time_cannot_restore_old_confirmed_version(selected,tmp_path):
    directory,saved=setup(selected,tmp_path)
    path=saved.parent/'provenance.json';provenance=json.loads(path.read_text())
    provenance['saved_at_utc']='2099-01-01T00:00:00+00:00';path.write_text(json.dumps(provenance))
    original=(directory/'human_checklist.csv').read_bytes()
    with pytest.raises(ValueError,match='ahead of this runtime'):
        apply_source_correction(directory,preview_source_correction(directory),approved=True)
    assert (directory/'human_checklist.csv').read_bytes()==original
    assert len(list(directory.glob('decisions-*')))==1


def test_legacy_notes_column_keeps_header_and_prior_notes(selected,tmp_path):
    directory,saved=setup(selected,tmp_path)
    checklist=directory/'human_checklist.csv';rows=read_rows(checklist)
    fields=[field for field in rows[0] if field!='review_notes']
    with checklist.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({field:row[field] for field in fields} for row in rows)
    preview=preview_source_correction(directory)
    apply_source_correction(directory,preview,approved=True)
    after=read_rows(checklist)
    assert list(after[0])==fields
    target=next(r for r in after if r['source']==SOURCE)
    assert 'Existing notes: '+SOURCE in target['notes'] and NOTE in target['notes']


def test_callback_requires_preview_and_separate_approval(selected,tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    from scripts.source_correction import render_source_correction
    directory,saved=setup(selected,tmp_path)
    callbacks={};google=ModuleType('google');colab=ModuleType('google.colab')
    colab.output=type('Output',(),{'register_callback':staticmethod(lambda name,cb:callbacks.update({name:cb}))})
    google.colab=colab;ipython=ModuleType('IPython');display=ModuleType('IPython.display')
    display.JSON=lambda value:value;ipython.display=display
    for name,module in [('google',google),('google.colab',colab),('IPython',ipython),('IPython.display',display)]:monkeypatch.setitem(sys.modules,name,module)
    page=render_source_correction(directory)
    assert 'data-approve disabled' in page and 'data-apply disabled' in page
    callback=next(iter(callbacks.values()));original=(directory/'human_checklist.csv').read_bytes()
    preview=callback('preview')
    assert (directory/'human_checklist.csv').read_bytes()==original
    with pytest.raises(ValueError,match='Explicit approval'):callback('apply',preview['token'],False)
    result=callback('apply',preview['token'],True)
    assert result['selected_label_visible']=='Uncertain'
    with pytest.raises(ValueError,match='Preview first'):callback('apply',preview['token'],True)
