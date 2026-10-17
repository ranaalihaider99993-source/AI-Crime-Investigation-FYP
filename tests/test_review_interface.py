import csv
import json
from pathlib import Path
import pytest
from scripts.context_review import context_plan, save_context_review
from scripts.review_dashboard import consolidate_review
from scripts.review_interface import render_review_interface, save_review_decisions
from tests.test_original_review import selected


def setup_review(selected,tmp_path):
    plan=context_plan(selected)
    directory=save_context_review(plan,tmp_path/'context')
    return plan,directory,consolidate_review(directory)


def answers(report):
    return [dict(source=r['source'],label_visible='Uncertain',videos_watched='Confirmed',
        incident_independence='Pending',reviewer='Human reviewer',review_notes='=note; uncertain label') for r in report['rows']]


def test_controls_video_grouping_and_pending_defaults(selected,tmp_path):
    plan,directory,report=setup_review(selected,tmp_path)
    report['rows'][0]['review_notes']='</textarea><script>bad()</script>'
    rendered=render_review_interface(report,plan,'test.callback')
    assert rendered.count('<video controls')==12
    assert rendered.count('data-review-source=')==12
    assert rendered.count('<option>Pending</option>')==36
    assert 'selected=' not in rendered  # No positive option selected from metadata.
    assert rendered.count('CLOSER INSPECTION')==5  # 3 Fighting + 2 reported-low-motion Vandalism.
    assert '</textarea><script>bad()' not in rendered
    assert '&lt;/textarea&gt;' in rendered
    assert 'kernel.invokeFunction' in rendered
    assert 'data-field="review_notes"' in rendered


def test_save_explicit_decisions_without_overwriting_inputs(selected,tmp_path):
    plan,directory,report=setup_review(selected,tmp_path)
    checklist=(directory/'human_checklist.csv').read_bytes()
    candidate=Path(plan['manifest_path']) if 'manifest_path' in plan else Path(json.loads(selected.read_text())['split_manifest'])
    before=candidate.read_bytes()
    payload=answers(report)
    result=save_review_decisions(directory,report,payload)
    path=Path(result['saved_csv'])
    with path.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==12 and all(r['videos_watched']=='Confirmed' for r in rows)
    assert all(r['review_notes']=="'=note; uncertain label" for r in rows)
    assert all(r['source_mapping_reviewed']=='False' and r['run_prototype_training']=='False' for r in rows)
    assert (path.parent/'provenance.json').is_file()
    assert save_review_decisions(directory,report,payload)['saved_csv']!=str(path)
    assert (directory/'human_checklist.csv').read_bytes()==checklist
    assert candidate.read_bytes()==before


@pytest.mark.parametrize('kind',['missing','duplicate','unknown','bad_option','forged_identity'])
def test_reject_invalid_decision_payload(selected,tmp_path,kind):
    plan,directory,report=setup_review(selected,tmp_path)
    payload=answers(report)
    if kind=='missing': payload.pop()
    if kind=='duplicate': payload[1]=dict(payload[0])
    if kind=='unknown': payload[0]['source']='unknown'
    if kind=='bad_option': payload[0]['label_visible']='auto approved'
    if kind=='forged_identity': payload[0]['split']='test'
    with pytest.raises(ValueError): save_review_decisions(directory,report,payload)
    assert not list(directory.glob('decisions-*'))


def test_save_rejects_stale_inputs(selected,tmp_path):
    plan,directory,report=setup_review(selected,tmp_path)
    with (directory/'human_checklist.csv').open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='inputs changed'):
        save_review_decisions(directory,report,answers(report))
    assert not list(directory.glob('decisions-*'))


def test_interactive_dashboard_callback_end_to_end(selected,tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    from scripts.review_dashboard import write_dashboard
    plan,directory,report=setup_review(selected,tmp_path)
    registered={}
    google=ModuleType('google');colab=ModuleType('google.colab')
    colab.output=type('Output',(),{'register_callback':staticmethod(lambda name,callback:registered.update({name:callback}))})
    google.colab=colab
    ipython=ModuleType('IPython');display=ModuleType('IPython.display')
    display.JSON=lambda result:result
    ipython.display=display
    monkeypatch.setitem(sys.modules,'google',google)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    monkeypatch.setitem(sys.modules,'IPython',ipython)
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    output=write_dashboard(directory,interactive=True)
    assert len(registered)==1
    name,callback=next(iter(registered.items()))
    page=(output/'review_dashboard.html').read_text()
    assert name in page and page.count('<video controls')==12
    assert '<details><summary>Technical evidence' in page
    receipt=callback(answers(report))
    assert Path(receipt['saved_csv']).is_file()
    assert not json.loads((Path(receipt['saved_csv']).parent/'provenance.json').read_text())['training_executed']
