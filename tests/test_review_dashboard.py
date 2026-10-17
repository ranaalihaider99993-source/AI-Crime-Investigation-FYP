import csv
import json
from pathlib import Path
import pytest
from scripts.context_review import context_plan, save_context_review
from scripts.review_dashboard import consolidate_review, write_dashboard, source_056_observation, OBSERVATION_056
from scripts.review import review_selected
from tests.test_original_review import selected


def edit_checklist(directory, change):
    path = directory/'human_checklist.csv'
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    change(rows)
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_consolidation_preserves_notes_flags_and_candidate(selected,tmp_path):
    plan = context_plan(selected)
    directory = save_context_review(plan,tmp_path/'context')
    def annotate(rows):
        other = next(r for r in rows if r['split'] != rows[0]['split'])
        rows[0].update(notes='<script>alert(1)</script>', review_notes='=malicious_formula', selected_label_visible='UNCERTAIN', possible_related_sources=other['source'])
        rows[1].update(review_notes='=untrusted_formula')
    edit_checklist(directory, annotate)
    checklist_before = (directory/'human_checklist.csv').read_bytes()
    candidate_before = Path(json.loads(selected.read_text())['split_manifest']).read_bytes()
    report = consolidate_review(directory)
    assert len(report['rows'])==12
    assert sum('USER-REPORTED LOW MOTION' in r['flags'] for r in report['rows'])==8
    assert 'LABEL REVIEW REQUIRED' in report['rows'][0]['flags']
    assert '<script>' in report['rows'][0]['review_notes']
    assert any(p['kind']=='USER_REPORTED_POSSIBLE_RELATION' and p['cross_split'] for p in report['pairs'])
    assert report['human_approval']=='PENDING' and not report['training_executed']
    assert not report['source_mapping_reviewed'] and not report['run_prototype_training']
    assert report['observation_056']['matched_source'] is None
    output = write_dashboard(directory)
    rendered = (output/'review_dashboard.html').read_text()
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert OBSERVATION_056 in rendered and 'UNRESOLVED' in rendered
    assert write_dashboard(directory)!=output
    assert (directory/'human_checklist.csv').read_bytes()==checklist_before
    assert Path(json.loads(selected.read_text())['split_manifest']).read_bytes()==candidate_before
    with (output/'review_report.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
        assert len(csv_rows)==12
        assert csv_rows[1]['review_notes']=="'=untrusted_formula"


def test_observation_identity_is_not_guessed():
    unique = source_056_observation(['stealing056_x264','burglary041_x264'])
    assert unique['matched_source']=='stealing056_x264'
    assert unique['attribution']=='USER-REPORTED; NOT VERIFIED'
    assert source_056_observation(['stealing056_x264','burglary056_x264'])['matched_source'] is None
    assert source_056_observation(['stealing1056_x264'])['matched_source'] is None


def test_temporal_evidence_cross_split_and_stale_rejection(selected,tmp_path):
    validation = json.loads(selected.read_text())
    temporal = selected.parent/'temporal_review'
    review_selected(validation['split_manifest'],validation['data_root'],temporal)
    path = temporal/'review.json'
    report = json.loads(path.read_text())
    # Inject an explicitly marked saved-evidence fixture to exercise consolidation,
    # not a claim that the generated videos depict a real shared incident.
    left=0
    right=next(i for i,c in enumerate(report['clips']) if c['split']!=report['clips'][left]['split'])
    report['pairs']=[dict(left=left,right=right,kind='possible_shared_scene_UNCERTAIN',detail='Test fixture heuristic',cross_split=True)]
    path.write_text(json.dumps(report))
    directory = save_context_review(context_plan(selected),tmp_path/'context')
    result = consolidate_review(directory)
    assert any(p['cross_split'] for p in result['pairs'])
    assert sum(r['possible_cross_split_overlap'].startswith('FLAGGED') for r in result['rows'])==2
    assert result['human_approval']=='PENDING'
    report['canonical_split_signature']='stale'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='temporal review|Temporal evidence'):
        consolidate_review(directory)


def test_checklist_and_context_identity_changes_rejected(selected,tmp_path):
    directory=save_context_review(context_plan(selected),tmp_path/'context')
    edit_checklist(directory,lambda rows: rows[0].update(split='wrong'))
    with pytest.raises(ValueError,match='Checklist identity'):
        consolidate_review(directory)
    directory=save_context_review(context_plan(selected),tmp_path/'context')
    path=directory/'context_metadata.json'
    saved=json.loads(path.read_text());saved['groups'][0]['source']='wrong'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Saved context differs'):
        consolidate_review(directory)
