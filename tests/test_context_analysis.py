import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.context_analysis import analyze_context,frame_metrics,adjacent_motion
from scripts.context_review import context_plan,save_context_review
from scripts.data import canonical_clip,file_hash
from scripts.review import temporal_samples
from scripts.review_dashboard import consolidate_review,write_dashboard
from scripts.review_interface import render_review_interface,save_review_decisions
from scripts.review_launcher import resume_decisions
from tests.test_original_review import selected
from tests.test_review_interface import answers
from tests.test_review import write_video


def test_quality_motion_static_dark_vs_changing(tmp_path):
    static=tmp_path/'static.avi';moving=tmp_path/'moving.avi'
    blank=np.zeros((32,48,3),np.uint8)
    write_video(static,[blank]*12)
    rng=np.random.default_rng(19)
    frames=[rng.integers(0,256,(32,48,3),dtype=np.uint8) for _ in range(12)]
    write_video(moving,frames)
    sf,sm=temporal_samples(static);mf,mm=temporal_samples(moving)
    static_motion=adjacent_motion(static,sm);moving_motion=adjacent_motion(moving,mm)
    assert static_motion['adjacent_frame_mae']==0
    assert static_motion['median_flow_pixels_per_frame']==0
    assert moving_motion['adjacent_frame_mae']>1
    assert moving_motion['changed_pixel_fraction']>0
    assert frame_metrics(sf)['dark_fraction']==1
    assert frame_metrics(sf)['laplacian_variance']==0
    assert frame_metrics(mf)['texture_std']>12


def test_analysis_all_context_roles_and_determinism(selected,tmp_path):
    plan=context_plan(selected)
    g=plan['groups'][0];original=g['videos'][0]
    other=next(g['videos'][0] for g in plan['groups'] if g['split']!=plan['groups'][0]['split'])
    path=Path(original['path']).with_name('context_1.avi')
    path.write_bytes(Path(other['path']).read_bytes())
    neighbor=dict(original,path=str(path),filename=path.name,canonical_clip=canonical_clip(path,Path(plan['dataset_root'])),
                  sha256=file_hash(path),role='following context',context_only=True,size_bytes=path.stat().st_size)
    g['videos'].append(neighbor)
    before=json.dumps(plan,sort_keys=True)
    result=analyze_context(plan,tmp_path/'analysis')
    repeated=analyze_context(plan,tmp_path/'repeat')
    assert result==repeated
    assert len(result['clips'])==13
    assert sum(r['context_only'] for r in result['clips'])==1
    assert any(p['kind']=='EXACT_BYTES' and p['cross_split'] and (p['left_context_only'] or p['right_context_only']) for p in result['pairs'])
    assert len(result['priorities'])==12
    assert all(r['contact_sheet'] and (tmp_path/'analysis'/r['contact_sheet']).is_file() for r in result['clips'])
    assert result['human_label_decision']==result['incident_independence']=='PENDING'
    assert not result['training_executed']
    assert json.dumps(plan,sort_keys=True)==before
    for name in ['analysis.json','clips.csv','pairs.csv','priority_queue.csv']:
        assert (tmp_path/'analysis'/name).is_file()
    assert all(p['left_source']!=p['right_source'] for p in result['pairs'])
    with pytest.raises(FileExistsError):analyze_context(plan,tmp_path/'analysis')


def test_cross_split_exact_and_uncertain_scene_evidence(selected,tmp_path):
    plan=context_plan(selected)
    a=plan['groups'][0]['videos'][0]
    target=next(g for g in plan['groups'] if g['split']!=a['split'])
    b=target['videos'][0]
    Path(b['path']).write_bytes(Path(a['path']).read_bytes())
    b['sha256']=file_hash(b['path'])
    result=analyze_context(plan,tmp_path/'exact')
    pair=next(p for p in result['pairs'] if {p['left_clip'],p['right_clip']}=={a['canonical_clip'],b['canonical_clip']})
    assert pair['status']=='FAIL' and pair['kind']=='EXACT_BYTES' and pair['cross_split']
    assert result['technical_status']=='FAIL'
    assert result['priorities'][0]['cross_split_suspicion']
    frames,_=temporal_samples(a['path'])
    write_video(Path(b['path']),frames,fps=7)
    b['sha256']=file_hash(b['path'])
    result=analyze_context(plan,tmp_path/'scene')
    pair=next(p for p in result['pairs'] if {p['left_clip'],p['right_clip']}=={a['canonical_clip'],b['canonical_clip']})
    assert pair['kind']=='POSSIBLE_SHARED_SCENE_UNCERTAIN' and pair['status']=='WARNING'
    assert pair['matching_samples']>=3 and pair['histogram_bhattacharyya'] is not None
    assert result['incident_independence']=='PENDING'


def test_failed_decode_and_hash_changes_are_technical_failures(selected,tmp_path,monkeypatch):
    from scripts import context_analysis
    plan=context_plan(selected)
    real_samples=context_analysis.temporal_samples
    broken=plan['groups'][0]['videos'][0]
    def sample(path,count):
        if str(path)==broken['path']:raise ValueError('fixture decoder failure')
        return real_samples(path,count)
    monkeypatch.setattr(context_analysis,'temporal_samples',sample)
    changed=plan['groups'][1]['videos'][0]
    Path(changed['path']).write_bytes(b'changed')
    result=analyze_context(plan,tmp_path/'fail')
    assert result['technical_status']=='FAIL'
    assert sum(r['technical_status']=='FAIL' for r in result['clips'])==2
    assert all(r['human_label_decision']=='PENDING' for r in result['clips'])
    assert not any(p['kind']=='EXACT_BYTES' for p in result['pairs'])


def test_analysis_dashboard_preserves_prior_human_snapshot(selected,tmp_path):
    plan=context_plan(selected)
    directory=save_context_review(plan,tmp_path/'context')
    report=consolidate_review(directory)
    receipt=save_review_decisions(directory,report,answers(report))
    checklist=(directory/'human_checklist.csv').read_bytes()
    candidate=Path(json.loads(selected.read_text())['split_manifest']);before=candidate.read_bytes()
    output=write_dashboard(directory,analyze=True)
    snapshot=json.loads((output/'review_report.json').read_text())
    restored,path=resume_decisions(directory,snapshot)
    assert path==receipt['saved_csv'] and len(restored)==12
    page=render_review_interface(snapshot,plan,'test.callback',restored,path)
    assert page.count('alt="Temporal contact sheet"')==12
    assert 'OpenCV technical evidence' in page
    assert page.count('data-review-source=')==12
    assert (directory/'human_checklist.csv').read_bytes()==checklist
    assert candidate.read_bytes()==before


def test_low_motion_darkness_and_unknown_coverage_warn_not_approve(selected,tmp_path,monkeypatch):
    from scripts import context_analysis
    plan=context_plan(selected)
    clip=plan['groups'][0]['videos'][0]
    write_video(Path(clip['path']),[np.zeros((32,48,3),np.uint8)]*12)
    clip['sha256']=file_hash(clip['path'])
    original_samples=context_analysis.temporal_samples
    def unknown(path,count):
        frames,metadata=original_samples(path,count)
        if str(path)==clip['path']:
            metadata['reported_frames']=None
            metadata['coverage']='Test fixture: unknown length prefix only'
        return frames,metadata
    monkeypatch.setattr(context_analysis,'temporal_samples',unknown)
    result=analyze_context(plan,tmp_path/'warnings')
    record=result['clips'][0]
    assert record['technical_status']=='WARNING'
    assert any(c['name']=='coverage' and c['status']=='WARNING' for c in record['checks'])
    assert any('Low adjacent-frame change' in c['detail'] and 'Mostly dark' in c['detail'] for c in record['checks'])
    assert record['human_label_decision']==record['incident_independence']=='PENDING'
