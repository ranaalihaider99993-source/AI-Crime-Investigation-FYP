import csv
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.data import CLASSES, inspect, split_manifest, write_csv
from scripts.review import review_selected, temporal_samples
from scripts.smoke import prepare_smoke, run_reviewed_smoke, read_csv
from tests.test_smoke import CONFIG


def write_video(path, frames, fps=5):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(48,32))
    assert writer.isOpened()
    for frame in frames: writer.write(frame)
    writer.release()


@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'data';frames={}
    for c,label in enumerate(CLASSES):
        for i in range(3):
            path=root/label/f'{label}{i:03}_x264.mp4/clip.avi'
            rng=np.random.default_rng(100*c+i)
            # Different textured, moving sequences, not crime footage or labels.
            sequence=[rng.integers(0,256,(32,48,3),dtype=np.uint8) for _ in range(24)]
            write_video(path,sequence);frames[str(path)]=sequence
    inspect(root,tmp_path/'inspection')
    split_manifest(tmp_path/'inspection/manifest.csv',tmp_path/'split.csv',per_class=3)
    return root,tmp_path/'split.csv',frames


def test_temporal_report_no_human_approval(dataset,tmp_path):
    root,manifest,_=dataset
    result=review_selected(manifest,root,tmp_path/'review')
    assert result['technical_status']=='PASS'
    assert result['human_review']=='PENDING'
    assert result['semantic_status']==result['incident_independence']=='UNVERIFIED'
    assert result['training_executed'] is False
    assert len(result['clips'])==12
    assert all(c['temporal']['unique_samples']==12 for c in result['clips'])
    assert all(c['temporal']['frame_indices'][0]==0 and c['temporal']['frame_indices'][-1]==23 for c in result['clips'])
    assert not result['pairs']
    page=(tmp_path/'review/review.html').read_text()
    assert page.count('src="data:image/jpeg;base64,')==12
    assert 'UNVERIFIED' in page and 'Human mapping approval' in page
    assert len(read_csv(tmp_path/'review/review_cases.csv'))==12


def test_similar_temporal_sequences_are_uncertain_not_incident_proof(dataset,tmp_path):
    root,manifest,frames=dataset;rows=read_csv(manifest)
    a=rows[0];b=next(r for r in rows if r['split']!=a['split'])
    # Different container timing; same synthetic visual sequence, new source ID.
    write_video(Path(b['path']),frames[a['path']],fps=7)
    result=review_selected(manifest,root,tmp_path/'review')
    assert result['technical_status']=='PASS'
    pair=next(p for p in result['pairs'] if p['left']==0 and result['clips'][p['right']]['source']==b['source'])
    assert pair['kind']=='possible_shared_scene_UNCERTAIN' and pair['cross_split']
    assert pair['temporal_matches']['matching_samples']>=3
    assert result['human_review']=='PENDING'
    assert 'class="warn"' in (tmp_path/'review/review.html').read_text()


def test_exact_duplicates_fail_technical_review(dataset,tmp_path):
    root,manifest,_=dataset;rows=read_csv(manifest)
    a=rows[0];b=next(r for r in rows if r['split']!=a['split'])
    shutil.copyfile(a['path'],b['path'])
    result=review_selected(manifest,root,tmp_path/'review')
    assert result['technical_status']=='FAIL'
    assert any(p['kind']=='exact_file_duplicate' and p['cross_split'] for p in result['pairs'])
    assert result['human_review']=='PENDING'


@pytest.mark.parametrize('problem',['source_overlap','mapping','outside'])
def test_review_rejects_unsafe_manifests(dataset,tmp_path,problem):
    root,manifest,_=dataset;rows=read_csv(manifest)
    if problem=='source_overlap':
        extra=dict(rows[0]);path=Path(extra['path']).with_name('extra.avi')
        shutil.copyfile(extra['path'],path)
        extra.update(path=str(path),canonical_clip=extra['canonical_clip'].rsplit('/',1)[0]+'/extra.avi',split='test' if extra['split']!='test' else 'train')
        rows.append(extra)
    elif problem=='mapping':
        rows[0]['label']='Normal'
    else:
        outside=tmp_path/'outside.avi';shutil.copyfile(rows[0]['path'],outside);rows[0]['path']=str(outside)
    write_csv(manifest,rows,list(rows[0]))
    result=review_selected(manifest,root,tmp_path/'review')
    assert result['technical_status']=='FAIL' and result['errors']
    assert (tmp_path/'review/review.html').is_file()
    assert result['human_review']=='PENDING'


def test_short_static_and_unreadable_videos_are_not_overstated(dataset,tmp_path):
    root,manifest,_=dataset;row=read_csv(manifest)[0]
    write_video(Path(row['path']),[np.zeros((32,48,3),np.uint8)])
    result=review_selected(manifest,root,tmp_path/'review')
    clip=result['clips'][0]
    assert clip['temporal']['unique_samples']==1
    assert any('Too few' in w for w in clip['warnings'])
    assert any('Low visual texture' in w for w in clip['warnings'])
    Path(row['path']).write_bytes(b'not a video')
    result=review_selected(manifest,root,tmp_path/'review')
    assert result['technical_status']=='FAIL' and result['clips'][0]['error']


def test_temporal_report_must_match_guarded_launch(dataset,tmp_path,monkeypatch):
    root,_,_=dataset
    prep=prepare_smoke(root,tmp_path/'inspection',tmp_path/'prep',CONFIG)
    assert prep['automatic_status']=='PASS'
    review_path=tmp_path/'review/review.json'
    review=review_selected(prep['split_manifest'],root,review_path.parent)
    monkeypatch.setattr('scripts.smoke.torch.cuda.is_available',lambda:False)
    with pytest.raises(ValueError,match='No CUDA GPU'):
        run_reviewed_smoke(tmp_path/'prep/validation.json',tmp_path/'runs',CONFIG,True,True,temporal_review_path=review_path)
    review['manifest_sha256']='0'*64;review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='different split'):
        run_reviewed_smoke(tmp_path/'prep/validation.json',tmp_path/'runs',CONFIG,True,True,False,temporal_review_path=review_path)
    review['technical_status']='FAIL';review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='technical failures'):
        run_reviewed_smoke(tmp_path/'prep/validation.json',tmp_path/'runs',CONFIG,True,True,False,temporal_review_path=review_path)
    assert not (tmp_path/'runs').exists()
