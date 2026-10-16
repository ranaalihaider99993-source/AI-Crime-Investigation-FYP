import base64
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import pytest
from scripts.data import CLASSES, inspect
from scripts.smoke import prepare_smoke
from scripts.original_review import original_review_plan, original_players, export_original_package
from tests.test_review import write_video
from tests.test_smoke import CONFIG
import numpy as np

QUOTA=dict(source='Google Drive API storageQuota',limit_bytes=10**10,usage_bytes=0,available_bytes=10**10)


@pytest.fixture
def selected(tmp_path):
    root=tmp_path/'data'
    ids={'Burglary':[41,1,2],'Stealing':[60,54,2],'Fighting':[5,2,1],'Vandalism':[19,43,1]}
    for c,label in enumerate(CLASSES):
        for number in ids[label]:
            rng=np.random.default_rng(c*100+number)
            frames=[rng.integers(0,256,(32,48,3),dtype=np.uint8) for _ in range(6)]
            # Deliberate repeated basename across all sources to test collision handling.
            write_video(root/label/f'{label}{number:03}_x264.mp4'/'original clip.avi',frames)
    inspect(root,tmp_path/'inspection')
    result=prepare_smoke(root,tmp_path/'inspection',tmp_path/'prep',CONFIG)
    assert result['automatic_status']=='PASS'
    return tmp_path/'prep/validation.json'


def test_original_players_preserve_bytes_and_highlight_eight(selected):
    plan=original_review_plan(selected)
    before=Path(plan['manifest_path']).read_bytes()
    players=list(original_players(plan))
    assert len(players)==12
    assert sum('User-reported low motion' in p for p in players)==8
    assert not plan['reported_sources_not_selected']
    assert len(plan['selected_low_motion_sources'])==8
    for clip,player in zip(plan['clips'],players):
        assert '<video controls preload="none"' in player
        assert clip['source'] in player and clip['label'] in player and clip['split'] in player
        payload=re.search(r'base64,([^"<]+)',player).group(1)
        assert base64.b64decode(payload)==Path(clip['path']).read_bytes()
    assert Path(plan['manifest_path']).read_bytes()==before
    assert plan['human_approval']=='PENDING' and plan['training_executed'] is False


def test_package_has_exact_twelve_unchanged_originals(selected,tmp_path,monkeypatch):
    plan=original_review_plan(selected)
    before=Path(plan['manifest_path']).read_bytes()
    # New workflow must not invoke a splitter at any stage.
    monkeypatch.setattr('scripts.data.split_manifest',lambda *args,**kwargs:pytest.fail('resplit'))
    destination=export_original_package(plan,tmp_path/'export',QUOTA)
    with ZipFile(destination) as archive:
        videos=[name for name in archive.namelist() if name.startswith('clips/')]
        assert len(videos)==12 and len(set(videos))==12
        assert len(archive.namelist())==17
        for clip in plan['clips']:
            payload=archive.read(clip['package_path'])
            assert payload==Path(clip['path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==clip['sha256']
            assert Path(clip['package_path']).name==clip['filename']
        assert archive.read('selected_clips.csv')==before
        metadata=json.loads(archive.read('review_metadata.json'))
        assert metadata['human_approval']=='PENDING' and metadata['training_executed'] is False
        assert archive.read('index.html').count(b'<video controls')==12
        assert b'original%20clip.avi' in archive.read('index.html')
        assert archive.read('human_checklist.csv').count(b'PENDING')==36
    original_archive=destination.read_bytes()
    with pytest.raises(FileExistsError,match='not overwritten'):
        export_original_package(plan,tmp_path/'export',QUOTA)
    assert destination.read_bytes()==original_archive
    assert Path(plan['manifest_path']).read_bytes()==before


@pytest.mark.parametrize('failure',['clip_size','total_size','split_changed','video_changed','outside'])
def test_original_plan_rejects_unbound_or_oversized_inputs(selected,tmp_path,failure):
    record=json.loads(selected.read_text());manifest=Path(record['split_manifest'])
    if failure=='clip_size':
        with pytest.raises(ValueError,match='size bound'):original_review_plan(selected,max_clip_bytes=1)
    elif failure=='total_size':
        with pytest.raises(ValueError,match='total'):original_review_plan(selected,max_total_bytes=1)
    elif failure=='split_changed':
        manifest.write_text(manifest.read_text()+'\n')
        with pytest.raises(ValueError,match='split file changed'):original_review_plan(selected)
    elif failure=='video_changed':
        plan=original_review_plan(selected);path=Path(plan['clips'][0]['path'])
        payload=bytearray(path.read_bytes());payload[-1]^=1;path.write_bytes(payload)
        with pytest.raises(ValueError,match='content changed'):original_review_plan(selected)
    else:
        plan=original_review_plan(selected);path=Path(plan['clips'][0]['path'])
        outside=tmp_path/'outside.avi';shutil.copyfile(path,outside);path.unlink();path.symlink_to(outside)
        with pytest.raises(ValueError,match='escapes'):original_review_plan(selected)


@pytest.mark.parametrize('failure',['drive_full','unknown_quota','disk_full','tampered_plan'])
def test_package_storage_and_binding_checks(selected,tmp_path,monkeypatch,failure):
    plan=original_review_plan(selected);quota=dict(QUOTA)
    if failure=='drive_full':quota.update(usage_bytes=QUOTA['limit_bytes'],available_bytes=0)
    if failure=='unknown_quota':quota['source']='unknown'
    if failure=='disk_full':monkeypatch.setattr('scripts.original_review.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    if failure=='tampered_plan':plan['total_video_bytes']=0
    with pytest.raises(ValueError):export_original_package(plan,tmp_path/'export',quota)
    assert not list((tmp_path/'export').glob('*.zip'))
    assert not list((tmp_path/'export').glob('*.partial'))


def test_inline_limit_prevents_unbounded_embedding(selected,monkeypatch):
    plan=original_review_plan(selected)
    monkeypatch.setattr('scripts.original_review.INLINE_TOTAL_BYTES',1)
    with pytest.raises(ValueError,match='inline limit'):
        list(original_players(plan))
    assert len(plan['clips'])==12  # Still available for bounded ZIP export.


def test_failed_stream_integrity_removes_incomplete_archive(selected,tmp_path,monkeypatch):
    import scripts.original_review as module
    plan=original_review_plan(selected)
    metadata=module._metadata_files
    def mutate_after_revalidation(*args):
        result=metadata(*args)
        path=Path(plan['clips'][0]['path'])
        payload=bytearray(path.read_bytes());payload[-1]^=1;path.write_bytes(payload)
        return result
    monkeypatch.setattr(module,'_metadata_files',mutate_after_revalidation)
    with pytest.raises(ValueError,match='changed during export'):
        export_original_package(plan,tmp_path/'export',QUOTA)
    assert not (tmp_path/'export/original_video_review.zip').exists()
    assert not list((tmp_path/'export').glob('*.partial'))


def test_existing_temporal_selection_cannot_be_silently_replaced(selected):
    folder=selected.parent/'temporal_review';folder.mkdir()
    (folder/'review.json').write_text(json.dumps({'canonical_split_signature':'wrong','clips':[]}))
    with pytest.raises(ValueError,match='temporal review belongs to different'):
        original_review_plan(selected)
