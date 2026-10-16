import csv
import cv2
import numpy as np
from scripts.data import inspect, split_manifest, CLASSES, source_id


def test_inspection_and_split(tmp_path):
    root=tmp_path/'data'; root.mkdir()
    for label in CLASSES:
        for i in range(3):
            folder=root/label/f'{label}{i:03}_x264.mp4'; folder.mkdir(parents=True)
            for j in range(2):
                writer=cv2.VideoWriter(str(folder/f'{j}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),5,(16,16))
                writer.write(np.zeros((16,16,3),np.uint8)); writer.release()
    (root/'labels.csv').write_text('clip,label\n')
    report=inspect(root,tmp_path/'report')
    assert report['readable']==30 and len(report['directories_named_mp4'])==15
    assert report['annotation_candidates']==['labels.csv']
    rows=split_manifest(tmp_path/'report/manifest.csv',tmp_path/'split.csv')
    sets={s:{r['source'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert all(len(v)==5 for v in sets.values())
    assert not sets['train'] & (sets['val'] | sets['test'])
    assert not sets['val'] & sets['test']
    assert source_id(root/'Burglary/Burglary001_x264_1.mp4',root)==source_id(root/'Burglary/Burglary001_x264_2.mp4',root)


def test_unreadable_and_missing_classes(tmp_path):
    import pytest
    (tmp_path/'bad.mp4').write_bytes(b'not a video')
    report=inspect(tmp_path,tmp_path/'out')
    assert report['unreadable']==1
    with pytest.raises(ValueError,match='at least 3'):
        split_manifest(tmp_path/'out/manifest.csv',tmp_path/'split.csv')
