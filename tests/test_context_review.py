import csv
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.context_review import context_plan, context_players, save_context_review, segment_number
from scripts.data import CLASSES, inspect
from scripts.smoke import prepare_smoke
from tests.test_original_review import selected
from tests.test_review import write_video
from tests.test_smoke import CONFIG


def test_unknown_order_stays_pending(selected, tmp_path):
    plan = context_plan(selected)
    assert len(plan['groups']) == 12
    assert all(len(g['videos']) == 1 and 'ambiguous' in g['warnings'][0] for g in plan['groups'])
    out = save_context_review(plan, tmp_path/'review')
    assert save_context_review(plan, tmp_path/'review') != out
    with (out/'human_checklist.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and all(r['incident_independence']=='PENDING' for r in rows)
    assert plan['human_approval']=='PENDING' and not plan['training_executed']
    assert sum('<video ' in p for p in context_players(plan)) == 12


def test_numeric_neighbors_boundaries_and_unchanged_candidate(tmp_path):
    root = tmp_path/'dataset'
    for c,label in enumerate(CLASSES):
        for source in range(3):
            for number in [1,2,10]:
                rng = np.random.default_rng(c*100+source*10+number)
                path = root/'DCSASS Dataset'/label/f'{label}{source:03}_x264.mp4'/f'{label}{source:03}_x264_{number}.avi'
                write_video(path,[rng.integers(0,256,(32,48,3),dtype=np.uint8) for _ in range(6)])
    inspect(root,tmp_path/'inspection')
    result = prepare_smoke(root,tmp_path/'inspection',tmp_path/'prep',CONFIG)
    assert result['automatic_status']=='PASS'
    candidate = Path(result['split_manifest'])
    before = candidate.read_bytes()
    plan = context_plan(tmp_path/'prep/validation.json')
    assert candidate.read_bytes()==before
    assert len(plan['groups'])==12
    assert len({g['source'] for g in plan['groups']})==12
    for g in plan['groups']:
        numbers = [segment_number(v['filename']) for v in g['videos']]
        assert numbers == sorted(numbers)
        assert all(v['source']==g['source'] and v['split']==g['split'] for v in g['videos'])
        assert sum(not v['context_only'] for v in g['videos'])==1
        assert len(g['videos']) in [2,3]
    assert any('Numbering gap' in w for g in plan['groups'] for w in g['warnings'])
    neighbor = next(v for g in plan['groups'] for v in g['videos'] if v['context_only'])
    path = Path(neighbor['path'])
    original = path.read_bytes()
    path.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='Context content|Selected original'):
        context_plan(tmp_path/'prep/validation.json')
    path.write_bytes(original)
    outside = tmp_path/'outside.avi'
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError,match='escapes|identity|outside'):
        context_plan(tmp_path/'prep/validation.json')


def test_stale_inspection_rejected(selected):
    validation = json.loads(selected.read_text())
    manifest = Path(validation['inspection_dir'])/'manifest.csv'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='Inspection artifacts changed'):
        context_plan(selected)
