"""Real-file inspection and source-group splitting for DCSASS."""
import csv
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict
import cv2

CLASSES = ['Burglary', 'Stealing', 'Fighting', 'Vandalism', 'Normal']
VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}


def category(path):
    for part in reversed(Path(path).parts):
        for label in CLASSES:
            if part.lower().startswith(label.lower()):
                return label
    return None


def source_id(path, root):
    rel = Path(path).relative_to(root)
    for i, part in enumerate(rel.parts[:-1]):
        if part.lower().endswith('.mp4'):
            return '/'.join(rel.parts[:i+1])
    match = re.match(r'(.+?_x264)(?:[_-]\d+)?$', rel.stem, re.I)
    return str(rel.parent / (match.group(1) if match else rel.stem))


def inspect(root, output):
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    rows = []
    for p in files:
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        cap = cv2.VideoCapture(str(p)); ok, frame = cap.read(); cap.release()
        rows.append(dict(path=str(p.resolve()), label=category(p), source=source_id(p, root), readable=bool(ok and frame is not None)))
    report = dict(directory_file_counts=dict(Counter(str(p.parent.relative_to(root)) for p in files)),
                  filename_category_candidates=dict(Counter(re.sub(r'[0-9_].*$', '', p.stem) for p in files if p.suffix.lower() in VIDEO_EXT)),
                  file_extensions=dict(Counter(p.suffix.lower() for p in files)),
                  video_counts=dict(Counter(r['label'] or 'unmapped' for r in rows)),
                  readable=sum(r['readable'] for r in rows), unreadable=sum(not r['readable'] for r in rows),
                  directories_named_mp4=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir() and p.suffix.lower()=='.mp4'],
                  annotation_candidates=[str(p.relative_to(root)) for p in files if p.suffix.lower() in {'.csv','.txt','.json','.xml','.yaml'}],
                  annotation_status='Unverified: manually inspect all candidates for bounding boxes and paired narratives.')
    (output/'inspection.json').write_text(json.dumps(report, indent=2))
    with (output/'manifest.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['path','label','source','readable']); w.writeheader(); w.writerows(rows)
    return report


def split_manifest(manifest, output, seed=42, per_class=30):
    rng=random.Random(seed)
    with open(manifest) as f:
        rows=[r for r in csv.DictReader(f) if r['label'] in CLASSES and r['readable'].lower()=='true']
    groups=defaultdict(list)
    for r in rows: groups[r['source']].append(r)
    if any(len({r['label'] for r in v}) != 1 for v in groups.values()):
        raise ValueError('Conflicting source labels')
    selected={label:sorted(k for k,v in groups.items() if v[0]['label']==label) for label in CLASSES}
    count=min(per_class, *(len(v) for v in selected.values()))
    if count < 3: raise ValueError('Need at least 3 independent sources in every class')
    result=[]
    for label, keys in selected.items():
        rng.shuffle(keys); keys=keys[:count]
        nv=max(1,int(count*.2)); nt=count-2*nv
        for i,k in enumerate(keys):
            split='train' if i<nt else 'val' if i<nt+nv else 'test'
            # One representative clip per source keeps prototype source-balanced.
            result.append(dict(rng.choice(groups[k]), split=split))
    with open(output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','label','source','readable','split']); w.writeheader(); w.writerows(result)
    return result
