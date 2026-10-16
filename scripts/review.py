"""Local OpenCV temporal review: technical checks and explicitly uncertain similarity.

No learned weights, crime inference, incident-independence certification or approval.
"""
import argparse
import base64
import csv
import html
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.data import (CLASSES, canonical_clip, class_summary, file_hash,
                          source_id, split_signature, validate_manifest)
from scripts.dataset_paths import resolve_dataset_root

SAMPLES = 12


def temporal_samples(path, count=SAMPLES):
    """At most count unique, evenly spaced frames; report metadata/coverage limits."""
    if not 4 <= count <= 24:
        raise ValueError('Temporal review requires 4–24 requested samples')
    cap = cv2.VideoCapture(str(path))
    frames, indices = [], []
    try:
        raw_n, raw_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        n = int(raw_n) if np.isfinite(raw_n) and raw_n > 0 else 0
        fps = float(raw_fps) if np.isfinite(raw_fps) and raw_fps > 0 else None
        if n:
            indices = np.unique(np.linspace(0, n-1, min(n,count)).astype(int)).tolist()
            for pos in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f'Cannot decode temporal sample frame {pos}')
                frames.append(frame)
            coverage = 'Evenly spaced across reported frame range; not a full decode'
        else:
            # Unknown-length streams are NOT falsely described as fully sampled.
            prefix = []
            for _ in range(240):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                # Bound memory for unknown-length/high-resolution videos.
                h,w=frame.shape[:2];scale=min(1.,640/w,360/h)
                prefix.append(cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale)))))
            if not prefix:
                raise ValueError('No readable video frames')
            indices = np.unique(np.linspace(0,len(prefix)-1,min(len(prefix),count)).astype(int)).tolist()
            frames = [prefix[i] for i in indices]
            coverage = 'Unknown total length; samples cover only first at most 240 decoded frames'
        return frames, dict(reported_frames=n or None, fps=fps,
                            duration_seconds=n/fps if n and fps else None,
                            frame_indices=indices, timestamps_seconds=[i/fps if fps else None for i in indices],
                            unique_samples=len(frames), coverage=coverage)
    finally:
        cap.release()


def fingerprint(frame):
    gray = cv2.cvtColor(cv2.resize(frame,(32,32)), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray,(9,8))
    bits = (small[:,1:] > small[:,:-1]).flatten()
    value = sum(int(v) << i for i,v in enumerate(bits))
    return dict(dhash=value, gray=gray, texture=float(gray.std()))


def compare_temporal(left, right, max_hamming=6, max_mae=12., min_texture=12.):
    """Greedy one-to-one frame matches, allowing temporal offsets. HEURISTIC ONLY.

    Shared camera backgrounds/static textured scenes can match. Missing matches do
    not establish incident independence. Low-texture frames are excluded.
    """
    candidates = []
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if min(a['texture'],b['texture']) < min_texture:
                continue
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            mae = float(np.abs(a['gray'].astype(float)-b['gray'].astype(float)).mean())
            if distance <= max_hamming and mae <= max_mae:
                candidates.append((distance,mae,i,j))
    used_a, used_b, matches = set(), set(), []
    for distance,mae,i,j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j)
            matches.append(dict(left_sample=i,right_sample=j,hamming=distance,gray_mae=round(mae,3)))
    denominator = min(len(left),len(right))
    fraction = len(matches)/denominator if denominator else 0.
    return dict(flagged=len(matches)>=3 and fraction>=.5, matching_samples=len(matches),
                fraction=round(fraction,3), examples=matches[:5])


def _sheet(frames, metadata):
    tiles=[]
    for frame, idx, time in zip(frames,metadata['frame_indices'],metadata['timestamps_seconds']):
        tile=np.zeros((114,192,3),np.uint8)
        # Letterbox previews rather than changing scene/people proportions.
        h,w=frame.shape[:2];scale=min(192/w,90/h)
        preview=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
        y=(90-preview.shape[0])//2;x=(192-preview.shape[1])//2
        tile[y:y+preview.shape[0],x:x+preview.shape[1]]=preview
        text=f'frame {idx}' + (f' / {time:.2f}s' if time is not None else ' / time unknown')
        cv2.putText(tile,text,(4,106),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        tiles.append(tile)
    while len(tiles)%4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    ok, encoded=cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise ValueError('Cannot encode temporal contact sheet')
    return base64.b64encode(encoded).decode()


def _render(report, pictures):
    esc=lambda s:html.escape(str(s),quote=True)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Selected clip review</title>',
           '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#f7f8fa;color:#17212b}h1,h2{line-height:1.2}article{padding:16px;margin:16px 0;background:white;border:2px solid #bac5cf;border-radius:8px}.warn{border-color:#ba6900;background:#fff6e8}.error{border-color:#b02020;background:#fff0f0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #abb8c4;padding:8px;text-align:left}code{overflow-wrap:anywhere}a{color:#075fa0}</style>',
           '<h1>Selected smoke-clip temporal review</h1>',
           f'<p><strong>Scope: {esc(report["scope"])}</strong></p>',
           f'<p>Technical checks: <strong>{esc(report["technical_status"])}</strong>. Human mapping approval: <strong>PENDING</strong>. Training: <strong>NOT EXECUTED</strong>.</p>',
           '<p>Class labels below are dataset naming assignments, not verified crime judgments. Similarity flags are possible shared scenes, not proof of the same incident. No flags does not prove independence. Inspect original clips before approving.</p>']
    if report['errors']:
        parts+=['<div class="error"><h2>Technical failures — do not train</h2><ul>']+[f'<li>{esc(x)}</li>' for x in report['errors']]+['</ul></div>']
    parts+=['<h2>Verified technical scope</h2><ul>']+[f'<li>{esc(x)}</li>' for x in report['verified_checks']]+['</ul>']
    parts+=['<h2>Suspicious pairs</h2><p>Orange: uncertain visual similarity needing review. Red: byte-identical files (technical duplicate).</p><table><tr><th>Clips</th><th>Splits</th><th>Finding</th><th>Evidence / action</th></tr>']
    for pair in report['pairs']:
        a,b=pair['left'],pair['right']
        finding='Exact duplicate video bytes' if pair['kind']=='exact_file_duplicate' else 'Possible shared scene (uncertain)'
        parts.append(f'<tr class="{"error" if pair["kind"]=="exact_file_duplicate" else "warn"}"><td><a href="#clip-{a}">{esc(report["clips"][a]["source"])}</a> ↔ <a href="#clip-{b}">{esc(report["clips"][b]["source"])}</a></td><td>{esc(report["clips"][a]["split"])} / {esc(report["clips"][b]["split"])}</td><td>{esc(finding)}</td><td>{esc(pair["detail"])} — inspect both original clips; correct grouping before training if related.</td></tr>')
    if not report['pairs']:
        parts.append('<tr><td colspan="4">No pair crossed the selected heuristic thresholds. Incident independence remains unverified.</td></tr>')
    parts+=['</table><h2>Temporal samples</h2>']
    for i,clip in enumerate(report['clips']):
        style='error' if clip.get('error') else 'warn' if clip['warnings'] else ''
        parts+=[f'<article class="{style}" id="clip-{i}"><h3>{esc(clip["split"])} · {esc(clip["label"])} · {esc(clip["source"])}</h3>',
                f'<p><code>{esc(clip["canonical_clip"])}</code></p>',
                '<ul>']+[f'<li>{esc(w)}</li>' for w in clip['warnings']]+['</ul>']
        if clip.get('error'):
            parts.append(f'<p>{esc(clip["error"])}</p>')
        if i in pictures:
            parts.append(f'<img alt="Temporal frames for {esc(clip["source"])}" src="data:image/jpeg;base64,{pictures[i]}">')
            parts.append(f'<p>{esc(clip["temporal"]["unique_samples"])} unique samples. {esc(clip["temporal"]["coverage"])}</p>')
        parts.append('</article>')
    parts+=['<h2>Limits and next action</h2><ul>']+[f'<li>{esc(x)}</li>' for x in report['limitations']]+['</ul>',
            '<p>Keep SOURCE_MAPPING_REVIEWED=False and RUN_PROTOTYPE_TRAINING=False until actual mapping review and explicit approval. This utility cannot give that approval.</p></html>']
    return '\n'.join(parts)


def review_selected(manifest, dataset_root, output, samples=SAMPLES, scope='Selected DCSASS clips; semantic labels and incident independence UNVERIFIED'):
    """Write self-contained HTML/JSON/CSV without modifying inputs or any approval."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    report=dict(scope=scope, technical_status='FAIL',human_review='PENDING',training_executed=False,
                errors=[],verified_checks=[],clips=[],pairs=[],requested_samples=samples,
                limitations=['Only the 12 selected clips are compared, not the whole dataset.',
                             'dHash + gray pixel similarity are uncalibrated classical heuristics, not trained crime/incident models.',
                             'Shared camera backgrounds can trigger false positives; crops, re-encoding, camera changes or temporal shifts may evade matching.',
                             'Low-texture/static clips need special review. Sparse temporal samples cannot certify labels or full readability.',
                             'One held-out source per class is a pipeline smoke test, not reliable accuracy.'])
    pictures, features={},{}
    try:
        root=resolve_dataset_root(dataset_root)
        with Path(manifest).open() as f:
            rows=list(csv.DictReader(f))
        validate_manifest(rows,require_splits=True,require_classes=True)
        if len(rows)!=12 or len({r['source'] for r in rows})!=12:
            raise ValueError('Review expects exactly 12 clips from 12 distinct canonical source IDs')
        for row in rows:
            path=Path(row['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Clip path escapes verified dataset root')
            if canonical_clip(path,root)!=row['canonical_clip'] or source_id(path,root)!=row['source']:
                raise ValueError('Physical path/canonical identity mismatch')
        report.update(canonical_split_signature=split_signature(manifest),manifest_sha256=file_hash(manifest),
                      class_order=CLASSES,split_counts={s:class_summary([r for r in rows if r['split']==s]) for s in ['train','val','test']})
        report['verified_checks']+=['All 12 paths exist and remain inside the verified dataset root.',
                                   'Canonical clip/source IDs and four-class naming assignments are consistent.',
                                   'Canonical clips are unique; source IDs have zero train/validation/test overlap.']
        for i,row in enumerate(rows):
            clip={k:row[k] for k in ['canonical_clip','source','label','split']}
            clip.update(warnings=[],sha256=None)
            report['clips'].append(clip)
            try:
                clip['sha256']=file_hash(row['path'])
                frames,metadata=temporal_samples(row['path'],samples)
                clip['temporal']=metadata
                features[i]=[fingerprint(frame) for frame in frames]
                pictures[i]=_sheet(frames,metadata)
                if metadata['unique_samples']<4:
                    clip['warnings'].append('Too few distinct frames for meaningful temporal/similarity review; inspect original video.')
                if metadata['reported_frames'] is None:
                    clip['warnings'].append('Unknown frame count: only a bounded prefix was sampled.')
                if np.median([v['texture'] for v in features[i]])<12:
                    clip['warnings'].append('Low visual texture: similarity checks have weak evidence.')
                motion=[float(np.abs(a['gray'].astype(float)-b['gray'].astype(float)).mean()) for a,b in zip(features[i],features[i][1:])]
                if motion and np.median(motion)<1.:
                    clip['warnings'].append('Little change across samples: possible static view; label/action remains uncertain.')
            except Exception as exc:
                clip['error']=f'{type(exc).__name__}: {exc}'
                report['errors'].append(f'{row["source"]}: {clip["error"]}')
        for i in range(len(rows)):
            for j in range(i+1,len(rows)):
                a,b=report['clips'][i],report['clips'][j]
                same_bytes=a['sha256'] and a['sha256']==b['sha256']
                match=compare_temporal(features.get(i,[]),features.get(j,[]))
                if same_bytes or match['flagged']:
                    kind='exact_file_duplicate' if same_bytes else 'possible_shared_scene_UNCERTAIN'
                    detail='SHA256 matches: same video bytes under distinct source IDs' if same_bytes else f"{match['matching_samples']} one-to-one temporal frame matches ({match['fraction']:.0%}); shared background or related footage is possible"
                    report['pairs'].append(dict(left=i,right=j,kind=kind,cross_split=a['split']!=b['split'],detail=detail,temporal_matches=match))
                    finding='Exact duplicate video bytes' if same_bytes else 'Possible shared scene (uncertain)'
                    for clip in [a,b]:
                        clip['warnings'].append(f'{finding}: compare with {b["source"] if clip is a else a["source"]}')
                    if same_bytes:
                        report['errors'].append(f'Byte-identical clips have different canonical source IDs: {a["source"]} / {b["source"]}')
        if not report['errors']:
            report['technical_status']='PASS'
            report['verified_checks'].append('Requested temporal samples decoded and exact-byte comparisons completed for the selected clips (not full-video decoding).')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    report['semantic_status']='UNVERIFIED'
    report['incident_independence']='UNVERIFIED'
    (output/'review.json').write_text(json.dumps(report,indent=2))
    (output/'review.html').write_text(_render(report,pictures))
    with (output/'review_cases.csv').open('w',newline='') as f:
        fields=['source','label','split','canonical_clip','warnings','error']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for clip in report['clips']:
            w.writerow({k:'; '.join(clip['warnings']) if k=='warnings' else clip.get(k,'') for k in fields})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('dataset_root');p.add_argument('output');p.add_argument('--samples',type=int,default=SAMPLES)
    a=p.parse_args();r=review_selected(a.manifest,a.dataset_root,a.output,a.samples)
    print(json.dumps({k:r[k] for k in ['technical_status','human_review','errors']},indent=2))
    raise SystemExit(0 if r['technical_status']=='PASS' else 1)
