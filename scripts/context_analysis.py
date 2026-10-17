"""Bounded deterministic OpenCV evidence, never crime/incident classification."""
import base64
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.data import canonical_clip, file_hash, source_id
from scripts.review import temporal_samples, fingerprint, compare_temporal, _sheet
from scripts.smoke import require

PARAMETERS = dict(samples=12, analysis_size=[160,90], low_motion_mae=1., blur_variance=30.,
                  dark_fraction=.8, bright_fraction=.8, low_texture_std=12.,
                  similarity_max_hamming=6, similarity_max_mae=12., similarity_min_matches=3,
                  similarity_min_fraction=.5)
RANK = {'PASS':0,'WARNING':1,'FAIL':2}


def frame_metrics(frames):
    """Quality proxies at a fixed resolution, not validated crime visibility tests."""
    gray=[cv2.cvtColor(cv2.resize(f,(160,90)),cv2.COLOR_BGR2GRAY) for f in frames]
    return dict(brightness_mean=float(np.mean([g.mean() for g in gray])),
                dark_fraction=float(np.mean([np.mean(g<20) for g in gray])),
                bright_fraction=float(np.mean([np.mean(g>235) for g in gray])),
                laplacian_variance=float(np.median([cv2.Laplacian(g,cv2.CV_64F).var() for g in gray])),
                texture_std=float(np.median([g.std() for g in gray])),
                sampled_frame_difference_mae=float(np.median([np.abs(a.astype(float)-b.astype(float)).mean() for a,b in zip(gray,gray[1:])])) if len(gray)>1 else None,
                min_width=min(f.shape[1] for f in frames),min_height=min(f.shape[0] for f in frames))


def adjacent_motion(path, metadata):
    """MAE and Farneback flow on adjacent pairs at sampled positions (not full decode)."""
    cap=cv2.VideoCapture(str(path));mae=[];flow=[];changed=[]
    requested=[i for i in metadata['frame_indices'] if metadata['reported_frames'] is None or i+1<metadata['reported_frames']]
    try:
        for index in requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,a=cap.read();ok2,b=cap.read()
            if not ok or not ok2 or a is None or b is None: continue
            a=cv2.cvtColor(cv2.resize(a,(160,90)),cv2.COLOR_BGR2GRAY)
            b=cv2.cvtColor(cv2.resize(b,(160,90)),cv2.COLOR_BGR2GRAY)
            delta=np.abs(a.astype(float)-b.astype(float))
            dense=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
            mae.append(float(delta.mean()));changed.append(float(np.mean(delta>10)))
            flow.append(float(np.median(np.linalg.norm(dense,axis=2))))
    finally:
        cap.release()
    return dict(adjacent_pairs_requested=len(requested),adjacent_pairs_decoded=len(mae),
                adjacent_frame_mae=float(np.median(mae)) if mae else None,
                changed_pixel_fraction=float(np.median(changed)) if changed else None,
                median_flow_pixels_per_frame=float(np.median(flow)) if flow else None)


def analyze_context(plan, output):
    """Analyze only the already validated 12 sources and bounded context (<=36 clips)."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    require(len(plan['groups'])==12 and len({g['source'] for g in plan['groups']})==12,'Exactly 12 disjoint source groups required')
    require(all(1<=len(g['videos'])<=3 and sum(not c['context_only'] for c in g['videos'])==1 for g in plan['groups']),'Each source requires one selected clip and at most two context clips')
    require(sum(len(g['videos']) for g in plan['groups'])<=36,'At most selected + two context clips per source')
    root=Path(plan['dataset_root']).resolve()
    records=[];features={};histograms={};seen=set()
    for group in plan['groups']:
        for clip in group['videos']:
            key=clip['canonical_clip']
            require(key not in seen,'Duplicate canonical analysis clip');seen.add(key)
            record={k:clip[k] for k in ['canonical_clip','source','label','split','filename','role','context_only','sha256']}
            require(record['source']==group['source'] and record['split']==group['split'] and record['label']==group['label'],'Context source/split/class differs')
            record.update(checks=[],metrics={},sampling=None,contact_sheet=None,content_verified=False,human_label_decision='PENDING',incident_independence='PENDING')
            def check(name,status,detail):record['checks'].append(dict(name=name,status=status,detail=detail))
            try:
                path=Path(clip['path']).resolve()
                require(path.is_relative_to(root),'Analysis file outside verified dataset root')
                require(canonical_clip(path,root)==key and source_id(path,root)==record['source'],'Analysis canonical identity differs')
                require(file_hash(path)==clip['sha256'],'Analysis file SHA-256 changed')
                check('identity_and_sha256','PASS','Contained canonical identity and saved content hash match')
                frames,metadata=temporal_samples(path,PARAMETERS['samples'])
                record['sampling']=metadata
                check('sample_decode','PASS',f"Decoded {len(frames)} samples; {metadata['coverage']}")
                quality=frame_metrics(frames);motion=adjacent_motion(path,metadata)
                record['metrics']={**quality,**motion}
                limitations=[]
                if metadata['reported_frames'] is None:limitations.append('Unknown length; bounded prefix only')
                if len(frames)<4:limitations.append('Fewer than four distinct sample positions')
                if motion['adjacent_pairs_decoded']<motion['adjacent_pairs_requested'] or not motion['adjacent_pairs_decoded']:
                    limitations.append('Adjacent motion pairs incomplete/unavailable')
                check('coverage','WARNING' if limitations else 'PASS','; '.join(limitations) or 'Across reported range plus adjacent pairs; sparse sampling can miss events')
                warnings=[]
                if motion['adjacent_frame_mae'] is not None and motion['adjacent_frame_mae']<PARAMETERS['low_motion_mae']:warnings.append('Low adjacent-frame change; static camera, inactivity or subtle action possible')
                if quality['laplacian_variance']<PARAMETERS['blur_variance']:warnings.append('Low sharpness proxy')
                if quality['texture_std']<PARAMETERS['low_texture_std']:warnings.append('Low texture; similarity evidence is weak')
                if quality['dark_fraction']>PARAMETERS['dark_fraction']:warnings.append('Mostly dark sampled pixels')
                if quality['bright_fraction']>PARAMETERS['bright_fraction']:warnings.append('Mostly bright sampled pixels')
                if quality['min_width']<160 or quality['min_height']<90:warnings.append('Low native resolution')
                check('motion_and_quality','WARNING' if warnings else 'PASS','; '.join(warnings) or 'No configured proxy thresholds triggered; does not verify labels')
                require(file_hash(path)==clip['sha256'],'Analysis file changed during decoding')
                record['content_verified']=True
                image=_sheet(frames,metadata)
                record['contact_sheet']=f'contact_{len(records):02}.jpg'
                (output/record['contact_sheet']).write_bytes(base64.b64decode(image))
                features[key]=[fingerprint(f) for f in frames]
                histograms[key]=np.mean([cv2.calcHist([cv2.cvtColor(cv2.resize(f,(160,90)),cv2.COLOR_BGR2GRAY)],[0],None,[32],[0,256]).ravel() for f in frames],axis=0).astype('float32')
            except (ValueError,OSError,cv2.error) as exc:
                check('file_decode_integrity','FAIL',str(exc))
                features.pop(key,None);histograms.pop(key,None)
            record['technical_status']=max((c['status'] for c in record['checks']),key=RANK.get)
            records.append(record)
    pairs=[]
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if left['source']==right['source']:continue
            a,b=left['canonical_clip'],right['canonical_clip']
            exact=left['content_verified'] and right['content_verified'] and left['sha256']==right['sha256']
            match=compare_temporal(features.get(a,[]),features.get(b,[]))
            available=a in histograms and b in histograms
            pairs.append(dict(left_clip=a,right_clip=b,left_source=left['source'],right_source=right['source'],
                left_split=left['split'],right_split=right['split'],left_role=left['role'],right_role=right['role'],
                left_context_only=left['context_only'],right_context_only=right['context_only'],cross_split=left['split']!=right['split'],
                status='FAIL' if exact else 'WARNING' if match['flagged'] or not available else 'PASS',
                kind='EXACT_BYTES' if exact else 'POSSIBLE_SHARED_SCENE_UNCERTAIN' if match['flagged'] else 'UNAVAILABLE' if not available else 'NO_THRESHOLD_FLAG',
                matched_sample_fraction=match['fraction'],matching_samples=match['matching_samples'],examples=match['examples'],
                histogram_bhattacharyya=float(cv2.compareHist(histograms[a],histograms[b],cv2.HISTCMP_BHATTACHARYYA)) if available else None,
                meaning='Exact bytes identify a technical duplicate, not a semantic incident judgment' if exact else
                'Scene resemblance is uncertain; shared backgrounds can match. No match does not establish independence'))
    for record in records:
        relevant=[p for p in pairs if record['canonical_clip'] in (p['left_clip'],p['right_clip']) and p['kind'] in {'EXACT_BYTES','POSSIBLE_SHARED_SCENE_UNCERTAIN'}]
        if relevant:
            record['checks'].append(dict(name='cross_source_similarity',status=max((p['status'] for p in relevant),key=RANK.get),detail=f'{len(relevant)} suspicious clip relationships; inspect pairs, especially across splits'))
            record['technical_status']=max((c['status'] for c in record['checks']),key=RANK.get)
    priorities=[]
    for group in plan['groups']:
        clips=[r for r in records if r['source']==group['source']]
        related=[p for p in pairs if group['source'] in (p['left_source'],p['right_source']) and p['kind'] in {'EXACT_BYTES','POSSIBLE_SHARED_SCENE_UNCERTAIN'}]
        status=max((c['technical_status'] for c in clips),key=RANK.get)
        cross=any(p['cross_split'] for p in related)
        warnings=sum(c['status']=='WARNING' for r in clips for c in r['checks'])
        score=100*RANK[status]+50*cross+warnings+10*(group['label']=='Fighting')
        priorities.append(dict(source=group['source'],label=group['label'],split=group['split'],priority_score=score,
            technical_status=status,cross_split_suspicion=cross,warning_count=warnings,
            human_label_decision='PENDING',incident_independence='PENDING'))
    priorities.sort(key=lambda p:(-p['priority_score'],p['source']))
    result=dict(method='Deterministic CPU OpenCV, sparse temporal/adjacent samples; no learned model or semantic understanding',
        opencv_version=cv2.__version__,parameters=PARAMETERS,canonical_split_signature=plan['canonical_split_signature'],
        manifest_sha256=plan['manifest_sha256'],
        technical_status=max((r['technical_status'] for r in records),key=RANK.get),clips=records,pairs=pairs,priorities=priorities,
        human_label_decision='PENDING',incident_independence='PENDING',training_executed=False,
        limitations='Thresholds are heuristic, not calibrated crime/incident classifiers. Motion includes camera/exposure/compression effects. Sparse samples can miss events; similarity can reflect backgrounds. No flags prove neither label correctness nor incident independence.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2))
    for name,rows in [('clips',records),('pairs',pairs),('priority_queue',priorities)]:
        if rows:
            with (output/(name+'.csv')).open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    return result
