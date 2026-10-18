"""Authorized weak-video-label IMAGE classifier; separate from reviewed training.

Existing selected source/split assignments are filtered, never regenerated.
Weak frame labels do not establish visible actions or validated crime labels.
"""
import csv
import hashlib
import io
import json
import uuid
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from scripts.data import CLASSES, file_hash, validate_manifest
from scripts.image_evidence_training import PREPROCESSING, preprocess


class WeakFrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Linear(64,4))

    def forward(self, images): return self.net(images)


def fingerprint(image):
    pixels = np.asarray(image)
    gray = np.asarray(image.convert('L').resize((9,8), Image.Resampling.BILINEAR))
    dhash = sum(int(v) << i for i,v in enumerate((gray[:,1:]>gray[:,:-1]).ravel()))
    small = np.asarray(image.resize((8,8),Image.Resampling.BILINEAR)).astype(np.int16)
    return hashlib.sha256(pixels.tobytes()).hexdigest(), dhash, small


def duplicate_reason(candidate, seen):
    digest, dhash, small = candidate
    for prior, _, split, source in seen:
        if digest == prior[0]: return dict(kind='exact_preprocessed_pixels', split=split, source=source)
        if (dhash ^ prior[1]).bit_count() <= 4 and np.abs(small-prior[2]).mean() < 6:
            return dict(kind='near_duplicate_dhash4_rgb_mae6', split=split, source=source)
    return None


def extract_frames(rows, output, frames_per_clip=8):
    validate_manifest(rows, require_splits=True, require_classes=False)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    seen, kept, removed = [], [], []
    # Explicit priority; later near-duplicate frames are removed, never reassigned.
    for row in sorted(rows, key=lambda r: (['train','val','test'].index(r['split']),r['source'],r['path'])):
        capture = cv2.VideoCapture(row['path'])
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or count < 1: raise ValueError('Unreadable selected video')
            for index in dict.fromkeys(np.linspace(0,count-1,min(count,frames_per_clip)).astype(int).tolist()):
                capture.set(cv2.CAP_PROP_POS_FRAMES,index); ok, frame = capture.read()
                if not ok: raise ValueError('Selected frame decoding failed')
                image = ImageOps.pad(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(128,128),
                                     method=Image.Resampling.BILINEAR,color=(0,0,0))
                signature = fingerprint(image)
                reason = duplicate_reason(signature, seen)
                record = dict(source=row['source'], split=row['split'], label=row['label'],
                              video_path=row['path'], video_sha256=file_hash(row['path']), frame_index=index,
                              pixel_sha256=signature[0], dhash=f'{signature[1]:016x}')
                if reason:
                    removed.append(dict(record,reason=reason)); continue
                name = f'{len(kept):04d}.png'; image.save(output/name)
                kept.append(dict(record,file=name,sha256=file_hash(output/name)))
                seen.append((signature,name,row['split'],row['source']))
        finally: capture.release()
    for split in ['train','val','test']:
        if not any(r['split']==split for r in kept): raise ValueError('No nonduplicate frames in '+split)
    if {r['label'] for r in kept if r['split']=='train'} != set(CLASSES):
        raise ValueError('Four training classes required after deduplication')
    return kept, removed


def metrics(logits, targets):
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
    predicted = logits.argmax(1).cpu().numpy(); targets = np.asarray(targets)
    precision, recall, f1, support = precision_recall_fscore_support(targets,predicted,labels=list(range(4)),zero_division=0)
    return dict(accuracy=float(accuracy_score(targets,predicted)),
        macro_f1=float(f1_score(targets,predicted,labels=list(range(4)),average='macro',zero_division=0)),
        confusion_matrix=confusion_matrix(targets,predicted,labels=list(range(4))).tolist(),
        per_class={label:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,label in enumerate(CLASSES)},
        unsupported_classes=[CLASSES[i] for i in range(4) if support[i]==0], predictions=predicted.tolist())


def train_frames(frames, folder, output, epochs=15, device='cuda'):
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model=WeakFrameCNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    tensors=[]
    for row in frames:
        data=(Path(folder)/row['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Frame changed after verification')
        tensors.append(preprocess(data))
    images=torch.stack(tensors)
    targets=torch.tensor([CLASSES.index(r['label']) for r in frames])
    subsets={s:torch.tensor([i for i,r in enumerate(frames) if r['split']==s]) for s in ['train','val','test']}
    config=dict(architecture='WeakFrameCNN-v1',classes=CLASSES,preprocessing=PREPROCESSING,
        epochs=epochs,seed=42,batch_size=8,learning_rate=.001,random_initialization=True,
        pretrained=False,validated=False,weak_labels=True,incident_independence='UNVERIFIED',
        production_source_mapping_reviewed=False,production_run_prototype_training=False)
    history=[]; best=float('inf')
    for number in range(epochs):
        model.train(); total=0.
        indices=subsets['train'][torch.randperm(len(subsets['train']))]
        for batch in indices.split(8):
            optimizer.zero_grad(); logits=model(images[batch].to(device))
            loss=nn.functional.cross_entropy(logits,targets[batch].to(device))
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); optimizer.step(); total+=loss.item()*len(batch)
        model.eval()
        with torch.inference_mode():
            val=model(images[subsets['val']].to(device))
            val_loss=nn.functional.cross_entropy(val,targets[subsets['val']].to(device)).item()
        history.append(dict(epoch=number+1,train_loss=total/len(indices),val_loss=val_loss))
        print(history[-1],flush=True)
        if val_loss<best:
            best=val_loss
            torch.save(dict(state_dict=model.state_dict(),config=config,epoch=number+1),Path(output)/'experimental_image_cnn.pth')
    saved=torch.load(Path(output)/'experimental_image_cnn.pth',map_location=device,weights_only=True)
    model.load_state_dict(saved['state_dict']); model.eval()
    with torch.inference_mode(): logits=model(images[subsets['test']].to(device)).cpu()
    test_rows=[frames[i] for i in subsets['test'].tolist()]
    frame_metrics=metrics(logits,targets[subsets['test']].tolist())
    grouped={source:[i for i,r in enumerate(test_rows) if r['source']==source] for source in sorted({r['source'] for r in test_rows})}
    source_logits=torch.stack([logits[v].mean(0) for v in grouped.values()])
    source_targets=[CLASSES.index(test_rows[v[0]]['label']) for v in grouped.values()]
    result=dict(frame=frame_metrics,source=metrics(source_logits,source_targets),source_order=list(grouped),
        test_loss=nn.functional.cross_entropy(logits,targets[subsets['test']]).item(),
        selected_epoch=saved['epoch'],device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',
        warning='UNVALIDATED weak video-derived labels; source-disjoint is not verified incident independence')
    for name,value in [('config',config),('history',history),('test_metrics',result)]:
        (Path(output)/(name+'.json')).write_text(json.dumps(value,indent=2))
    return result


def run(review, root, runs, *, authorized=False):
    if authorized is not True: raise ValueError('Explicit UNVALIDATED weak-label authorization required')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0): raise RuntimeError('T4 required')
    from scripts.review_paths import ensure_review_dataset, _BINDINGS, recovery_evidence
    from scripts.review_launcher import resume_decisions
    from scripts.review_dashboard import consolidate_review
    from scripts.experimental_smoke import eligible_rows
    plan=ensure_review_dataset(review,root)
    report=consolidate_review(review); answers,decisions=resume_decisions(review,report)
    original_hash=file_hash(decisions); original_records=recovery_evidence(review)
    binding=_BINDINGS.get(str(Path(review).resolve()))
    validation=Path(binding['runtime_validation']) if binding else Path(plan['validation_path'])
    saved=json.loads(validation.read_text())
    with open(saved['split_manifest']) as reader: rows=list(csv.DictReader(reader))
    selected,excluded=eligible_rows(rows,answers,allow_uncertain_labels=True)
    output=Path(runs)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    frames,removed=extract_frames(selected,output/'frames')
    provenance=dict(status='UNVALIDATED EXPERIMENTAL',dataset='mateohervas/dcsass-dataset/versions/1',
        dataset_verification=original_records,decisions_sha256=original_hash,decisions_path=decisions,
        saved_answers=answers,excluded_sources=excluded,canonical_split_signature=plan['canonical_split_signature'],
        split_policy='Filtered existing selected source/split assignments; no new canonical split',
        label_policy='Original weak video categories; Confirmed/Uncertain and watched only; Rejected/Pending excluded',
        human_decisions_modified=False,incident_independence='UNVERIFIED; human answers preserved',
        dedup_policy='Exact preprocessed RGB pixels or dHash Hamming<=4 AND 8x8 RGB MAE<6; global train/val/test audit',
        residual_duplicate_risk='Heuristic does not prove absence of visual/incident similarity',
        frames=frames,removed_frames=removed,
        counts={s:dict(frames=sum(r['split']==s for r in frames),
            sources=len({r['source'] for r in frames if r['split']==s}),
            classes=dict(Counter(r['label'] for r in frames if r['split']==s))) for s in ['train','val','test']},
        implementation_sha256=file_hash(__file__),pretrained=False,external_ai_apis=False)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with (output/'execution.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
        result=train_frames(frames,output/'frames',output,epochs=15)
    if file_hash(decisions)!=original_hash or recovery_evidence(review)!=original_records:
        raise ValueError('Original records changed during experiment')
    completion=dict(status='completed',output=str(output),counts=provenance['counts'],
        checkpoint_sha256=file_hash(output/'experimental_image_cnn.pth'),metrics=result,
        removed_frames=len(removed),validated=False,human_files_unchanged=True)
    (output/'completion.json').write_text(json.dumps(completion,indent=2))
    # Read back the persisted checkpoint and records before reporting success.
    torch.load(output/'experimental_image_cnn.pth',map_location='cpu',weights_only=True)
    assert json.loads((output/'test_metrics.json').read_text())==result
    print(json.dumps(completion,indent=2),flush=True)
    return completion
