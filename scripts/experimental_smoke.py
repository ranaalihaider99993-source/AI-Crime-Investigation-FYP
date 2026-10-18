"""Explicitly authorized technical experiment, separate from reviewed training gates."""
import csv
import json
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from scripts.classifier import CrimeCNN, VideoDataset, epoch
from scripts.common import seed_all, experiment_metadata
from scripts.data import CLASSES, validate_manifest, file_hash, write_csv
from scripts.review_launcher import resume_decisions
from scripts.review_dashboard import consolidate_review
from scripts.review_paths import ensure_review_dataset


class ExperimentalDataset(VideoDataset):
    """Allow documented missing validation classes after excluding uncertain labels."""
    def __init__(self, manifest, split):
        with open(manifest) as reader:
            rows = list(csv.DictReader(reader))
        validate_manifest(rows, require_splits=True, require_classes=False)
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError('No eligible clips for '+split)
        self.size, self.frames = 64, 3


def eligible_rows(rows, answers, *, allow_uncertain_labels=False):
    included, excluded = [], []
    for row in rows:
        answer = answers.get(row['source'], {})
        accepted = {'Confirmed','Uncertain'} if allow_uncertain_labels else {'Confirmed'}
        if answer.get('label_visible') not in accepted or answer.get('videos_watched') != 'Confirmed':
            excluded.append(dict(source=row['source'], split=row['split'],
                                 reason='Requires saved Confirmed label AND all videos watched',
                                 saved_answer=answer))
        else:
            included.append(row)
    return included, excluded


def run_experimental_smoke(review, root, runs, *, authorized=False, allow_uncertain_labels=False):
    if authorized is not True:
        raise ValueError('Explicit experimental authorization required; production gates stay False')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('An authorized T4 GPU runtime is required; no CPU fallback')
    plan = ensure_review_dataset(review, root)
    report = consolidate_review(review)
    answers, decisions_path = resume_decisions(review, report)
    # Recover only the selected clip paths; canonical source/split membership is fixed.
    from scripts.review_paths import _BINDINGS
    binding = _BINDINGS.get(str(Path(review).resolve()))
    validation = Path(binding['runtime_validation']) if binding else Path(plan['validation_path'])
    saved_validation = json.loads(validation.read_text())
    with open(saved_validation['split_manifest']) as reader:
        rows = list(csv.DictReader(reader))
    included, excluded = eligible_rows(rows, answers, allow_uncertain_labels=allow_uncertain_labels)
    validate_manifest(included, require_splits=True, require_classes=False)
    if {r['label'] for r in included if r['split']=='train'} != set(CLASSES):
        raise RuntimeError('Human decision required: Confirmed/watched training examples missing for one or more classes')
    if any(not any(r['split']==split for r in included) for split in ['val','test']):
        raise RuntimeError('Human decision required: no eligible validation or test clips')
    output = Path(runs)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = dict(seed=42, epochs=2, batch_size=4, image_size=64, frames_per_clip=3, learning_rate=.001)
    metadata = dict(experiment='EXPERIMENTAL CNN EXECUTION SMOKE; NOT VALIDATED CRIME PERFORMANCE',
                    production_source_mapping_reviewed=False, production_run_prototype_training=False,
                    canonical_review_split_signature=plan['canonical_split_signature'],
                    label_policy=('UNVALIDATED original dataset labels; explicitly authorized inclusion of Uncertain; exclude Pending/Rejected'
                                  if allow_uncertain_labels else 'Saved human Confirmed label and videos_watched only; exclude Pending/Uncertain/Rejected'),
                    allow_uncertain_labels=allow_uncertain_labels,
                    implementation_sha256=file_hash(__file__),
                    incident_independence='UNVERIFIED; all results exploratory and potentially incident-correlated',
                    human_approval_granted=False, validated_performance=False,
                    pretrained_weights=False, external_ai_apis=False, config=config,
                    decisions_path=decisions_path, decisions_sha256=file_hash(decisions_path),
                    included=[dict(row, saved_answer=answers[row['source']]) for row in included], excluded=excluded,
                    split_policy='Filter existing selection only; never replace or reassign sources',
                    split_counts={split:{label:sum(r['split']==split and r['label']==label for r in included)
                                         for label in CLASSES} for split in ['train','val','test']},
                    device=torch.cuda.get_device_name(0))
    manifest = output/'experimental_subset.csv'
    write_csv(manifest, included, list(included[0]))
    metadata['subset_sha256'] = file_hash(manifest)
    metadata['environment'] = experiment_metadata(manifest, 42)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2))
    try:
        with (output/'execution.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
            seed_all(42)
            model = CrimeCNN().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            loaders = {split:DataLoader(ExperimentalDataset(manifest,split),batch_size=4,shuffle=split=='train',num_workers=0) for split in ['train','val','test']}
            history, best = [], float('inf')
            for number in range(2):
                tr = epoch(model,loaders['train'],'cuda',optimizer)
                val = epoch(model,loaders['val'],'cuda')
                history.append(dict(epoch=number+1,train=tr,val=val))
                print(json.dumps(history[-1]),flush=True)
                if val['loss'] < best:
                    best = val['loss']
                    torch.save(dict(state_dict=model.state_dict(),config=config,classes=CLASSES,metadata=metadata,epoch=number+1),output/'experimental_cnn.pth')
                (output/'history.json').write_text(json.dumps(history,indent=2))
            checkpoint = torch.load(output/'experimental_cnn.pth',map_location='cuda',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            result = epoch(model,loaders['test'],'cuda')
            (output/'test_metrics.json').write_text(json.dumps(result,indent=2))
            print('Exploratory test metrics; unverified incident independence:',json.dumps(result),flush=True)
        completion = dict(status='completed', run_dir=str(output),checkpoint=str(output/'experimental_cnn.pth'),
                          checkpoint_sha256=file_hash(output/'experimental_cnn.pth'),metrics=result,
                          included=len(included),excluded=excluded,device=metadata['device'],
                          validated_performance=False,production_flags=False)
        (output/'completion.json').write_text(json.dumps(completion,indent=2))
        print(json.dumps(completion,indent=2),flush=True)
        return completion
    except Exception:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise
