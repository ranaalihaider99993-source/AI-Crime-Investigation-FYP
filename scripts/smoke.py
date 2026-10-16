"""Prepare an unapproved real-data candidate; run only an explicitly reviewed smoke."""
import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from scripts.data import (CLASSES, canonical_clip, class_summary, file_hash,
                          source_id, split_manifest, split_signature,
                          validate_manifest, write_csv)
from scripts.classifier import VideoDataset, train, evaluate
from scripts.dataset_paths import resolve_inspection_paths


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def smoke_config(config):
    require(config['epochs'] == 2 and config['sources_per_class'] == 3,
            'This launcher permits only 2 epochs and 3 sources per class; full training is not authorized')
    require(1 <= config['batch_size'] <= 4 and 16 <= config['image_size'] <= 64
            and 1 <= config['frames_per_clip'] <= 3 and config['patience'] >= 2
            and config['learning_rate'] > 0,
            'Invalid bounded smoke configuration')


def artifact_hashes(inspection_dir):
    return {name: file_hash(Path(inspection_dir)/name)
            for name in ['manifest.csv', 'inspection.json', 'excluded_duplicates.csv']}


def _persist_report(report, output):
    (output/'validation.json').write_text(json.dumps(report, indent=2))
    lines = ['# CNN smoke preparation', '', f"Automatic checks: **{report['automatic_status']}**",
             '', '**Human mapping review: PENDING. Training has NOT run.**', '',
             f"GPU detected during preparation: {report['gpu_available']} ({report['gpu_name']}). The launcher requires CUDA again before training.", '',
             '| Check | Status | Detail |', '| --- | --- | --- |']
    for check in report['checks']:
        detail = check['detail'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    lines += ['', '## Available readable data', '', '| Class | Clips | Original sources |', '| --- | ---: | ---: |']
    for label, counts in report.get('available_readable', {}).items():
        lines.append(f"| {label} | {counts['clips']} | {counts['sources']} |")
    lines += ['', '## Candidate split', '', '| Class | Train sources/clips | Validation sources/clips | Test sources/clips |', '| --- | --- | --- | --- |']
    for label in CLASSES:
        counts = report.get('split_counts', {})
        if counts:
            cells = [f"{counts[s][label]['sources']}/{counts[s][label]['clips']}" for s in ['train','val','test']]
            lines.append('| ' + ' | '.join([label] + cells) + ' |')
    lines += ['', '## Warnings and required human review', ''] + [f'- {w}' for w in report['warnings']]
    lines += ['', 'Review source_review.csv and review_sheet.jpg. Check original-source naming,',
              'category assignment and related/duplicate incidents across different source IDs.',
              'A few preview frames cannot establish clip-level labels or incident independence.',
              'Only set SOURCE_MAPPING_REVIEWED after reviewing the mapping; automatic PASS is not approval.',
              'Full training is outside this launcher. No real metrics or checkpoint exist from preparation alone.']
    (output/'validation.md').write_text('\n'.join(lines)+'\n')


def prepare_smoke(data_root, inspection_dir, output, config):
    """Reuse notebook 01 artifacts, write checks even on failure, never approve/train."""
    root, inspection_dir, output = Path(data_root).absolute(), Path(inspection_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(automatic_status='FAIL', human_mapping_review='PENDING', training_executed=False,
                  config=dict(config), data_root=str(root), inspection_dir=str(inspection_dir),
                  checks=[], warnings=[
                      'Class/source consistency is validated against naming rules, not verified ground truth.',
                      'Different IDs may depict the same incident. Human review is still required.',
                      'A 12-clip smoke run tests execution only: each class has ONE validation and ONE test source.',
                      'Duplicate audit hashes were verified by notebook 01; this report checks saved records and current paths/sizes, without rehashing the whole dataset.'],
                  gpu_available=torch.cuda.is_available(),
                  gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    def check(name, action):
        try:
            detail = action()
            report['checks'].append(dict(name=name, status='PASS', detail=str(detail)))
            return True
        except Exception as exc:
            report['checks'].append(dict(name=name, status='FAIL', detail=f'{type(exc).__name__}: {exc}'))
            return False

    if not check('bounded two-epoch configuration', lambda: smoke_config(config) or '2 epochs, 3 sources/class'):
        _persist_report(report, output); return report
    rows, inspection, duplicates = [], {}, []
    def load():
        nonlocal rows, inspection, duplicates
        rows = read_csv(inspection_dir/'manifest.csv')
        inspection = json.loads((inspection_dir/'inspection.json').read_text())
        duplicates = read_csv(inspection_dir/'excluded_duplicates.csv')
        require(inspection.get('schema_version') == 2 and inspection.get('mapped_classes') == CLASSES,
                'Run updated notebook 01: schema v2 and the four-class mapping are required')
        report['artifact_hashes'] = artifact_hashes(inspection_dir)
        return f'{len(rows)} manifest rows; {len(duplicates)} excluded copies'
    if not check('load existing inspection artifacts', load):
        _persist_report(report, output); return report

    def resolve_paths():
        nonlocal root, inspection_dir, rows, inspection, duplicates
        report['original_inspection_dir'] = str(inspection_dir)
        report['original_artifact_hashes'] = artifact_hashes(inspection_dir)
        root, inspection_dir, resolution = resolve_inspection_paths(root, inspection_dir, output/'resolved_inspection')
        rows = read_csv(inspection_dir/'manifest.csv')
        inspection = json.loads((inspection_dir/'inspection.json').read_text())
        duplicates = read_csv(inspection_dir/'excluded_duplicates.csv')
        report.update(data_root=str(root), inspection_dir=str(inspection_dir), path_resolution=resolution,
                      artifact_hashes=artifact_hashes(inspection_dir))
        return f"Verified root: {root}; relocated copies: {resolution['relocated_copies']}; SHA256 reverified: {resolution['reverified_sha256_files']}"
    if not check('verified KaggleHub root and saved-path resolution', resolve_paths):
        _persist_report(report, output); return report

    def manifest_check():
        require(bool(rows), 'Empty manifest')
        validate_manifest(rows)
        for row in rows:
            path = Path(row['path']).resolve()
            require(path.is_relative_to(root), f'Path outside dataset root: {path}')
            require(canonical_clip(path, root) == row['canonical_clip'], 'Physical path/canonical clip mismatch; rerun notebook 01 if cache paths changed')
            require(source_id(path, root) == row['source'], 'Physical path/source ID mismatch')
        require(inspection['unique_video_clips'] == len(rows), 'Inspection clip count differs from manifest')
        require(inspection['unique_original_sources'] == len({r['source'] for r in rows}), 'Inspection source count differs')
        require(inspection['mapped_counts'] == class_summary(rows), 'Inspection class counts differ')
        require(inspection['unmapped_clips'] == sum(not r['label'] for r in rows), 'Unmapped counts differ')
        readable = [r for r in rows if r['readable'].lower() == 'true']
        require(inspection['readable'] == len(readable) and inspection['unreadable'] == len(rows)-len(readable), 'Readability counts differ')
        report['available_readable'] = class_summary(readable)
        report['unmapped_clips'] = inspection['unmapped_clips']
        return 'Canonical clips unique; existing physical paths bound to canonical IDs; four-class/source counts consistent'
    manifest_ok = check('canonical manifest, files and class counts', manifest_check)

    def duplicate_check():
        require(inspection['excluded_duplicate_copies'] == len(duplicates), 'Duplicate report count differs')
        require(inspection['raw_video_files'] == len(rows)+len(duplicates), 'Raw/unique/excluded accounting differs')
        kept = {r['canonical_clip']: Path(r['path']).resolve() for r in rows}
        kept_paths = set(kept.values())
        seen, hashes = set(), {}
        for row in duplicates:
            clip = row['canonical_clip']
            keep, excluded = Path(row['kept_path']).resolve(), Path(row['excluded_path']).resolve()
            require(clip in kept and keep == kept[clip], 'Duplicate record kept path differs from manifest')
            require(excluded.is_relative_to(root) and excluded.is_file(), f'Missing/outside excluded copy: {excluded}')
            require(keep.is_file() and keep != excluded and excluded not in seen and excluded not in kept_paths, 'Duplicate record paths are not distinct')
            require(canonical_clip(excluded, root) == clip, 'Excluded copy has a different canonical clip')
            require(keep.stat().st_size == excluded.stat().st_size, 'Duplicate copy size changed: rerun inspection')
            require(re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None, 'Missing/invalid recorded SHA256')
            require(row['reason'] == 'same canonical clip and SHA256', 'Unknown duplicate-audit reason')
            require(clip not in hashes or hashes[clip] == row['sha256'], 'Conflicting recorded hashes')
            hashes[clip] = row['sha256']; seen.add(excluded)
        return f'{len(duplicates)} consistent exclusion records (recorded hashes; bytes not rehashed)'
    duplicates_ok = check('saved duplicate audit consistency', duplicate_check) if manifest_ok else False

    def sources_check():
        readable = [r for r in rows if r['label'] and r['readable'].lower() == 'true']
        summary = class_summary(readable)
        require(all(v['sources'] >= 3 for v in summary.values()), f'Need 3 independent readable sources per class: {summary}')
        unknown = sorted({r['source'] for r in readable if not re.fullmatch(r'(burglary|stealing|fighting|vandalism)\d+_x264', r['source'])})
        report['source_names_requiring_investigation'] = unknown
        require(not unknown, f'Unrecognized source names need investigation before training: {unknown[:10]}')
        fighting = summary['Fighting']['sources']
        report['warnings'].append(f'Actual readable Fighting source count: {fighting}; only three are selected for this smoke, split 1/1/1. This cannot establish reliable Fighting performance.')
        return f'Recognized naming rules; at least 3 readable sources/class; Fighting={fighting}'
    sources_ok = check('source naming and independent-source sufficiency', sources_check) if manifest_ok else False
    if manifest_ok and duplicates_ok and sources_ok:
        def candidate():
            split_path = output/'candidate_splits.csv'
            selected = split_manifest(inspection_dir/'manifest.csv', split_path, seed=config['seed'], per_class=3)
            split_report = json.loads(split_path.with_suffix('.report.json').read_text())
            require(len(selected) == 12 and all(not v for v in split_report['source_overlap'].values()), 'Candidate is not a disjoint 12-source split')
            for split in ['train','val','test']:
                require(all(v == {'clips':1,'sources':1} for v in split_report['split_counts'][split].values()), 'Unexpected smoke class counts')
            report.update(split_manifest=str(split_path), split_signature=split_signature(split_path),
                          split_file_sha256=file_hash(split_path), split_counts=split_report['split_counts'],
                          source_overlap=split_report['source_overlap'])
            # One review row per mapped original source, with selected sources marked.
            selected_sources = {r['source']: r['split'] for r in selected}
            by_source = {}
            for r in sorted(rows, key=lambda r:r['canonical_clip']):
                if r['label']:
                    by_source.setdefault(r['source'], r)
            review = [dict(source=key, label=r['label'], selected_split=selected_sources.get(key,''),
                           representative_clip=r['canonical_clip'], path=r['path']) for key,r in sorted(by_source.items())]
            write_csv(output/'source_review.csv', review, ['source','label','selected_split','representative_clip','path'])
            return '12 source-disjoint candidate clips; all class/split counts 1; source_review.csv written'
        if check('candidate source split and overlap', candidate):
            def frames():
                # Use the same sampled frames/decoding as training. No optimizer or model runs.
                tiles, selected_hashes, hash_owners = [], {}, {}
                for split in ['train','val','test']:
                    dataset = VideoDataset(report['split_manifest'], split, config['image_size'], config['frames_per_clip'])
                    for i, row in enumerate(dataset.rows):
                        x, _ = dataset[i]
                        digest = file_hash(row['path'])
                        require(digest not in hash_owners or hash_owners[digest] == split,
                                'Identical video bytes across different source IDs/splits; investigate source grouping')
                        hash_owners[digest] = split
                        selected_hashes[row['canonical_clip']] = digest
                        frames = [cv2.cvtColor((frame.permute(1,2,0).numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR) for frame in x]
                        tile = np.zeros((120,480,3), dtype=np.uint8)
                        for j, image in enumerate(frames):
                            tile[40:120,j*160:(j+1)*160] = cv2.resize(image,(160,80))
                        cv2.putText(tile, f"{split} {row['label']} {row['source']}", (5,16), cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
                        cv2.putText(tile, Path(row['canonical_clip']).name[:65], (5,32), cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                        tiles.append(tile)
                report['selected_clip_sha256'] = selected_hashes
                require(cv2.imwrite(str(output/'review_sheet.jpg'), np.vstack(tiles)), 'Cannot save review sheet')
                return 'Selected frames decoded; no byte-identical clips across splits; review_sheet.jpg written; visual labels remain unverified'
            check('selected-frame decode, exact-byte leakage and review sheet', frames)
    report['automatic_status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['checks']) and len(report['checks']) == 8 else 'FAIL'
    _persist_report(report, output)
    return report


def run_reviewed_smoke(report_path, runs_dir, config, source_mapping_reviewed=False,
                       run_prototype_training=False, require_gpu=True):
    """Both user gates + unchanged passing report + GPU required by Colab entrypoint."""
    if not run_prototype_training:
        return None
    require(source_mapping_reviewed, 'Human source-mapping review is required; automatic PASS is not approval')
    smoke_config(config)
    report = json.loads(Path(report_path).read_text())
    require(report['automatic_status'] == 'PASS', 'Automatic validation failed; fix it before training')
    require(report['config'] == config, 'Configuration changed; rerun preparation and review')
    require(report['original_artifact_hashes'] == artifact_hashes(report['original_inspection_dir']), 'Original inspection artifacts changed; rerun preparation and review')
    require(report['artifact_hashes'] == artifact_hashes(report['inspection_dir']), 'Inspection artifacts changed; rerun preparation and review')
    require(report['split_signature'] == split_signature(report['split_manifest']) and
            report['split_file_sha256'] == file_hash(report['split_manifest']), 'Candidate split changed; rerun preparation and review')
    selected = read_csv(report['split_manifest'])
    require(report['selected_clip_sha256'] == {r['canonical_clip']: file_hash(r['path']) for r in selected},
            'Selected video contents changed; rerun preparation and human review')
    if require_gpu:
        require(torch.cuda.is_available(), 'No CUDA GPU allocated: select T4 GPU and reconnect; no CPU training was substituted')
    runs_dir = Path(runs_dir)
    run_dir = runs_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'review_attestation.json').write_text(json.dumps(dict(source_mapping_reviewed=True,
        reviewed_report_sha256=file_hash(report_path), canonical_split_signature=report['split_signature']), indent=2))
    # This call creates fresh random weights. No checkpoint is loaded for training.
    history = train(report['split_manifest'], run_dir, config)
    metrics = evaluate(run_dir/'split_manifest.csv', run_dir/'crime_classifier.pth', run_dir/'test_metrics.json')
    saved = torch.load(run_dir/'crime_classifier.pth', map_location='cpu', weights_only=True)
    result = dict(status='completed', run_dir=str(run_dir.resolve()), checkpoint=str((run_dir/'crime_classifier.pth').resolve()),
                  metrics=metrics, actual_training_device=saved['metadata']['training_device'], gpu_name=saved['metadata']['gpu_name'],
                  epochs_executed=len(history), class_order=CLASSES,
                  limitation='12 clips / one held-out source per class: execution smoke only, not reliable accuracy')
    (run_dir/'completion.json').write_text(json.dumps(result, indent=2))
    # Updated only after this invocation trains AND evaluates successfully.
    (runs_dir/'latest_completed.json').write_text(json.dumps(result, indent=2))
    return result
