"""Bounded source context for an existing candidate; never split, approve, or train."""
import csv
import html
import json
import re
import uuid
from pathlib import Path
from scripts.data import canonical_clip, file_hash, source_id, validate_manifest
from scripts.original_review import original_review_plan, original_players, MAX_CLIP_BYTES, INLINE_TOTAL_BYTES
from scripts.smoke import artifact_hashes, read_csv, require


def segment_number(filename):
    match = re.search(r'(?:^|[_-])(\d+)$', Path(filename).stem)
    return int(match[1]) if match else None


def context_plan(validation_path):
    selected = original_review_plan(validation_path)
    validation = json.loads(Path(validation_path).read_text())
    inspection = Path(validation['inspection_dir'])
    require(artifact_hashes(inspection) == validation['artifact_hashes'], 'Inspection artifacts changed; context review stopped')
    rows = read_csv(inspection/'manifest.csv')
    validate_manifest(rows)
    root = Path(selected['dataset_root'])
    hashes = dict(json.loads((inspection/'inspection.json').read_text()).get('clip_sha256', {}))
    for record in read_csv(inspection/'excluded_duplicates.csv'):
        key, digest = record['canonical_clip'], record['sha256']
        require(key not in hashes or hashes[key] == digest, 'Conflicting inspection hashes')
        hashes[key] = digest
    groups = []
    for clip in selected['clips']:
        candidates = [r for r in rows if r['source'] == clip['source']]
        require(any(r['canonical_clip'] == clip['canonical_clip'] for r in candidates), 'Selected clip missing from inspection')
        numbers = [segment_number(Path(r['path']).name) for r in candidates]
        current = segment_number(clip['filename'])
        warnings = []
        neighbors = []
        if current is None or any(n is None for n in numbers) or len(set(numbers)) != len(numbers):
            warnings.append('Chronological ordering is unknown or ambiguous; no neighboring clips guessed.')
        else:
            ordered = sorted(zip(numbers, candidates), key=lambda item: item[0])
            # Position must be taken from numeric order, not manifest order.
            position = next(i for i, (n, r) in enumerate(ordered) if r['canonical_clip'] == clip['canonical_clip'])
            for index, role in [(position-1, 'preceding context'), (position+1, 'following context')]:
                if not 0 <= index < len(ordered):
                    warnings.append(f'No {role} available in this source.')
                    continue
                number, row = ordered[index]
                path = Path(row['path']).resolve()
                require(path.is_relative_to(root), 'Context path escapes verified dataset root')
                require(canonical_clip(path, root) == row['canonical_clip'] and source_id(path, root) == clip['source'], 'Context path identity differs')
                require(row['label'] == clip['label'], 'Context class mapping differs')
                size = path.stat().st_size
                require(0 < size <= MAX_CLIP_BYTES, 'Context clip empty or exceeds size bound')
                if abs(number-current) != 1:
                    warnings.append(f'Numbering gap before {path.name}; nearest available segment only.')
                digest = file_hash(path)
                require(hashes.get(row['canonical_clip']) == digest, 'Context content differs from inspection or lacks saved SHA256')
                neighbors.append(dict(clip, path=str(path), filename=path.name, canonical_clip=row['canonical_clip'],
                                      size_bytes=size, sha256=digest, role=role, context_only=True))
        center = dict(clip, role='SELECTED experiment clip', context_only=False)
        videos = [c for c in neighbors if c['role']=='preceding context'] + [center] + [c for c in neighbors if c['role']=='following context']
        groups.append(dict(source=clip['source'], label=clip['label'], split=clip['split'], selected=clip['canonical_clip'], videos=videos, warnings=warnings))
    total = sum(c['size_bytes'] for g in groups for c in g['videos'])
    require(total <= INLINE_TOTAL_BYTES, 'Context exceeds 128 MiB inline bound; no unbounded video embedding')
    return dict(canonical_split_signature=selected['canonical_split_signature'], manifest_sha256=selected['manifest_sha256'],
                validation_path=selected['validation_path'], dataset_root=str(root), groups=groups, total_video_bytes=total,
                human_approval='PENDING', semantic_labels='UNVERIFIED', incident_independence='UNVERIFIED', training_executed=False,
                ordering='Numeric filename order only; timestamps and incident continuity are not verified.')


def context_players(plan):
    """Original byte playback, rechecking hashes immediately before embedding."""
    for group in plan['groups']:
        yield '<h2>'+html.escape(group['source'])+'</h2><p>'+html.escape(' '.join(group['warnings']))+'</p>'
        for clip in group['videos']:
            require(Path(clip['path']).resolve().is_relative_to(Path(plan['dataset_root'])), 'Playback path escapes verified dataset root')
            yield '<h4>'+html.escape(clip['role'])+'</h4>'
            yield from original_players(dict(clips=[clip], total_video_bytes=clip['size_bytes']))


def save_context_review(plan, output):
    """Metadata only; unique folder preserves any previously filled checklist."""
    directory = Path(output)/('review-'+uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'context_metadata.json').write_text(json.dumps(plan, indent=2))
    fields = ['source','label','split','selected_clip','context_clips','warnings','watched_all','selected_label_visible',
              'label_visible_only_in_context','possible_related_sources','incident_independence','reviewer','notes','review_notes']
    with (directory/'human_checklist.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in plan['groups']:
            writer.writerow(dict(source=group['source'], label=group['label'], split=group['split'], selected_clip=group['selected'],
                                 context_clips='; '.join(c['canonical_clip'] for c in group['videos'] if c['context_only']),
                                 warnings='; '.join(group['warnings']), watched_all='PENDING', selected_label_visible='PENDING',
                                 label_visible_only_in_context='PENDING', possible_related_sources='PENDING', incident_independence='PENDING'))
    (directory/'README.md').write_text(
        '# Source-context human review\n\nAll 12 selected clips and split assignments are unchanged. Neighbors are context only, never additional training samples.\n\n'
        +plan['ordering']+'\n\nWatch selected and neighboring videos fully. Mark whether the selected clip itself shows its label, whether evidence appears only in context, and any unclear/absent labels. Compare all sources for related incidents; record source IDs and uncertainty in the checklist. Different source IDs do not establish incident independence.\n\n'
        'All semantic decisions remain PENDING. Both training flags must remain False. No training was executed. This folder contains metadata, not copied videos.\n')
    return directory
