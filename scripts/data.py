"""Deterministic DCSASS mirror deduplication and canonical source-group splits."""
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict
import cv2

# Keep the currently reviewed FOUR-class prototype; Normal is not mapped yet.
CLASSES = ['Burglary', 'Stealing', 'Fighting', 'Vandalism']
VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
FIELDS = ['path', 'canonical_clip', 'label', 'source', 'readable']
SPLITS = ('train', 'val', 'test')


def canonical_clip(path, root):
    """Strip only leading DCSASS Dataset wrappers, regardless of case/separators.

    Do not strip interior category/source components or reduce a clip to its basename.
    Windows-origin identifiers are case-insensitive; physical paths remain unchanged.
    """
    parts = list(Path(path).relative_to(root).parts)
    while parts and re.sub(r'[\s_-]+', ' ', parts[0].casefold()).strip() == 'dcsass dataset':
        parts.pop(0)
    if not parts:
        raise ValueError('Empty canonical clip')
    return '/'.join(p.casefold() for p in parts)


def category(path):
    for part in reversed(Path(path).parts):
        for label in CLASSES:
            if re.match(re.escape(label) + r'(?:[\d_\s.-]|$)', part, re.I):
                return label
    return None


def source_id(path, root):
    rel = Path(canonical_clip(path, root))
    # Original-video directories contain numbered segment files. The original
    # basename is global, so conflicting category assignments cannot hide leakage.
    for part in reversed(rel.parts[:-1]):
        if part.endswith('.mp4'):
            return part[:-4]
    match = re.match(r'(.+?_x264)(?:[_-]\d+)?$', rel.stem, re.I)
    return match.group(1) if match else rel.stem


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(rows, require_splits=False, require_classes=False):
    """Fail explicitly (also with python -O) on stale or unsafe manifests."""
    clips, paths, owners, source_labels = set(), set(), {}, {}
    for row in rows:
        if not row.get('canonical_clip') or not row.get('source'):
            raise ValueError('Legacy/noncanonical manifest: regenerate inspection with notebook 01')
        if not set(FIELDS).issubset(row) or str(row['readable']).lower() not in {'true', 'false'}:
            raise ValueError('Missing manifest fields or invalid readability status')
        clip = row['canonical_clip']
        if clip != canonical_clip(Path(clip), Path('.')) or '\\' in clip or clip.startswith('/') or '..' in Path(clip).parts:
            raise ValueError(f'Invalid canonical clip: {clip}')
        if clip in clips:
            raise ValueError(f'Duplicate canonical clip: {clip}')
        clips.add(clip)
        path = Path(row['path'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing video file: {path}')
        physical = str(path.resolve())
        if physical in paths:
            raise ValueError(f'Duplicate physical clip: {path}')
        paths.add(physical)
        source = row['source']
        if source != source_id(Path(clip), Path('.')):
            raise ValueError('Source ID does not match canonical original video')
        if row['label'] != (category(clip) or ''):
            raise ValueError('Class label does not match the reviewed four-class mapping')
        if source in source_labels and source_labels[source] != row['label']:
            raise ValueError(f'Conflicting source labels: {source}')
        source_labels[source] = row['label']
        if require_splits:
            split = row.get('split')
            if split not in SPLITS:
                raise ValueError(f'Invalid split: {split}')
            if row['label'] not in CLASSES or str(row['readable']).lower() != 'true':
                raise ValueError('Training split contains unmapped or unreadable clip')
            if source in owners and owners[source] != split:
                raise ValueError(f'Source leakage in split manifest: {source}')
            owners[source] = split
    if require_classes:
        for split in SPLITS:
            counts = Counter(r['label'] for r in rows if r.get('split') == split)
            sources = {label: len({r['source'] for r in rows if r.get('split') == split and r['label'] == label}) for label in CLASSES}
            if set(counts) != set(CLASSES) or len(set(counts.values())) != 1 or len(set(sources.values())) != 1:
                raise ValueError(f'Missing or unbalanced class counts in {split}: {dict(counts)}')


def split_signature(manifest):
    """Bind checkpoints to canonical clip/label/source/split membership, not cache paths."""
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    validate_manifest(rows, require_splits=True, require_classes=True)
    membership = sorted((r['canonical_clip'], r['source'], r['label'], r['split']) for r in rows)
    return hashlib.sha256(json.dumps(membership).encode()).hexdigest()


def class_summary(rows):
    return {label: {'clips': sum(r['label'] == label for r in rows),
                    'sources': len({r['source'] for r in rows if r['label'] == label})}
            for label in CLASSES}


def inspect(root, output):
    from scripts.dataset_paths import resolve_dataset_root
    root, output = resolve_dataset_root(root), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output == root:
        raise ValueError('Inspection output must differ from the dataset root')
    output.mkdir(parents=True, exist_ok=True)
    # Exclude generated inspection outputs if the caller placed them below root.
    files = sorted((p for p in root.rglob('*') if p.is_file() and output not in p.parents),
                   key=lambda p: (len(p.relative_to(root).parts), str(p.relative_to(root)).casefold(), str(p.relative_to(root))))
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Dataset file symlink escapes verified root: {path}')
    candidates = defaultdict(list)
    for path in files:
        if path.suffix.lower() in VIDEO_EXT:
            candidates[canonical_clip(path, root)].append(path)
    rows, duplicates, clip_hashes = [], [], {}
    for clip, copies in sorted(candidates.items()):
        kept = copies[0]  # shortest relative path, then case-insensitive + exact lexical order
        # Persist evidence even for single-copy clips, enabling verified relocation.
        kept_hash = file_hash(kept)
        clip_hashes[clip] = kept_hash
        for excluded in copies[1:]:
            if kept.stat().st_size != excluded.stat().st_size or file_hash(excluded) != kept_hash:
                conflict = dict(canonical_clip=clip, kept_path=str(kept), conflicting_path=str(excluded))
                (output/'duplicate_conflict.json').write_text(json.dumps(conflict, indent=2))
                raise ValueError(f'Canonical collision with different video contents: {clip}')
            duplicates.append(dict(canonical_clip=clip, kept_path=str(kept), excluded_path=str(excluded),
                                   sha256=kept_hash, reason='same canonical clip and SHA256'))
        cap = cv2.VideoCapture(str(kept))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        rows.append(dict(path=str(kept), canonical_clip=clip, label=category(clip) or '',
                         source=source_id(kept, root), readable=bool(ok and frame is not None)))
    validate_manifest(rows)
    unmapped = [r for r in rows if not r['label']]
    # Keep raw directory counts separate from deduplicated clip/category counts.
    report = dict(schema_version=2, dataset_root=str(root), clip_sha256=clip_hashes, raw_video_files=sum(len(v) for v in candidates.values()),
                  unique_video_clips=len(rows), excluded_duplicate_copies=len(duplicates),
                  unique_original_sources=len({r['source'] for r in rows}),
                  canonicalization='Leading case-insensitive DCSASS Dataset wrappers removed; lowercase relative keys; original-video basename source IDs',
                  duplicate_policy='Shortest relative physical path then lexical order; all excluded copies SHA256 verified',
                  mapped_classes=CLASSES, mapped_counts=class_summary(rows),
                  unmapped_clips=len(unmapped), unmapped_sources=len({r['source'] for r in unmapped}),
                  unmapped_category_counts=dict(Counter(re.split(r'[\d]', r['source'], maxsplit=1)[0].rstrip('_-. ') for r in unmapped)),
                  directory_file_counts=dict(Counter(str(p.parent.relative_to(root)) for p in files)),
                  file_extensions=dict(Counter(p.suffix.lower() for p in files)),
                  video_counts=dict(Counter(r['label'] or 'unmapped' for r in rows)),
                  readable=sum(r['readable'] for r in rows), unreadable=sum(not r['readable'] for r in rows),
                  directories_named_mp4=[str(p.relative_to(root)) for p in sorted(root.rglob('*')) if p.is_dir() and p.suffix.lower() == '.mp4' and output not in p.parents],
                  annotation_candidates=[str(p.relative_to(root)) for p in files if p.suffix.lower() in {'.csv', '.txt', '.json', '.xml', '.yaml'}],
                  annotation_status='Unverified: manually inspect all candidates for bounding boxes and paired narratives.')
    write_csv(output/'manifest.csv', rows, FIELDS)
    write_csv(output/'excluded_duplicates.csv', duplicates, ['canonical_clip', 'kept_path', 'excluded_path', 'sha256', 'reason'])
    (output/'inspection.json').write_text(json.dumps(report, indent=2))
    return report


def split_manifest(manifest, output, seed=42, per_class=30):
    if per_class < 3:
        raise ValueError('Need at least 3 requested sources per class')
    with open(manifest) as f:
        all_rows = list(csv.DictReader(f))
    validate_manifest(all_rows)
    rows = [r for r in all_rows if r['label'] in CLASSES and str(r['readable']).lower() == 'true']
    groups = defaultdict(list)
    for row in rows:
        groups[row['source']].append(row)
    selected = {label: sorted(k for k, v in groups.items() if v[0]['label'] == label) for label in CLASSES}
    count = min(per_class, *(len(v) for v in selected.values()))
    if count < 3:
        raise ValueError(f'Need at least 3 readable independent sources in every class: {class_summary(rows)}')
    rng, result = random.Random(seed), []
    for label, keys in selected.items():
        rng.shuffle(keys)
        keys = keys[:count]
        nv = max(1, int(count * .2))
        nt = count - 2 * nv
        for i, key in enumerate(keys):
            split = 'train' if i < nt else 'val' if i < nt + nv else 'test'
            # One clip per source, sorted before seeded choice: independent of input row order.
            result.append(dict(rng.choice(sorted(groups[key], key=lambda r: r['canonical_clip'])), split=split))
    validate_manifest(result, require_splits=True, require_classes=True)
    counts = {s: class_summary([r for r in result if r['split'] == s]) for s in SPLITS}
    report = dict(seed=seed, requested_sources_per_class=per_class, selected_sources_per_class=count,
                  available_readable=class_summary(rows), split_counts=counts,
                  excluded_unmapped_clips=sum(r['label'] not in CLASSES for r in all_rows),
                  excluded_unreadable_mapped_clips=sum(r['label'] in CLASSES and str(r['readable']).lower() != 'true' for r in all_rows),
                  source_overlap={f'{a}/{b}': sorted({r['source'] for r in result if r['split'] == a} & {r['source'] for r in result if r['split'] == b})
                                  for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]},
                  limitations=['Source-balanced prototype uses one representative clip per original video.',
                               'Fighting has only eight sources in the user inspection; available readable counts above govern the actual run.',
                               'With 8 selected sources/class this policy gives 6 train, 1 validation, 1 test per class; with 3 it gives 1/1/1. Tiny holdouts cannot establish reliable generalization.',
                               'Naming canonicalization cannot detect different original IDs showing the same incident; review near duplicates manually.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, result, FIELDS + ['split'])
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    return result
