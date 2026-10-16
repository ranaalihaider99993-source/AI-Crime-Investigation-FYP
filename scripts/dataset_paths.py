"""Bound KaggleHub root normalization and verified, non-destructive path rebasing."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from scripts.data import (VIDEO_EXT, FIELDS, canonical_clip, file_hash,
                          source_id, validate_manifest, write_csv)


def _wrapper(name):
    return re.sub(r'[\s_-]+', ' ', name.casefold()).strip() == 'dcsass dataset'


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected an absolute path without traversal: {path}')
    return path


def _kaggle_package(path):
    """Recognize this dataset's exact identity, never an arbitrary common ancestor."""
    path = _absolute(path)
    parts = path.parts
    for i in range(len(parts)):
        # Standard KaggleHub cache, including the user's /root/.cache/.../versions/1.
        if parts[i:i+4] == ('kagglehub', 'datasets', 'mateohervas', 'dcsass-dataset'):
            tail = parts[i+4:]
            if len(tail) >= 2 and tail[0] == 'versions' and tail[1].isdigit():
                return Path(*parts[:i+6])
        # Kaggle/Colab mounted input layouts. A version is a separate boundary.
        if parts[i:i+2] == ('kaggle', 'input'):
            end = None
            if parts[i+2:i+5] == ('datasets', 'mateohervas', 'dcsass-dataset'):
                end = i+5
            elif parts[i+2:i+3] == ('dcsass-dataset',):
                end = i+3
            if end is not None:
                if parts[end:end+1] == ('versions',):
                    if len(parts) <= end+1 or not parts[end+1].isdigit():
                        return None
                    end += 2
                return Path(*parts[:end])
    return None


def resolve_dataset_root(download_path):
    """Treat KaggleHub's returned path as trusted input, with tightly bounded ascent.

    Only recognized DCSASS wrapper directories inside THIS dataset's package may
    be lifted to that package root. Custom download locations remain exact boundaries.
    """
    selected = _absolute(Path(download_path).absolute())
    if not selected.is_dir():
        raise FileNotFoundError(f'Dataset directory missing: {selected}')
    package = _kaggle_package(selected)
    if package is not None:
        relative = selected.relative_to(package)
        if any(not _wrapper(part) for part in relative.parts):
            raise ValueError(f'KaggleHub returned a category/source subdirectory, not a dataset root: {selected}')
        root = package.resolve()
        if not selected.resolve().is_relative_to(root):
            raise ValueError('Dataset wrapper symlink escapes the verified package root')
        return root
    # Do not ascend custom paths merely because their name resembles a wrapper.
    return selected.resolve()


def _csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def _inside_file(path, root):
    path = _absolute(path)
    if not path.is_relative_to(root):
        raise ValueError(f'Path outside verified dataset directory: {path}')
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Video symlink escapes verified dataset directory: {path}')
    if not resolved.is_file():
        raise FileNotFoundError(f'Missing video inside verified dataset directory: {path}')
    return resolved


def resolve_inspection_paths(download_path, inspection_dir, output):
    """Reuse current paths or rebase into verified root using canonical keys + SHA256.

    Never read old external files. Outside paths are provenance only, accepted
    solely under a recorded root or this dataset's recognized historical package.
    Every relocated physical copy must match recorded content evidence. Missing or
    ambiguous evidence requires notebook 01, rather than relaxing containment.
    """
    root = resolve_dataset_root(download_path)
    inspection_dir, output = Path(inspection_dir).resolve(), Path(output).resolve()
    rows = _csv(inspection_dir/'manifest.csv')
    duplicates = _csv(inspection_dir/'excluded_duplicates.csv')
    inspection = json.loads((inspection_dir/'inspection.json').read_text())
    if not rows:
        raise ValueError('Empty manifest; rerun notebook 01')
    saved_root = inspection.get('dataset_root')
    old_root = _absolute(saved_root) if saved_root else _kaggle_package(_absolute(rows[0]['path']))
    # Legacy schema v2 did not record its root. Same-root files are still bounded
    # by the trusted current root; only recognized historical Kaggle paths may move.
    if old_root is None:
        old_root = root

    indexed = None
    hash_cache = {}
    def digest(path):
        if path not in hash_cache:
            hash_cache[path] = file_hash(path)
        return hash_cache[path]

    def index_files():
        nonlocal indexed
        indexed = defaultdict(list)
        for path in root.rglob('*'):
            if path.suffix.lower() in VIDEO_EXT and path.is_file():
                verified = _inside_file(path, root)
                indexed[canonical_clip(path, root)].append(verified)
        for paths in indexed.values():
            paths.sort(key=lambda p: (len(p.relative_to(root).parts), str(p.relative_to(root)).casefold(), str(p.relative_to(root))))
        # A different dataset release may not be silently substituted.
        expected = {r['canonical_clip'] for r in rows}
        if set(indexed) != expected:
            raise ValueError('Current dataset clip identities differ from saved inspection; rerun notebook 01')
        expected_counts = defaultdict(int)
        for row in rows + duplicates:
            expected_counts[row['canonical_clip']] += 1
        if any(len(indexed[key]) != expected_counts[key] for key in expected):
            raise ValueError('Current duplicate-copy layout/count differs from inspection; rerun notebook 01')

    hashes = dict(inspection.get('clip_sha256', {}))
    for record in duplicates:
        key, value = record['canonical_clip'], record['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Invalid recorded duplicate SHA256; rerun notebook 01')
        if key in hashes and hashes[key] != value:
            raise ValueError('Conflicting saved SHA256 evidence; rerun notebook 01')
        hashes[key] = value

    used, old_to_new, relocations = set(), {}, []
    def bind(old_path, key):
        old_path = _absolute(old_path)
        if old_path in old_to_new:
            raise ValueError(f'Duplicate physical record in inspection: {old_path}')
        # Existing in-bound paths never gain permission to escape through symlinks.
        if old_path.is_relative_to(root):
            target = _inside_file(old_path, root)
            if canonical_clip(old_path, root) != key:
                raise ValueError('Physical path/canonical clip mismatch')
        else:
            if not old_path.is_relative_to(old_root):
                raise ValueError(f'Saved path outside recorded/recognized dataset root: {old_path}')
            relative = old_path.relative_to(old_root)
            if canonical_clip(old_path, old_root) != key:
                raise ValueError('Saved relative identity differs from canonical clip')
            if indexed is None:
                index_files()
            candidates = [p for p in indexed.get(key, []) if p not in used]
            # Preserve the physical-copy identity when its wrapper layout still exists.
            exact = [p for p in candidates if str(p.relative_to(root)).casefold() == str(relative).casefold()]
            if len(exact) > 1:
                raise ValueError('Ambiguous case-insensitive relative identity; rerun notebook 01')
            if exact:
                target = exact[0]
            elif candidates:
                target = candidates[0]  # canonical identity + equal copy count + SHA256 below
            else:
                raise FileNotFoundError(f'No unused canonical copy inside current dataset: {key}')
            target = _inside_file(target, root)
            expected = hashes.get(key)
            if not expected:
                raise ValueError(f'No saved SHA256 evidence for relocation of {key}; rerun notebook 01')
            if digest(target) != expected:
                raise ValueError(f'Relocated video SHA256 differs: {key}; rerun notebook 01')
            relocations.append(dict(canonical_clip=key, old_path=str(old_path), new_path=str(target), sha256=expected))
        if canonical_clip(target, root) != key:
            raise ValueError('Resolved file does not retain its canonical relative identity')
        if target in used:
            raise ValueError(f'Multiple records resolve to one physical copy: {key}')
        used.add(target)
        old_to_new[old_path] = target
        return str(target)

    rebound = []
    kept = {}
    for row in rows:
        key = row['canonical_clip']
        if key in kept:
            raise ValueError(f'Duplicate canonical clip: {key}')
        new = dict(row, path=bind(row['path'], key))
        if source_id(Path(key), Path('.')) != row['source']:
            raise ValueError('Canonical original source changed; rerun notebook 01')
        kept[key] = (_absolute(row['path']), new['path'])
        rebound.append(new)
    rebound_duplicates = []
    for row in duplicates:
        key = row['canonical_clip']
        if key not in kept or _absolute(row['kept_path']) != kept[key][0]:
            raise ValueError('Duplicate kept path does not match original manifest')
        rebound_duplicates.append(dict(row, kept_path=kept[key][1], excluded_path=bind(row['excluded_path'], key)))
    validate_manifest(rebound)
    resolution = dict(selected_download_root=str(Path(download_path).absolute()), verified_dataset_root=str(root),
                      recorded_or_recognized_root=str(old_root), relocated_copies=len(relocations),
                      reverified_sha256_files=len(hash_cache), original_inspection_preserved=True)
    if not relocations:
        return root, inspection_dir, resolution
    output.mkdir(parents=True, exist_ok=True)
    updated = dict(inspection, dataset_root=str(root), inspection_origin_root=str(old_root), path_resolution=resolution)
    write_csv(output/'manifest.csv', rebound, FIELDS)
    write_csv(output/'excluded_duplicates.csv', rebound_duplicates, ['canonical_clip','kept_path','excluded_path','sha256','reason'])
    write_csv(output/'path_relocations.csv', relocations, ['canonical_clip','old_path','new_path','sha256'])
    (output/'inspection.json').write_text(json.dumps(updated, indent=2))
    return root, output, resolution
