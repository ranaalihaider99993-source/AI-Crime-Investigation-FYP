"""Append-only complete dataset backups; archive hashes never replace data checks."""
import json
import shutil
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.data import file_hash
from scripts.review_paths import ensure_review_dataset, recovery_evidence, _BINDINGS


def backup_directory(review):
    return Path(review).parent.parent.parent / 'dataset_backups'


def _verify(review, root):
    _BINDINGS.pop(str(Path(review).resolve()), None)
    return ensure_review_dataset(review, root)


def create_backup(review, root, quota):
    evidence = recovery_evidence(review)
    if evidence['recorded_versions'] != ['1']:
        raise ValueError('Backup requires unambiguous saved DCSASS version 1 evidence')
    plan = _verify(review, root)
    root = Path(plan['dataset_root'])
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in root.rglob('*')):
        raise ValueError('Dataset contains a symlink or escaping path')
    size = sum(p.stat().st_size for p in files)
    required = size + len(files) * 2048 + 256 * 1024**2
    if quota.get('source') != 'Google Drive API storageQuota' or quota.get('available_bytes', 0) < required:
        raise RuntimeError(f'Insufficient or unverified Drive space: need {required}; quota={quota}')
    if shutil.disk_usage(tempfile.gettempdir()).free < required:
        raise RuntimeError(f'Insufficient local archive space: need {required}')
    destination = backup_directory(review) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='crimescope-backup-') as temporary:
        archive = Path(temporary)/'dcsass-v1.tar'
        with tarfile.open(archive, 'x') as target:
            for path in files:
                target.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        digest = file_hash(archive)
        saved = destination/archive.name
        with archive.open('rb') as source, saved.open('xb') as target:
            shutil.copyfileobj(source, target, length=8*1024**2)
        if file_hash(saved) != digest:
            raise ValueError('Drive archive SHA-256 mismatch; backup is not published')
    # Check original evidence and source data again before publishing provenance.
    if recovery_evidence(review) != evidence:
        raise ValueError('Saved evidence changed during backup')
    _verify(review, root)
    receipt = dict(status='COMPLETE', archive=saved.name, archive_sha256=digest,
                   archive_bytes=saved.stat().st_size, uncompressed_bytes=size,
                   extraction_budget_bytes=required, file_count=len(files),
                   requested_handle='mateohervas/dcsass-dataset/versions/1',
                   verification='COMPLETE_SAVED_VIDEO_MANIFEST_MATCH',
                   unique_clips=evidence['unique_clips'], physical_copies=evidence['physical_copies'],
                   source_fingerprints=evidence['source_fingerprints'],
                   canonical_split_signature=plan['canonical_split_signature'], quota=quota,
                   human_files_modified=False, training_executed=False)
    (destination/'dcsass-v1.tar.sha256').write_text(digest+'  '+saved.name+'\n')
    (destination/'provenance.json').write_text(json.dumps(receipt, indent=2))
    print('VERIFIED PERSISTENT DATASET BACKUP:', destination, json.dumps(receipt), flush=True)
    return destination


def safe_extract(archive, destination, budget):
    """Reject traversal, links, special files, collisions and oversized archives."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, 'r:') as source:
        members = source.getmembers()
        seen = set()
        total = 0
        for member in members:
            relative = Path(member.name)
            target = (destination/relative).resolve()
            if (relative.is_absolute() or '..' in relative.parts or '\\' in member.name
                    or ':' in member.name or not target.is_relative_to(destination)
                    or not member.isfile() or target in seen or target.exists()):
                raise ValueError('Unsafe or colliding archive member: '+member.name)
            seen.add(target)
            total += member.size
        if total > budget:
            raise ValueError('Archive exceeds extraction budget')
        for member in members:
            target = destination/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer, length=8*1024**2)


def restore_backup(review):
    """Restore only published local Drive archives; never downloads from Kaggle."""
    receipts = sorted(backup_directory(review).glob('*/provenance.json'), reverse=True)
    if not receipts:
        return None
    evidence = recovery_evidence(review)
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') != 'COMPLETE' or receipt.get('source_fingerprints') != evidence['source_fingerprints']:
            continue
        archive = receipt_path.parent/receipt['archive']
        if archive.parent != receipt_path.parent or not archive.is_file() or file_hash(archive) != receipt['archive_sha256']:
            raise ValueError('Persistent archive missing or SHA-256 mismatch')
        required = receipt['extraction_budget_bytes']
        if shutil.disk_usage(tempfile.gettempdir()).free < required:
            raise RuntimeError('Insufficient local space to restore verified archive')
        root = Path(tempfile.mkdtemp(prefix='crimescope-dcsass-v1-'))
        safe_extract(archive, root, required)
        plan = _verify(review, root)
        print('RESTORED AND FULLY REVERIFIED:', root, flush=True)
        return Path(plan['dataset_root'])
    return None
