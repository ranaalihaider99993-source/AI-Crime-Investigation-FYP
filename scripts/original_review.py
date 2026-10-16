"""Original-only playback/export for an EXISTING candidate. Never resplit/train."""
import base64
import csv
import hashlib
import html
import io
import json
import mimetypes
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile, ZIP_STORED
from scripts.data import canonical_clip, file_hash, source_id, split_signature, validate_manifest
from scripts.dataset_paths import resolve_dataset_root

LOW_MOTION = frozenset(['burglary041','stealing060','stealing054','stealing002',
                        'fighting005','fighting002','vandalism019','vandalism043'])
MAX_CLIP_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
INLINE_TOTAL_BYTES = 128 * 1024**2
RESERVE_BYTES = 16 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_key(source):
    return source.casefold().removesuffix('_x264')


def original_review_plan(validation_path, max_clip_bytes=MAX_CLIP_BYTES, max_total_bytes=MAX_TOTAL_BYTES):
    """Validate the saved selection and byte identities without calling any splitter."""
    report = json.loads(Path(validation_path).read_text())
    _require(report['automatic_status']=='PASS', 'Existing smoke preparation must pass first')
    root = resolve_dataset_root(report['data_root'])
    manifest = Path(report['split_manifest'])
    _require(file_hash(manifest)==report['split_file_sha256'], 'Candidate split file changed; do not resplit for this workflow')
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    validate_manifest(rows,require_splits=True,require_classes=True)
    _require(len(rows)==12 and len({r['source'] for r in rows})==12, 'Exactly 12 selected clips/sources required')
    _require(split_signature(manifest)==report['split_signature'], 'Canonical split assignments changed')
    clips = []
    for row in rows:
        path = Path(row['path']).resolve()
        _require(path.is_relative_to(root), 'Original video escapes verified dataset root')
        _require(canonical_clip(path,root)==row['canonical_clip'] and source_id(path,root)==row['source'], 'Original path identity differs')
        size = path.stat().st_size
        _require(0<size<=max_clip_bytes, f'Original clip exceeds size bound or is empty: {path.name} ({size} bytes)')
        _require(file_hash(path)==report['selected_clip_sha256'][row['canonical_clip']], 'Selected original content changed')
        _require('/' not in row['source'] and '\\' not in row['source'] and row['source'] not in {'.','..'}, 'Unsafe source identifier')
        clips.append(dict(path=str(path),filename=path.name,label=row['label'],source=row['source'],split=row['split'],
                          canonical_clip=row['canonical_clip'],size_bytes=size,sha256=report['selected_clip_sha256'][row['canonical_clip']],
                          package_path=f"clips/{row['split']}/{row['source']}/{path.name}",
                          reported_low_motion=_source_key(row['source']) in LOW_MOTION))
    total = sum(c['size_bytes'] for c in clips)
    _require(total<=max_total_bytes, f'Selected originals total {total} bytes exceeds {max_total_bytes} byte package bound')
    temporal_path=Path(validation_path).parent/'temporal_review'/'review.json'
    temporal_hash=None
    if temporal_path.is_file():
        temporal=json.loads(temporal_path.read_text())
        _require(temporal['canonical_split_signature']==report['split_signature']
                 and {c['canonical_clip']:c['sha256'] for c in temporal['clips']}==report['selected_clip_sha256'],
                 'Existing temporal review belongs to different source assignments or content; do not substitute a new candidate')
        temporal_hash=file_hash(temporal_path)
    selected_keys = {_source_key(c['source']) for c in clips}
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),dataset_root=str(root),manifest_path=str(manifest),
                validation_path=str(Path(validation_path).resolve()),temporal_review_sha256=temporal_hash,manifest_sha256=report['split_file_sha256'],
                canonical_split_signature=report['split_signature'],clips=clips,total_video_bytes=total,
                max_clip_bytes=max_clip_bytes,max_total_bytes=max_total_bytes,
                selected_low_motion_sources=sorted(LOW_MOTION & selected_keys),
                reported_sources_not_selected=sorted(LOW_MOTION - selected_keys),
                human_approval='PENDING',semantic_labels='UNVERIFIED',incident_independence='UNVERIFIED',training_executed=False)


def _player(clip, url):
    escape=lambda s:html.escape(str(s),quote=True)
    flag='<p><strong>User-reported low motion: watch closely; label/incident remains unverified.</strong></p>' if clip['reported_low_motion'] else ''
    border='#b76a00' if clip['reported_low_motion'] else '#a3b0bb'
    return (f'<section style="border:2px solid {border};padding:12px;margin:12px 0">'
            f'<h3>{escape(clip["label"])} · {escape(clip["source"])} · {escape(clip["split"])}</h3>'
            f'<p>Original filename: <code>{escape(clip["filename"])}</code> ({clip["size_bytes"]:,} bytes)</p>{flag}'
            f'<video controls preload="none" playsinline style="width:100%;max-width:720px" src="{escape(url)}">'
            'Your browser cannot play this original codec. Use the downloaded original in a compatible player.</video>'
            '<p>If playback fails, open the original from the review ZIP with a compatible player. No transcoding was performed.</p></section>')


def original_players(plan):
    """Yield one embedded HTML5 player at a time, never rely on inaccessible file URLs."""
    _require(plan['total_video_bytes']<=INLINE_TOTAL_BYTES,
             f'Originals exceed the {INLINE_TOTAL_BYTES} byte inline limit. Export the bounded package and use its index.html instead')
    for clip in plan['clips']:
        path = Path(clip['path'])
        _require(path.stat().st_size==clip['size_bytes'] and file_hash(path)==clip['sha256'], 'Original changed since planning')
        mime = mimetypes.guess_type(clip['filename'])[0] or 'application/octet-stream'
        payload = base64.b64encode(path.read_bytes()).decode('ascii')
        yield _player(clip,f'data:{mime};base64,{payload}')


def drive_storage_quota():
    """Read actual Drive quota in authorized Colab; disk_usage(FUSE) is NOT quota."""
    try:
        from google.colab import auth
        import google.auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        credentials,_ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.metadata.readonly'])
        service = build('drive','v3',credentials=credentials,cache_discovery=False)
        storage = service.about().get(fields='storageQuota').execute()['storageQuota']
        if 'limit' not in storage or 'usage' not in storage:
            raise ValueError('Drive did not return a finite quota/usage; available storage cannot be verified')
        limit,usage = int(storage['limit']),int(storage['usage'])
        _require(limit>=0 and usage>=0, 'Invalid Drive storage metadata')
        return dict(source='Google Drive API storageQuota',limit_bytes=limit,usage_bytes=usage,available_bytes=max(0,limit-usage))
    except Exception as exc:
        raise RuntimeError('Drive storage check failed. Authorize Drive metadata access in Colab; no archive was exported. '+str(exc)) from exc


def _metadata_files(plan, quota, required):
    # Original paths are metadata only. No root-wide copying or globbing occurs.
    metadata = dict(plan,storage_check=dict(quota,required_free_bytes=required),
                    content_policy='Original bytes; no re-encoding, cropping, resizing or filename changes')
    checklist = io.StringIO(newline='')
    fields=['source','label','split','filename','reported_low_motion','watched_entire_clip',
            'label_decision','incident_independence_decision','reviewer','notes']
    writer=csv.DictWriter(checklist,fieldnames=fields);writer.writeheader()
    for clip in plan['clips']:
        writer.writerow({**{k:clip[k] for k in fields[:5]},'watched_entire_clip':'PENDING',
                         'label_decision':'PENDING','incident_independence_decision':'PENDING','reviewer':'','notes':''})
    header='<!doctype html><html lang="en"><meta charset="utf-8"><title>Original selected videos</title><body style="font:16px system-ui;max-width:1000px;margin:auto"><h1>12 original selected clips — human approval PENDING</h1><p>Watch every full clip; verify labels and related incidents. Highlighted low motion is user-reported, not a semantic decision. Editing this checklist does not enable training.</p>'
    players=''.join(_player(c,quote(c['package_path'])) for c in plan['clips'])
    readme=('Extract the complete ZIP before opening index.html. It references exactly 12 originals in clips/.\n'
            'Bytes and basenames are unchanged. Filenames that collide are separated by split/source folders.\n'
            'If the original codec is unsupported by your browser, open it in a compatible media player.\n'
            'Fill human_checklist.csv only after watching full clips. PENDING is not approval.\n'
            'Resolve uncertain labels and possible related incidents, especially across splits.\n'
            'Do not change this candidate selection or train during this workflow.\n'
            'Keep SOURCE_MAPPING_REVIEWED=False and RUN_PROTOTYPE_TRAINING=False.\n')
    return {'review_metadata.json':json.dumps(metadata,indent=2).encode(),
            'human_checklist.csv':checklist.getvalue().encode(),'index.html':(header+players+'</body></html>').encode(),
            'README.txt':readme.encode()}


def export_original_package(plan, output_dir, quota):
    """Stream exactly 12 bound originals into a new ZIP after quota/space checks."""
    # Revalidate against the saved preflight, not caller-edited plan values.
    current=original_review_plan(plan['validation_path'],plan['max_clip_bytes'],plan['max_total_bytes'])
    bound_fields=['dataset_root','manifest_path','manifest_sha256','canonical_split_signature','clips',
                  'total_video_bytes','max_clip_bytes','max_total_bytes','temporal_review_sha256']
    _require(all(current[k]==plan[k] for k in bound_fields), 'Selected plan changed; regenerate the original review plan')
    _require(quota.get('source')=='Google Drive API storageQuota', 'A successful Drive API quota check is required')
    _require(isinstance(quota.get('limit_bytes'),int) and isinstance(quota.get('usage_bytes'),int)
             and quota['limit_bytes']>=0 and quota['usage_bytes']>=0
             and isinstance(quota.get('available_bytes'),int)
             and quota['available_bytes']==max(0,quota['limit_bytes']-quota['usage_bytes']), 'Invalid Drive storage check')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    # Two copies of estimated archive size cover FUSE rename/copy behavior; retain margin.
    required=2*(plan['total_video_bytes']+1024**2)+RESERVE_BYTES
    _require(quota['available_bytes']>=required, f'Insufficient Drive storage: need {required} free bytes, available {quota["available_bytes"]}')
    local=Path('/content') if Path('/content').is_dir() else Path(tempfile.gettempdir())
    _require(shutil.disk_usage(output).free>=required and shutil.disk_usage(local).free>=required,
             f'Insufficient filesystem/cache space: need {required} bytes')
    files=_metadata_files(plan,quota,required)
    files['selected_clips.csv']=Path(plan['manifest_path']).read_bytes()
    _require(hashlib.sha256(files['selected_clips.csv']).hexdigest()==plan['manifest_sha256'], 'Candidate snapshot changed during export')
    _require(sum(len(content) for content in files.values())<=1024**2, 'Review metadata exceeds the bounded 1 MiB budget')
    _require(len(set(c['package_path'] for c in plan['clips']))==12, 'Archive member collision')
    destination=output/'original_video_review.zip'
    if destination.exists():
        # Preserve an existing package instead of overwriting a possibly annotated copy.
        raise FileExistsError(f'Package already exists: {destination}. Download it or choose a new export directory; it was not overwritten')
    partial=output/'original_video_review.zip.partial'
    _require(not partial.exists(), 'A partial export exists; inspect/remove that generated partial file before retrying')
    try:
        with ZipFile(partial,'x',compression=ZIP_STORED,allowZip64=True) as archive:
            for clip in plan['clips']:
                digest=hashlib.sha256();size=0
                with Path(clip['path']).open('rb') as source,archive.open(clip['package_path'],'w',force_zip64=True) as target:
                    for block in iter(lambda:source.read(1024**2),b''):
                        size+=len(block)
                        _require(size<=clip['size_bytes'], 'Original grew during export')
                        digest.update(block);target.write(block)
                _require(size==clip['size_bytes'] and digest.hexdigest()==clip['sha256'], 'Original changed during export')
            for name,content in files.items():archive.writestr(name,content)
        # Confirm archive boundaries and metadata sizes before declaring completion.
        with ZipFile(partial) as archive:
            _require(set(archive.namelist())==set(files)|{c['package_path'] for c in plan['clips']}, 'Unexpected archive members')
            _require(archive.testzip() is None, 'Archive CRC verification failed')
        partial.rename(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination
