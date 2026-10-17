"""Verified runtime-only relocation of EXISTING saved review artifacts.

Original Drive metadata/manifests/human files are never rewritten. A temporary
path-only view preserves exact canonical membership, hashes and decision inputs.
"""
import copy
from collections import Counter
import shutil
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from scripts.context_review import context_plan
from scripts.data import VIDEO_EXT, canonical_clip, source_id, file_hash, write_csv
from scripts.dataset_paths import resolve_inspection_paths, resolve_dataset_root, _kaggle_package
from scripts.smoke import artifact_hashes, read_csv, require

_BINDINGS={}
DATASET='mateohervas/dcsass-dataset'


def _same_context(saved,current):
    # Historical paths are provenance, never permission to read outside either
    # root. Validate their lexical identities even when their files no longer exist.
    for plan in [saved,current]:
        root=Path(plan['dataset_root'])
        require(root.is_absolute() and '..' not in root.parts,'Invalid recorded dataset root')
        for group in plan['groups']:
            for clip in group['videos']:
                path=Path(clip['path'])
                require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(root),'Saved clip path outside its recorded dataset root')
                require(canonical_clip(path,root)==clip['canonical_clip'] and source_id(path,root)==clip['source'],'Saved clip path canonical identity differs')
    for key in ['canonical_split_signature','manifest_sha256','total_video_bytes','groups']:
        left,right=copy.deepcopy(saved[key]),copy.deepcopy(current[key])
        if key=='groups':
            for groups in [left,right]:
                for group in groups:
                    for clip in group['videos']:clip.pop('path',None)
        require(left==right,'Saved context differs from verified candidate/context: '+key)


def validation_location(directory):
    directory=Path(directory)
    saved=json.loads((directory/'context_metadata.json').read_text())
    paths=[Path(saved['validation_path']),directory.parent.parent/'validation.json']
    for path in dict.fromkeys(paths):
        if path.is_file():
            report=json.loads(path.read_text())
            require(report['automatic_status']=='PASS' and report['split_file_sha256']==saved['manifest_sha256']
                    and report['split_signature']==saved['canonical_split_signature'],'Saved validation does not bind to this context candidate')
            return path.resolve()
    raise FileNotFoundError('Saved validation.json unavailable. Restore it from Drive backup in '+str(directory.parent.parent)+'; do not rerun preparation')


def temporal_location(directory):
    return validation_location(directory).parent/'temporal_review'/'review.json'


def discover_dataset_roots(directory):
    saved=json.loads((Path(directory)/'context_metadata.json').read_text())
    candidates=[Path(saved['dataset_root'])]
    caches=[Path.home()/'.cache/kagglehub',Path('/root/.cache/kagglehub'),Path('/content/.cache/kagglehub')]
    if os.environ.get('KAGGLEHUB_CACHE'):caches.append(Path(os.environ['KAGGLEHUB_CACHE']))
    for cache in caches:
        versions=cache/'datasets'/DATASET/'versions'
        try:
            if versions.is_dir():candidates.extend(sorted(p for p in versions.iterdir() if p.is_dir() and p.name.isdigit()))
        except OSError:continue
    output=Path(directory).parent.parent.parent
    candidates.extend([Path('/kaggle/input/dcsass-dataset'),Path('/content/dcsass-dataset'),Path('/content/datasets/dcsass-dataset'),output/'datasets/dcsass-dataset'])
    existing=[]
    for path in candidates:
        try:
            if path.is_dir():existing.append(path.absolute())
        except OSError:continue
    return list(dict.fromkeys(existing))


def _artifacts(directory):
    directory=Path(directory).resolve();saved=json.loads((directory/'context_metadata.json').read_text())
    validation=validation_location(directory);report=json.loads(validation.read_text())
    prep=directory.parent.parent
    candidates=[Path(report['split_manifest']),prep/Path(report['split_manifest']).name]
    manifest=next((p for p in candidates if p.is_file() and file_hash(p)==report['split_file_sha256']),None)
    require(manifest is not None,'Saved candidate CSV missing or changed. Restore the ORIGINAL file from Drive backup; do not regenerate the split')
    options=[Path(report['inspection_dir']),prep/'resolved_inspection',prep.parent/'inspection']
    if report.get('original_inspection_dir'):options.append(Path(report['original_inspection_dir']))
    expected=[report['artifact_hashes']]
    if report.get('original_artifact_hashes'):expected.append(report['original_artifact_hashes'])
    inspection=None
    for path in dict.fromkeys(options):
        try:
            if artifact_hashes(path) in expected:inspection=path;break
        except FileNotFoundError:continue
    require(inspection is not None,'Original inspection artifacts missing or changed; restore the saved manifest.csv, inspection.json and excluded_duplicates.csv from Drive backup. Do not rerun inspection/preparation for this review')
    return saved,validation,report,manifest.resolve(),inspection.resolve()


def _fingerprints(directory,validation,manifest,inspection):
    return {str(p):file_hash(p) for p in [Path(directory)/'context_metadata.json',validation,manifest,
            inspection/'manifest.csv',inspection/'inspection.json',inspection/'excluded_duplicates.csv']}


def _current_binding(directory,binding):
    require(all(Path(p).is_file() and file_hash(p)==digest for p,digest in binding['fingerprints'].items()),'Saved review provenance changed after dataset relocation')
    runtime=context_plan(binding['runtime_validation'])
    runtime['manifest_sha256']=binding['original_manifest_sha256']
    runtime['validation_path']=binding['original_validation_path']
    saved=json.loads((Path(directory)/'context_metadata.json').read_text())
    _same_context(saved,runtime)
    return runtime


def ensure_review_dataset(directory, dataset_root=None):
    """Use original paths, or verify complete saved identities/copy counts/SHA256."""
    directory=Path(directory).resolve()
    binding=_BINDINGS.get(str(directory))
    if binding and (dataset_root is None or resolve_dataset_root(dataset_root)==Path(binding['root'])):
        try:return _current_binding(directory,binding)
        except FileNotFoundError:_BINDINGS.pop(str(directory),None)
    saved=json.loads((directory/'context_metadata.json').read_text())
    if dataset_root is None:
        try:
            current=context_plan(saved['validation_path'])
        except FileNotFoundError:pass
        else:
            _same_context(saved,current)
            require(current['dataset_root']==saved['dataset_root'],'Original root identity changed')
            return current
    saved,validation,report,manifest,inspection=_artifacts(directory)
    candidates=[Path(dataset_root)] if dataset_root is not None else discover_dataset_roots(directory)
    errors=[]
    for candidate in candidates:
        workspace=Path(tempfile.mkdtemp(prefix='crimescope-review-paths-'))
        try:
            print('Verifying saved dataset identities, duplicate copies and SHA-256:',candidate,flush=True)
            root,resolved,resolution=resolve_inspection_paths(candidate,inspection,workspace/'inspection')
            # Even an existing root must supply saved SHA evidence for every file
            # on this relocation branch; the original validator may have kept paths.
            inspect_meta=json.loads((resolved/'inspection.json').read_text())
            hashes=dict(inspect_meta.get('clip_sha256',{}))
            duplicates=read_csv(resolved/'excluded_duplicates.csv')
            for row in duplicates:
                require(row['canonical_clip'] not in hashes or hashes[row['canonical_clip']]==row['sha256'],'Conflicting inspection SHA evidence')
                hashes[row['canonical_clip']]=row['sha256']
            full=read_csv(resolved/'manifest.csv');indexed={r['canonical_clip']:r for r in full}
            if resolution['reverified_sha256_files']!=len(full)+len(duplicates):
                expected_counts=Counter(r['canonical_clip'] for r in full+duplicates)
                actual_counts=Counter()
                for path in root.rglob('*'):
                    if path.suffix.lower() not in VIDEO_EXT or not path.is_file():continue
                    require(path.resolve().is_relative_to(root),'Dataset video symlink escapes verified root')
                    key=canonical_clip(path,root);actual_counts[key]+=1
                    require(hashes.get(key)==file_hash(path),'Missing or differing dataset SHA-256: '+key)
                require(actual_counts==expected_counts,'Dataset canonical identities or duplicate-copy counts differ')
            selected=read_csv(manifest)
            for row in selected:
                require(row['canonical_clip'] in indexed,'Selected clip absent from verified dataset')
                row['path']=indexed[row['canonical_clip']]['path']
            shadow=workspace/'candidate_paths.csv';write_csv(shadow,selected,list(selected[0]))
            relocated=dict(report,data_root=str(root),inspection_dir=str(resolved),artifact_hashes=artifact_hashes(resolved),
                           split_manifest=str(shadow),split_file_sha256=file_hash(shadow))
            runtime_validation=workspace/'validation.json';runtime_validation.write_text(json.dumps(relocated,indent=2))
            binding=dict(root=str(root),runtime_validation=str(runtime_validation),original_manifest_sha256=saved['manifest_sha256'],
                         original_validation_path=saved['validation_path'],fingerprints=_fingerprints(directory,validation,manifest,inspection))
            current=_current_binding(directory,binding)
            _BINDINGS[str(directory)]=binding
            print('Verified runtime-only dataset root:',root,'— original review, candidate, annotations and decisions unchanged.',flush=True)
            return current
        except (OSError,ValueError,KeyError) as exc:
            errors.append(str(candidate)+': '+str(exc).replace('rerun notebook 01','restore the exact saved artifacts/dataset; do not rerun inspection'))
            shutil.rmtree(workspace)
    detail='\n'.join(errors) if errors else 'No DCSASS dataset root was found in this runtime.'
    raise FileNotFoundError('Dataset directory missing or no relocated root passed verification.\n'+detail+
        '\nRun Notebook 02 recovery cell "Restore exact saved-review dataset". Set REVIEW_DATASET_ROOT to your full original dataset copy, '
        'or run the default recovery cell to investigate recorded version evidence and verify one downloaded candidate against the COMPLETE saved manifest. An unknown historical version is not guessed. '
        'Only the 12 exported videos are insufficient for full saved-inspection verification. Do not rerun inspection, preparation or splitting.')


def recovery_evidence(directory):
    """Inspect bound records, not README/test examples or unbound external logs."""
    saved,validation,report,manifest,inspection=_artifacts(directory)
    metadata=json.loads((inspection/'inspection.json').read_text())
    full=read_csv(inspection/'manifest.csv');duplicates=read_csv(inspection/'excluded_duplicates.csv')
    keys={r['canonical_clip'] for r in full}
    require(full and len(keys)==len(full),'Saved full manifest is empty or has duplicate canonical IDs')
    hashes=dict(metadata.get('clip_sha256',{}))
    for row in duplicates:
        key,digest=row['canonical_clip'],row['sha256']
        require(key in keys and (key not in hashes or hashes[key]==digest),'Conflicting saved duplicate SHA evidence')
        hashes[key]=digest
    require(all(re.fullmatch(r'[0-9a-f]{64}',hashes.get(key,'')) for key in keys),
            'Complete original SHA-256 evidence is unavailable. Restore the ORIGINAL inspection/duplicate audit from backup; no candidate download can be safely accepted without it')
    citations={}
    def inspect_values(value,origin):
        if isinstance(value,dict):
            for key,item in value.items():inspect_values(item,origin+'.'+str(key))
        elif isinstance(value,list):
            for i,item in enumerate(value):inspect_values(item,origin+'['+str(i)+']')
        elif isinstance(value,str) and value.startswith('/'):
            try:package=_kaggle_package(Path(value))
            except ValueError:return
            if package and len(package.parts)>=2 and package.parts[-2]=='versions':
                version=package.parts[-1]
                examples=citations.setdefault(version,[])
                if len(examples)<5:examples.append(dict(record=origin,path=value))
    inspect_values(saved,'context_metadata.json')
    inspect_values(report,'validation.json')
    inspect_values(metadata,'inspection.json')
    inspect_values(read_csv(manifest),'candidate_manifest')
    inspect_values(full,'inspection_manifest')
    return dict(recorded_versions=sorted(citations,key=int),version_evidence=citations,
                unique_clips=len(full),physical_copies=len(full)+len(duplicates),
                source_fingerprints=_fingerprints(directory,validation,manifest,inspection))


def recover_review_dataset(directory, *, existing_root=None, version=None, allow_candidate=False):
    """Explicit recovery action; a candidate is NEVER trusted as the original release.

    Exact historical version if bound records establish one; otherwise an explicit
    candidate opt-in downloads ONE current release and requires identical complete
    manifest/copy-count/SHA evidence. No iteration, split, inspection or training.
    """
    directory=Path(directory).resolve()
    if existing_root is not None:
        plan=ensure_review_dataset(directory,existing_root);return Path(plan['dataset_root'])
    evidence=recovery_evidence(directory)
    known=set(evidence['recorded_versions'])
    if version is None and len(known)==1:version=next(iter(known))
    if version is not None:
        require(str(version).isdigit() and int(version)>0,'RECOVERY_VERSION must be a positive exact Kaggle version number')
        if known:require(str(version) in known,'Recovery version conflicts with recorded version')
        handle=DATASET+'/versions/'+str(version)
        mode='recorded-version' if known=={str(version)} else 'explicit-version-candidate'
        print('Recorded version evidence:',evidence['version_evidence'] or 'No historical version ID recorded',flush=True)
    else:
        require(allow_candidate is True,'Original Kaggle version is not recorded unambiguously. Run the recovery cell with its explicit verified-candidate option, or provide RECOVERY_VERSION/full original backup. No unverified release will be accepted')
        handle=DATASET;mode='current-release-candidate'
        print('Historical version is NOT established. One current DCSASS release will be downloaded as an UNTRUSTED CANDIDATE, not assumed original.',flush=True)
        if known:print('Conflicting recorded versions:',evidence['version_evidence'],flush=True)
    print('Expected from saved records:',evidence['unique_clips'],'canonical clips;',evidence['physical_copies'],'physical video copies. ALL identities, copy counts and SHA-256 hashes must match.',flush=True)
    print('This can download the FULL large dataset. Keep Colab connected; local free space:',round(shutil.disk_usage(tempfile.gettempdir()).free/1024**3,1),'GiB. No review files will be replaced.',flush=True)
    print('If Kaggle requests authentication, sign in/accept dataset access on Kaggle and run import kagglehub; kagglehub.login() in Colab, then retry. Never paste credentials into committed cells.',flush=True)
    # Candidate opt-in may reuse a valid package already found since launcher startup.
    if allow_candidate is True:
        for root in discover_dataset_roots(directory):
            # Recovery revalidates the full package, not just a previously bound
            # selected/context subset, even when the SDK/cache is already present.
            _BINDINGS.pop(str(directory),None)
            try:
                plan=ensure_review_dataset(directory,root)
                print('Existing local package fully verified; no download needed.',flush=True)
                return Path(plan['dataset_root'])
            except FileNotFoundError:continue
    import kagglehub
    print('Downloading for verification, NOT YET ACCEPTED:',handle,flush=True)
    try:
        downloaded=kagglehub.dataset_download(handle)
    except Exception as exc:
        raise RuntimeError('Kaggle download did not complete; no review files changed. For authentication/access errors, authorize with kagglehub.login() and accept Kaggle dataset access, then rerun this recovery cell. For network/storage errors, restore connectivity or free Colab-local space without deleting Drive reviews/backups. Credentials must stay out of notebook source.') from exc
    # Force a fresh complete verification even if the SDK reused an existing cache.
    _BINDINGS.pop(str(directory),None)
    try:
        require(all(Path(path).is_file() and file_hash(path)==digest for path,digest in evidence['source_fingerprints'].items()),'Saved dataset records changed during download; retry against the preserved original records')
        plan=ensure_review_dataset(directory,downloaded)
        require(all(Path(path).is_file() and file_hash(path)==digest for path,digest in evidence['source_fingerprints'].items()),'Saved dataset records changed during verification')
    except (OSError,ValueError,KeyError) as exc:
        _BINDINGS.pop(str(directory),None)
        raise RuntimeError('Downloaded candidate REJECTED by complete saved-manifest verification. Existing human files/corrections are unchanged. Recover the original full backup or exact dataset version; do not regenerate the review or split. A latest release is never accepted merely because its name matches.') from exc
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_dir=directory/('dataset-recovery-'+stamp+'-'+uuid.uuid4().hex[:8]);receipt_dir.mkdir(exist_ok=False)
    package=_kaggle_package(Path(downloaded).absolute())
    cache_version=package.parts[-1] if package and package.parts[-2]=='versions' else None
    receipt=dict(mode=mode,requested_handle=handle,recorded_version_evidence=evidence['version_evidence'],
                 historical_version_established=len(known)==1,downloaded_cache_version=cache_version,
                 verification='COMPLETE_SAVED_VIDEO_MANIFEST_MATCH',unique_clips=evidence['unique_clips'],physical_copies=evidence['physical_copies'],
                 dataset_root=plan['dataset_root'],canonical_split_signature=plan['canonical_split_signature'],
                 original_source_fingerprints=evidence['source_fingerprints'],human_files_modified=False,training_executed=False)
    (receipt_dir/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('ACCEPTED: complete saved video identities, copy counts and SHA-256 match. Historical version remains unknown unless explicitly established by the recorded evidence above.',flush=True)
    print('Recovery receipt:',receipt_dir/'recovery_receipt.json',flush=True)
    return Path(plan['dataset_root'])
