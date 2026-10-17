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
from pathlib import Path
from scripts.context_review import context_plan
from scripts.data import VIDEO_EXT, canonical_clip, source_id, file_hash, write_csv
from scripts.dataset_paths import resolve_inspection_paths, resolve_dataset_root
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
        'or set RECOVERY_VERSION to its exact Kaggle version and run recovery. If the original version is unknown, recover that version number or the original full dataset backup first. '
        'Only the 12 exported videos are insufficient for full saved-inspection verification. Do not rerun inspection, preparation or splitting.')


def recover_review_dataset(directory, *, existing_root=None, version=None):
    """Explicit recovery cell only. Never implicitly download an unknown latest release."""
    if existing_root is not None:
        plan=ensure_review_dataset(directory,existing_root);return Path(plan['dataset_root'])
    saved,_,report,_,inspection=_artifacts(directory)
    recorded=[saved['dataset_root'],report['data_root'],json.loads((inspection/'inspection.json').read_text()).get('dataset_root','')]
    known={m[1] for value in recorded if (m:=re.search(r'/versions/(\d+)(?:/|$)',value))}
    if version is None:
        require(len(known)==1,'Original Kaggle version is not recorded unambiguously. Set RECOVERY_VERSION to the original version number from Kaggle dataset history, or set REVIEW_DATASET_ROOT to a full original dataset backup; latest will NOT be substituted')
        version=next(iter(known))
    require(str(version).isdigit() and int(version)>0,'RECOVERY_VERSION must be a positive exact Kaggle version number')
    if known:require(known=={str(version)},'Recovery version conflicts with recorded version')
    import kagglehub
    handle=DATASET+'/versions/'+str(version)
    print('Downloading exact version for verification, not yet accepted:',handle,flush=True)
    downloaded=kagglehub.dataset_download(handle)
    plan=ensure_review_dataset(directory,downloaded)
    return Path(plan['dataset_root'])
