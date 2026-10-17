"""Open an existing review, never create data/candidates or launch training."""
import json
from pathlib import Path
from scripts.smoke import read_csv, require
from scripts.review_interface import OPTIONS, DECISIONS


def choose_context(output, explicit=None):
    """Prefer the context with most recent saved human activity, not a blank rerun."""
    if explicit:
        directory=Path(explicit)
        require((directory/'context_metadata.json').is_file() and (directory/'human_checklist.csv').is_file(),
                'Explicit review folder must contain context_metadata.json and human_checklist.csv')
        return directory
    folders=list((Path(output)/'smoke_preparation'/'source_context_review').glob('review-*/context_metadata.json'))
    require(folders, 'No existing source-context review found in Drive. Select the Drive account/folder containing your existing artifacts; this launcher never creates a new candidate.')
    def activity(metadata):
        directory=metadata.parent
        receipts=[p for p in directory.glob('decisions-*/provenance.json') if list(p.parent.glob('review_decisions*.csv'))]
        # An annotated/edited checklist or saved decisions take precedence over
        # a freshly generated blank context folder.
        checklist=directory/'human_checklist.csv'
        rows=read_csv(checklist) if checklist.is_file() else []
        annotated=any(any(r.get(k,'').strip() for k in ['reviewer','notes','review_notes']) or
                      any(r.get(k,'Pending').strip().casefold() not in {'','pending'} for k in
                          ['selected_label_visible','watched_all','incident_independence']) for r in rows)
        human_times=[p.stat().st_mtime for p in receipts]+([checklist.stat().st_mtime] if annotated else [])
        return (bool(human_times),max(human_times) if human_times else metadata.stat().st_mtime,str(directory))
    return max(folders,key=activity).parent


def resume_decisions(directory, report):
    """Restore explicit saved answers only; no inference from metadata/checklists."""
    directory=Path(directory)
    receipts=sorted(directory.glob('decisions-*/provenance.json'),key=lambda p:(p.stat().st_mtime,str(p)),reverse=True)
    if not receipts:
        return {},None
    receipt=receipts[0]
    provenance=json.loads(receipt.read_text())
    require(provenance['inputs']==report['inputs'] and provenance['canonical_split_signature']==report['canonical_split_signature'],
            'Latest saved human decisions refer to changed review inputs. All files are preserved; reconcile the snapshots before resuming. No decisions were silently discarded.')
    files=list(receipt.parent.glob('review_decisions*.csv'))
    require(len(files)==1, 'Saved decisions CSV is missing or ambiguous; keep the files and recover the correct saved snapshot')
    rows=read_csv(files[0])
    expected={r['source']:r for r in report['rows']}
    require(len(rows)==12 and len({r['source'] for r in rows})==12 and {r['source'] for r in rows}==set(expected),'Saved decisions sources differ')
    answers={}
    for row in rows:
        identity=expected[row['source']]
        require(all(row[k]==identity[k] for k in ['label','split','selected_clip']) and row['canonical_split_signature']==report['canonical_split_signature'], 'Saved decisions candidate identity differs')
        require(all(row[k] in OPTIONS for k in DECISIONS), 'Unknown saved decision option')
        require(row['decision_origin']=='EXPLICIT HUMAN FORM SUBMISSION; NOT INDEPENDENTLY VERIFIED', 'Unrecognized decision attribution')
        require(row['source_mapping_reviewed']=='False' and row['run_prototype_training']=='False', 'Saved decision flags differ from disabled review workflow')
        answer={k:row[k] for k in (*DECISIONS,'reviewer','review_notes')}
        for k in ['reviewer','review_notes']:
            # Undo only the CSV formula protection applied by our explicit saver.
            if answer[k].startswith("'") and answer[k][1:].lstrip().startswith(('=','+','-','@')):
                answer[k]=answer[k][1:]
        answers[row['source']]=answer
    return answers,str(files[0])


def recovery_message(directory, error):
    recorded=''
    try:
        metadata=json.loads((Path(directory)/'context_metadata.json').read_text())
        recorded=f"Recorded dataset root: {metadata['dataset_root']}\nSaved validation report: {metadata['validation_path']}\n"
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return (f'Review dashboard stopped safely: {error}\nExisting review: {directory}\n'
            'No dataset inspection, preparation, split, or training was run. Existing annotations remain untouched.\n'+recorded+
            'Recovery: confirm the correct Google Drive account and existing review folder. '
            'Read context_metadata.json (dataset_root and validation_path) and the saved validation/inspection manifests for the recorded paths. '
            'If Kaggle cache files disappeared, download the SAME recorded DCSASS version and restore its files at those recorded paths, '
            'then rerun this launcher. Files must pass the existing containment, canonical-identity and SHA-256 checks. '
            'A different download root is not silently substituted. Do not Run all, rerun preparation, regenerate a split, or enable training flags. '
            'If inputs were edited, reconcile the existing saved snapshots rather than deleting annotations.')


def launch_saved_review(output, explicit=None):
    """Called only after Colab setup/Drive mounting; never downloads the dataset."""
    from scripts.review_dashboard import write_dashboard
    directory=None
    try:
        directory=choose_context(output,explicit)
        print('Existing review folder:',directory)
        dashboard=write_dashboard(directory,interactive=True,resume=True)
    except (ValueError,FileNotFoundError,KeyError,OSError) as exc:
        raise RuntimeError(recovery_message(directory or output,exc)) from exc
    print('Dashboard:',dashboard/'review_dashboard.html')
    print('Save writes a new versioned CSV inside this review folder; original annotations remain untouched.')
    print('SOURCE_MAPPING_REVIEWED=False; RUN_PROTOTYPE_TRAINING=False. No training executed.')
    return directory,dashboard
