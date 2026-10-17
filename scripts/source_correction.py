"""Explicit, byte-preserving single-source correction; no training or inference."""
import csv
import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from scripts.data import file_hash
from scripts.smoke import require
from scripts.checklist_sync import latest_saved_decision, _status

SOURCE='vandalism019_x264'
NOTE=('A person is visibly writing on a wall, but authorization is unknown. '
      'The footage does not conclusively establish vandalism, so the label remains uncertain.')


def patch_record(content, changes):
    """Replace ONLY the exact target CSV record; all other bytes are untouched."""
    stream=io.BytesIO(content)
    def lines():
        first=True
        while line:=stream.readline():
            yield line.decode('utf-8-sig' if first else 'utf-8')
            first=False
    reader=csv.reader(lines());header=next(reader)
    require(len(header)==len(set(header)) and 'source' in header,'Invalid/duplicate CSV columns')
    matches=[]
    while True:
        start=stream.tell()
        try:values=next(reader)
        except StopIteration:break
        end=stream.tell()
        if not values:continue
        require(len(values)==len(header),'Malformed CSV record')
        row=dict(zip(header,values))
        if row['source']==SOURCE:matches.append((start,end,row))
    require(len(matches)==1,'Exactly one exact vandalism019_x264 source record is required')
    start,end,before=matches[0]
    require(set(changes)<=set(header),'Correction cannot add/regenerate checklist columns')
    after=dict(before,**changes)
    if after==before:return content,before,after
    old_record=content[start:end]
    ending='\r\n' if old_record.endswith(b'\r\n') else '\n' if old_record.endswith(b'\n') else ''
    buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator=ending or '\n')
    writer.writerow([after[k] for k in header]);encoded=buffer.getvalue().encode('utf-8')
    if not ending:encoded=encoded[:-1]
    return content[:start]+encoded+content[end:],before,after


def corrected_notes(old):
    marker='[User-requested correction: '+SOURCE+']\n'+NOTE
    return old if marker in old else old+('\n\n' if old else '')+marker


def preview_source_correction(directory):
    from scripts.review_dashboard import consolidate_review
    from scripts.review_launcher import resume_decisions
    directory=Path(directory).resolve();checklist=directory/'human_checklist.csv'
    require(not checklist.is_symlink(),'Checklist must not be a symlink')
    report=consolidate_review(directory)
    require(any(r['source']==SOURCE for r in report['rows']),'Target source is not in this existing candidate; no source substitution allowed')
    latest=latest_saved_decision(directory);require(latest is not None,'Existing saved decisions are required to prevent an older decision reverting this correction')
    decisions,receipt,provenance=latest
    resume_decisions(directory,report,allow_checklist_edits=True)
    original=checklist.read_bytes();decision_bytes=decisions.read_bytes()
    require(hashlib.sha256(original).hexdigest()==report['inputs']['checklist_sha256'],'Checklist changed during preview')
    _,before,_=patch_record(original,{})
    _,saved_before,_=patch_record(decision_bytes,{})
    require(_status(before['selected_label_visible']) in {'Confirmed','Uncertain'} and saved_before['label_visible'] in {'Confirmed','Uncertain'},'Label is no longer Confirmed/Uncertain; inspect newer decisions before correction')
    require(_status(before['incident_independence'])=='Pending' and saved_before['incident_independence']=='Pending','Incident independence is not Pending; inspect the conflicting decision rather than overwriting it')
    note_field='review_notes' if 'review_notes' in before else 'notes'
    require(note_field in before,'Existing checklist has no notes column')
    _,_,after=patch_record(original,dict(selected_label_visible='Uncertain',**{note_field:corrected_notes(before[note_field])}))
    saved_after=dict(saved_before,label_visible='Uncertain',review_notes=corrected_notes(saved_before['review_notes']))
    return dict(directory=str(directory),source=SOURCE,inputs_before=report['inputs'],
                canonical_split_signature=report['canonical_split_signature'],decision_file=str(decisions.resolve()),
                decision_sha256=file_hash(decisions),decision_provenance_sha256=file_hash(receipt),
                before=before,after=after,saved_before=saved_before,saved_after=saved_after,
                saved_decision_changes=[dict(field=k,before=saved_before[k],after=saved_after[k]) for k in saved_before if saved_before[k]!=saved_after[k]],
                changes=[dict(field=k,before=before[k],after=after[k]) for k in before if before[k]!=after[k]],
                message='PREVIEW ONLY: single exact-source label downgrade and appended user observation. Watched and independence values are unchanged. No approval/training inferred.')


def apply_source_correction(directory, preview, *, approved=False):
    require(approved is True,'Explicit approval is required')
    directory=Path(directory).resolve();checklist=directory/'human_checklist.csv';lock=directory/'.checklist_sync.lock'
    with lock.open('x') as guard:guard.write('Single-source correction')
    temporary=None;written=None;original=None;new_decision_dir=None;audit_dir=None;committed=False
    try:
        fresh=preview_source_correction(directory)
        require(fresh==preview,'Correction preview is stale or altered; preview and approve again')
        if not fresh['changes'] and fresh['saved_after']==fresh['saved_before']:
            return dict(message='Correction already present; no files changed',source=SOURCE)
        original=checklist.read_bytes()
        prior_decision_bytes=Path(fresh['decision_file']).read_bytes()
        require(hashlib.sha256(prior_decision_bytes).hexdigest()==fresh['decision_sha256'],'Saved decisions changed after preview')
        content,_,_=patch_record(original,{k:v for k,v in fresh['after'].items() if fresh['before'][k]!=v})
        now=datetime.now(timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
        audit_dir=directory/('correction-'+stamp+'-'+uuid.uuid4().hex[:8]);audit_dir.mkdir(exist_ok=False)
        backup=audit_dir/('human_checklist.backup.'+stamp+'.csv')
        with backup.open('xb') as f:f.write(original)
        require(file_hash(backup)==fresh['inputs_before']['checklist_sha256'],'Original checklist backup hash differs')
        written=hashlib.sha256(content).hexdigest()
        temporary=directory/('.source-correction.'+uuid.uuid4().hex+'.tmp');temporary.write_bytes(content)
        latest=latest_saved_decision(directory)
        require(now>datetime.fromisoformat(latest[2]['saved_at_utc'].replace('Z','+00:00')),'Saved decision time is ahead of this runtime; resolve the clock conflict before correction')
        require(file_hash(checklist)==fresh['inputs_before']['checklist_sha256'] and str(latest[0].resolve())==fresh['decision_file'] and
                file_hash(latest[0])==fresh['decision_sha256'] and file_hash(latest[1])==fresh['decision_provenance_sha256'],'Review files changed after preview')
        os.replace(temporary,checklist)
        from scripts.review_dashboard import consolidate_review
        updated=consolidate_review(directory)
        # New decision version: patch the SAME target only; leave all prior files
        # and every non-target decision record byte-for-byte intact.
        edits={k:v for k,v in fresh['saved_after'].items() if fresh['saved_before'][k]!=v}
        edits['saved_at_utc']=now.isoformat()
        decision_content,_,_=patch_record(prior_decision_bytes,edits)
        pending_dir=directory/('decisions-'+uuid.uuid4().hex);pending_dir.mkdir(exist_ok=False)
        new_decision_dir=pending_dir
        new_csv=new_decision_dir/('review_decisions_'+stamp+'_'+uuid.uuid4().hex[:8]+'.csv')
        new_csv.write_bytes(decision_content)
        provenance=dict(inputs=updated['inputs'],canonical_split_signature=updated['canonical_split_signature'],saved_at_utc=now.isoformat(),
                        human_approval='NOT AUTOMATICALLY GRANTED',training_executed=False,source_mapping_reviewed=False,run_prototype_training=False,
                        correction_source=SOURCE,explicit_approval=True,prior_decision_file=fresh['decision_file'],prior_decision_sha256=fresh['decision_sha256'])
        audit={**fresh,'explicit_approval':True,'applied_at_utc':now.isoformat(),'backup_path':str(backup),
               'inputs_after':updated['inputs'],'new_decision_file':str(new_csv),'new_decision_sha256':file_hash(new_csv)}
        latest=latest_saved_decision(directory)
        require(str(latest[0].resolve())==fresh['decision_file'] and file_hash(latest[0])==fresh['decision_sha256'] and file_hash(latest[1])==fresh['decision_provenance_sha256'] and file_hash(checklist)==written,'Review files changed during correction; no new decision version published')
        (audit_dir/'correction_audit.json').write_text(json.dumps(audit,indent=2))
        (audit_dir/'post_correction_report.json').write_text(json.dumps(updated,indent=2))
        # Publish the receipt last so incomplete versions are never selected.
        (new_decision_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
        committed=True
        return dict(message='Approved single-source correction applied. Rerun the launcher before further form saves. Training flags unchanged.',
                    source=SOURCE,backup_path=str(backup),decision_file=str(new_csv),audit_path=str(audit_dir/'correction_audit.json'),
                    checklist=str(checklist),selected_label_visible='Uncertain',watched_all=fresh['after']['watched_all'],incident_independence=fresh['after']['incident_independence'])
    except Exception:
        if original is not None and written and checklist.is_file() and file_hash(checklist)==written:
            restore=directory/('.correction-restore.'+uuid.uuid4().hex+'.tmp');restore.write_bytes(original);os.replace(restore,checklist)
        # Remove only our unpublished failed version, never existing decisions.
        if new_decision_dir and not committed:
            for name in ['provenance.json',new_csv.name]:
                (new_decision_dir/name).unlink(missing_ok=True)
            new_decision_dir.rmdir()
        raise
    finally:
        if temporary:temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def render_source_correction(directory):
    from google.colab import output
    from IPython.display import JSON
    name='crimescope.correct.'+uuid.uuid4().hex;uid='correct-'+uuid.uuid4().hex;pending={}
    def callback(action,token=None,approved=False):
        if action=='preview':
            preview=preview_source_correction(directory);key=uuid.uuid4().hex;pending.clear();pending[key]=preview
            return JSON(dict(token=key,**preview))
        require(action=='apply' and token in pending,'Preview first; unknown/expired correction token')
        result=apply_source_correction(directory,pending[token],approved=approved);pending.clear();return JSON(result)
    output.register_callback(name,callback)
    return (f'<section id="{uid}"><h2>Correct vandalism019_x264 only</h2>'
            '<p>User-reported observation, not verified ground truth: '+NOTE+'</p>'
            '<p>Only this source is patched. Previous notes/decisions are retained. Watched stays exactly as recorded; independence must already be Pending. Save other form edits before previewing.</p>'
            '<button data-preview>Preview source correction</button><pre data-diff style="white-space:pre-wrap"></pre>'
            '<label><input type="checkbox" data-approve disabled> I explicitly approve only this source correction.</label>'
            '<button data-apply disabled>Apply approved source correction</button><pre data-status style="white-space:pre-wrap">Not applied.</pre></section>'
            '<script>(()=>{const root=document.getElementById('+json.dumps(uid)+');const callback='+json.dumps(name)+';'+r'''
let token=null;const preview=root.querySelector('[data-preview]'),approve=root.querySelector('[data-approve]'),apply=root.querySelector('[data-apply]'),status=root.querySelector('[data-status]');
async function invoke(action,approved){if(!window.google?.colab?.kernel?.invokeFunction)throw Error('Run in the live authorized Colab runtime.');const r=await google.colab.kernel.invokeFunction(callback,[action,token,approved],{});const v=r.data?.['application/json'];if(!v)throw Error('No receipt');return v;}
approve.onchange=()=>{apply.disabled=!(token&&approve.checked);};
preview.onclick=async()=>{token=null;approve.checked=false;approve.disabled=true;apply.disabled=true;preview.disabled=true;
try{const p=await invoke('preview',false);token=p.token;root.querySelector('[data-diff]').textContent=JSON.stringify({message:p.message,source:p.source,changes:p.changes,saved_decision_changes:p.saved_decision_changes,watched_all:p.after.watched_all,saved_videos_watched:p.saved_after.videos_watched,incident_independence:p.after.incident_independence,prior_decision_file:p.decision_file},null,2);approve.disabled=false;status.textContent='Preview only. No files changed.';}
catch(e){status.textContent='NOT APPLIED: '+e.message;}finally{preview.disabled=false;}};
apply.onclick=async()=>{if(!token||!approve.checked)return;apply.disabled=true;preview.disabled=true;
try{const receipt=await invoke('apply',true);token=null;approve.checked=false;approve.disabled=true;status.textContent=JSON.stringify(receipt,null,2);}
catch(e){token=null;approve.checked=false;approve.disabled=true;status.textContent='NOT APPLIED: '+e.message+' Preview again.';}finally{preview.disabled=false;}};
})();</script>''')
