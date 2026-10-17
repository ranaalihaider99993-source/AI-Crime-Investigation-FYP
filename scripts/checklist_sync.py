"""Preview and explicitly approve exact-source human decision synchronization.

No video semantics, automatic confirmation, training calls or flag mutations.
"""
import csv
import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from scripts.data import file_hash
from scripts.smoke import read_csv, require

FIELD_MAP = {'label_visible':'selected_label_visible','videos_watched':'watched_all',
             'incident_independence':'incident_independence'}


def latest_saved_decision(directory):
    """Select by recorded UTC save time, not mutable Drive filesystem timestamps."""
    candidates=[]
    for receipt in Path(directory).glob('decisions-*/provenance.json'):
        provenance=json.loads(receipt.read_text())
        stamp=datetime.fromisoformat(provenance['saved_at_utc'].replace('Z','+00:00'))
        require(stamp.tzinfo is not None,'Saved decisions timestamp must include timezone')
        candidates.append((stamp,receipt,provenance))
    if not candidates:return None
    candidates.sort(key=lambda item:item[0],reverse=True)
    require(len(candidates)==1 or candidates[0][0]!=candidates[1][0],'Latest decisions have ambiguous equal save times; select/reconcile snapshots before synchronization')
    _,receipt,provenance=candidates[0]
    files=list(receipt.parent.glob('review_decisions*.csv'))
    require(len(files)==1,'Latest saved decisions CSV is missing or ambiguous; no older snapshot was substituted')
    path=files[0]
    require(not path.is_symlink() and not receipt.is_symlink() and path.resolve().is_relative_to(Path(directory).resolve()),'Saved decision files must stay inside this context folder')
    return path,receipt,provenance


def approved_sync_matches(directory, decision_path, provenance, report):
    """Allow resume after OUR approved checklist write without rewriting old receipts."""
    for audit_path in Path(directory).glob('sync-*/sync_audit.json'):
        try:
            audit=json.loads(audit_path.read_text())
            backup=Path(audit['backup_path'])
            if (audit['kind']=='HUMAN_CHECKLIST_SYNC_V1' and audit['explicit_approval'] is True
                and audit['decision_file']==str(decision_path.resolve())
                and audit['decision_sha256']==file_hash(decision_path)
                and audit['decision_provenance_sha256']==file_hash(decision_path.parent/'provenance.json')
                and audit['decision_original_inputs']==provenance['inputs']
                and audit['inputs_after']==report['inputs']
                and audit['canonical_split_signature']==report['canonical_split_signature']
                and backup.resolve().is_relative_to(audit_path.parent.resolve())
                and file_hash(backup)==audit['inputs_before']['checklist_sha256']):
                return True
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return False


def _status(value):
    value=value.strip().casefold()
    return {'pending':'Pending','uncertain':'Uncertain','rejected':'Rejected','confirmed':'Confirmed',
            'yes':'Confirmed','true':'Confirmed','verified':'Confirmed','no':'Rejected','false':'Rejected'}.get(value,'Uncertain')


def checklist_status(rows):
    """Summarize attributed checklist answers only. Never a training approval."""
    sources=[]
    for row in rows:
        statuses={field:_status(row.get(field,'Pending')) for field in FIELD_MAP.values()}
        overall=next((s for s in ['Rejected','Uncertain','Pending'] if s in statuses.values()),'Confirmed')
        reasons=[f'{field}: {status} (human checklist)' for field,status in statuses.items() if status!='Confirmed']
        sources.append(dict(source=row['source'],status=overall,fields=statuses,reasons=reasons,
                            review_notes=row.get('review_notes',''),existing_notes=row.get('notes','')))
    return dict(counts={s:sum(r['status']==s for r in sources) for s in ['Pending','Confirmed','Uncertain','Rejected']},
                sources=sources,remaining_pending_or_uncertain=[r for r in sources if any(s in {'Pending','Uncertain'} for s in r['fields'].values())],
                rejected=[r for r in sources if r['status']=='Rejected'],
                meaning='Human checklist answers only; Confirmed here means all three answers are Confirmed, not training approval or independent verification.',
                human_approval='PENDING',training_executed=False)


def preview_checklist_sync(directory):
    from scripts.review_dashboard import consolidate_review
    from scripts.review_launcher import resume_decisions
    directory=Path(directory).resolve()
    checklist=directory/'human_checklist.csv'
    require(not checklist.is_symlink(),'Checklist must be an ordinary file, not a symlink')
    current=consolidate_review(directory)
    latest=latest_saved_decision(directory)
    require(latest is not None,'No saved human decisions found. Save the review form first')
    decision_path,receipt,provenance=latest
    answers,restored=resume_decisions(directory,current,allow_checklist_edits=True)
    require(Path(restored).resolve()==decision_path.resolve(),'Latest decision snapshot changed while building preview')
    with checklist.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    require(file_hash(checklist)==current['inputs']['checklist_sha256'],'Checklist changed during preview')
    require(len(rows)==len(answers)==12 and len({r['source'] for r in rows})==12 and {r['source'] for r in rows}==set(answers),'Exact checklist/decision source IDs required')
    fields=list(fields)
    for field in [*FIELD_MAP.values(),'reviewer','review_notes']:
        if field not in fields:fields.append(field)
    proposed=[];changes=[]
    for original in rows:
        row=dict(original);answer=answers[row['source']]
        for source_field,target_field in FIELD_MAP.items():
            row[target_field]=answer[source_field]  # Exact explicit saved answer; never a heuristic/default promotion.
        new_reviewer=answer['reviewer'].strip()
        old_reviewer=row.get('reviewer','')
        prior_reviewers=[v[1:] if v.startswith("'") and v[1:].lstrip().startswith(('=','+','-','@')) else v for v in old_reviewer.split('; ')]
        old_unescaped=old_reviewer[1:] if old_reviewer.startswith("'") and old_reviewer[1:].lstrip().startswith(('=','+','-','@')) else old_reviewer
        if new_reviewer and new_reviewer!=old_unescaped and new_reviewer not in prior_reviewers and not old_reviewer.endswith('; '+new_reviewer):
            row['reviewer']=old_reviewer+'; '+new_reviewer if old_reviewer else ("'"+new_reviewer if new_reviewer.lstrip().startswith(('=','+','-','@')) else new_reviewer)
        note=answer['review_notes']
        if note:
            block=f'[Saved human decision: {decision_path.name}; source {row["source"]}]\n{note}'
            old=row.get('review_notes','')
            if block not in old:row['review_notes']=old+('\n\n' if old else '')+block
        for field in fields:
            if original.get(field,'')!=row.get(field,''):
                changes.append(dict(source=row['source'],field=field,before=original.get(field,''),after=row.get(field,''),
                                    basis='Explicit saved human decision; checklist write requires your approval'))
        proposed.append(row)
    return dict(directory=str(directory),decision_file=str(decision_path.resolve()),decision_sha256=file_hash(decision_path),
                decision_provenance_sha256=file_hash(receipt),inputs_before=current['inputs'],
                decision_original_inputs=provenance['inputs'],canonical_split_signature=current['canonical_split_signature'],
                fields=fields,proposed_rows=proposed,changes=changes,status_before=checklist_status(rows),status_after=checklist_status(proposed),
                message='PREVIEW ONLY. No checklist or decision file changed. Confirmed values are copied only from explicit saved human answers after approval.')


def apply_checklist_sync(directory, preview, *, approved=False):
    """Explicit approval is mandatory; recheck everything, back up, then replace."""
    require(approved is True,'Explicit approval is required before writing the checklist')
    directory=Path(directory).resolve();checklist=directory/'human_checklist.csv'
    lock=directory/'.checklist_sync.lock'
    try:
        guard=lock.open('x')
    except FileExistsError as exc:
        raise ValueError('Another checklist sync is in progress; no files changed. If a runtime crashed, inspect the checklist and backups before resolving the stale lock') from exc
    backup=None;temporary=None;written_hash=None;original=None
    try:
        with guard:guard.write('Explicit checklist synchronization in progress')
        fresh=preview_checklist_sync(directory)
        require(fresh==preview,'Preview is stale or altered. Preview again and approve the new proposed changes')
        if not fresh['changes']:
            return dict(message='No changes needed; checklist unchanged',status=fresh['status_after'])
        original=checklist.read_bytes()
        require(hashlib.sha256(original).hexdigest()==fresh['inputs_before']['checklist_sha256'],'Checklist changed after preview')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output=directory/('sync-'+stamp+'-'+uuid.uuid4().hex[:8]);output.mkdir(exist_ok=False)
        backup=output/('human_checklist.backup.'+stamp+'.csv')
        with backup.open('xb') as f:f.write(original)
        require(file_hash(backup)==fresh['inputs_before']['checklist_sha256'],'Checklist backup verification failed')
        buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fresh['fields'])
        writer.writeheader();writer.writerows(fresh['proposed_rows'])
        content=buffer.getvalue().encode();written_hash=hashlib.sha256(content).hexdigest()
        temporary=directory/('.human_checklist.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        latest=latest_saved_decision(directory)
        require(file_hash(checklist)==fresh['inputs_before']['checklist_sha256'] and str(latest[0].resolve())==fresh['decision_file']
                and file_hash(latest[0])==fresh['decision_sha256'] and file_hash(latest[1])==fresh['decision_provenance_sha256'],
                'Review inputs changed before write; preview and approve again')
        os.replace(temporary,checklist)
        from scripts.review_dashboard import consolidate_review
        report=consolidate_review(directory)
        report['checklist_status']=checklist_status(read_csv(checklist))
        audit={**fresh,'kind':'HUMAN_CHECKLIST_SYNC_V1','explicit_approval':True,'backup_path':str(backup),
               'inputs_after':report['inputs'],'approved_at_utc':datetime.now(timezone.utc).isoformat(),'training_executed':False}
        (output/'post_sync_report.json').write_text(json.dumps(report,indent=2))
        (output/'sync_audit.json').write_text(json.dumps(audit,indent=2))
        return dict(message='Approved checklist sync saved. Rerun the launcher before further form saves; both training flags are unchanged.',
                    checklist=str(checklist),backup_path=str(backup),report_path=str(output/'post_sync_report.json'),status=report['checklist_status'])
    except Exception:
        # If our replacement fails validation/reporting, restore only our own bytes.
        if original is not None and written_hash and checklist.is_file() and file_hash(checklist)==written_hash:
            restore=directory/('.human_checklist.restore.'+uuid.uuid4().hex+'.tmp')
            restore.write_bytes(original);os.replace(restore,checklist)
        raise
    finally:
        if temporary and temporary.exists():temporary.unlink()
        lock.unlink(missing_ok=True)


def render_sync_controls(directory):
    """Two distinct Colab actions: read-only preview, then an explicitly checked apply."""
    from google.colab import output
    from IPython.display import JSON
    name='crimescope.sync.'+uuid.uuid4().hex;uid='sync-'+uuid.uuid4().hex;pending={}
    def callback(action,token=None,approved=False):
        if action=='preview':
            preview=preview_checklist_sync(directory);key=uuid.uuid4().hex
            pending.clear();pending[key]=preview
            # Proposed changes/status are public; full original rows stay in kernel memory.
            return JSON(dict(token=key,decision_file=preview['decision_file'],changes=preview['changes'],
                             before=preview['status_before'],after=preview['status_after'],message=preview['message']))
        require(action=='apply' and token in pending,'Preview first; unknown or expired approval token')
        result=apply_checklist_sync(directory,pending[token],approved=approved)
        pending.clear()
        return JSON(result)
    output.register_callback(name,callback)
    return (f'<section id="{uid}"><h2>Synchronize saved human decisions to checklist</h2>'
            '<p>Save any video-form edits first. This sync uses the latest SAVED CSV only. Preview does not write. '
            'Inspect every proposed field change, including any Confirmed answers. Only your explicit approval writes the checklist; '
            'notes are retained and a timestamped backup is made. No training approval is granted.</p>'
            '<button type="button" data-preview>Preview latest saved decisions</button>'
            '<div data-preview-output></div>'
            '<label><input type="checkbox" data-approve disabled> I reviewed this preview and explicitly approve these checklist changes.</label>'
            '<button type="button" data-apply disabled>Apply approved checklist changes</button><p data-sync-status role="status">Not applied.</p></section>'
            '<script>(()=>{const root=document.getElementById('+json.dumps(uid)+');const callback='+json.dumps(name)+';'+r'''
let token=null;const preview=root.querySelector('[data-preview]'),apply=root.querySelector('[data-apply]'),approve=root.querySelector('[data-approve]'),status=root.querySelector('[data-sync-status]');
async function invoke(action,token,approved){
 if(!window.google?.colab?.kernel?.invokeFunction)throw Error('Use the live authorized Colab runtime.');
 const result=await google.colab.kernel.invokeFunction(callback,[action,token,approved],{});
 const value=result.data?.['application/json'];if(!value)throw Error('No receipt returned.');return value;
}
approve.addEventListener('change',()=>{apply.disabled=!(token&&approve.checked);});
preview.addEventListener('click',async()=>{
 token=null;approve.checked=false;approve.disabled=true;apply.disabled=true;preview.disabled=true;
 try{const result=await invoke('preview',null,false);token=result.token;
 const holder=root.querySelector('[data-preview-output]');holder.replaceChildren();
 const text=document.createElement('p');text.textContent=result.message+' Latest saved decisions: '+result.decision_file;holder.append(text);
 const counts=document.createElement('p');counts.textContent='Before: '+JSON.stringify(result.before.counts)+' → Proposed: '+JSON.stringify(result.after.counts);holder.append(counts);
 const table=document.createElement('table');table.style.borderCollapse='collapse';
 const header=document.createElement('tr');for(const label of ['Source','Field','Current checklist','Proposed saved human answer']){const cell=document.createElement('th');cell.textContent=label;header.append(cell);}table.append(header);
 for(const change of result.changes){const row=document.createElement('tr');for(const value of [change.source,change.field,change.before,change.after]){const cell=document.createElement('td');cell.textContent=value;cell.style.cssText='border:1px solid #aaa;padding:6px;white-space:pre-wrap';row.append(cell);}table.append(row);}holder.append(table);
 const remaining=document.createElement('pre');remaining.style.whiteSpace='pre-wrap';remaining.textContent='Proposed remaining Pending/Uncertain sources and reasons:\n'+JSON.stringify(result.after.remaining_pending_or_uncertain,null,2);holder.append(remaining);
 approve.disabled=false;status.textContent='Preview only. Review changes and explicitly approve to write.';
 }catch(error){status.textContent='NOT APPLIED: '+error.message;}finally{preview.disabled=false;}
});
apply.addEventListener('click',async()=>{
 if(!token||!approve.checked)return;
 apply.disabled=true;preview.disabled=true;
 try{const result=await invoke('apply',token,true);token=null;approve.checked=false;approve.disabled=true;
 status.style.whiteSpace='pre-wrap';status.textContent=JSON.stringify(result,null,2);
 }catch(error){token=null;approve.checked=false;approve.disabled=true;status.textContent='NOT APPLIED: '+error.message+' Preview again before retrying.';}
 finally{preview.disabled=false;}
});})();</script>''')
