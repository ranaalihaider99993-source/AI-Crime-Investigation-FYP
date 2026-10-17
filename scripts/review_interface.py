"""Explicit human decisions for verified review snapshots; never a training gate."""
import csv
import html
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from scripts.context_review import context_players
from scripts.smoke import require

OPTIONS = ('Pending','Confirmed','Uncertain','Rejected')
DECISIONS = ('label_visible','videos_watched','incident_independence')


def save_review_decisions(context_directory, snapshot, decisions):
    """Colab callback target. Validate identities anew; create a separate CSV."""
    from scripts.review_dashboard import consolidate_review
    current = consolidate_review(context_directory)
    require(current['inputs']==snapshot['inputs'] and current['canonical_split_signature']==snapshot['canonical_split_signature'],
            'Review inputs changed. Reopen dashboard before saving decisions')
    require(isinstance(decisions,list) and len(decisions)==12, 'Exactly 12 decision rows required')
    expected = {r['source']:r for r in current['rows']}
    require(all(isinstance(r,dict) and isinstance(r.get('source'),str) for r in decisions), 'Invalid decision rows')
    require(len({r['source'] for r in decisions})==12 and {r['source'] for r in decisions}==set(expected), 'Decision sources changed or duplicated')
    rows=[]
    stamp=datetime.now(timezone.utc).isoformat()
    for decision in decisions:
        require(set(decision)=={'source',*DECISIONS,'reviewer','review_notes'}, 'Unexpected decision fields')
        require(all(decision[k] in OPTIONS for k in DECISIONS), 'Unknown decision option')
        require(all(isinstance(decision[k],str) and len(decision[k])<=10000 for k in ['reviewer','review_notes']), 'Invalid reviewer/notes')
        bound=expected[decision['source']]
        rows.append(dict(source=bound['source'],label=bound['label'],split=bound['split'],selected_clip=bound['selected_clip'],
                         **{k:decision[k] for k in (*DECISIONS,'reviewer','review_notes')},
                         decision_origin='EXPLICIT HUMAN FORM SUBMISSION; NOT INDEPENDENTLY VERIFIED',
                         saved_at_utc=stamp,canonical_split_signature=current['canonical_split_signature'],
                         source_mapping_reviewed=False,run_prototype_training=False))
    output=Path(context_directory)/('decisions-'+uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    path=output/'review_decisions.csv'
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in row.items()})
    (output/'provenance.json').write_text(json.dumps(dict(inputs=current['inputs'],
        canonical_split_signature=current['canonical_split_signature'],saved_at_utc=stamp,
        human_approval='NOT AUTOMATICALLY GRANTED',training_executed=False,
        source_mapping_reviewed=False,run_prototype_training=False),indent=2))
    return dict(saved_csv=str(path),message='Saved human decisions. Original checklist unchanged. Both training flags remain False.')


def render_review_interface(report, plan, callback_name=None):
    """Embedded original videos plus HTML controls. No playback-based auto-confirmation."""
    esc=lambda value:html.escape(str(value),quote=True)
    uid='review-'+uuid.uuid4().hex
    rows={r['source']:r for r in report['rows']}
    parts=[f'<div id="{uid}"><h2>Watch and record human decisions</h2>',
        '<p>Controls start Pending. Confirmed means you affirm the stated claim; Rejected means you disagree; Uncertain means unresolved. '
        'For watched, affirm only after watching ALL available videos in that source group. For incident independence, compare against ALL selected sources. '
        'These answers never enable training. Notes include attributed prior observations, not verified ground truth.</p>',
        '<p>Save decisions to Drive using the button below. Save before rerunning the cell or leaving this runtime. '
        'If a video codec will not play, leave watched Pending/Uncertain and use your exported originals with a compatible player.</p>']
    for group in plan['groups']:
        row=rows[group['source']]
        low_motion=any(c.get('reported_low_motion') for c in group['videos']) or any(s in row['flags'].lower() for s in ['little change','low motion','static view'])
        priority=group['label']=='Fighting' or (group['label']=='Vandalism' and low_motion)
        parts.append(f'<section data-review-source="{esc(group["source"])}" style="border:3px solid {"#b03000" if priority else "#aaa"};padding:12px;margin:16px 0">')
        if priority:
            parts.append('<strong>CLOSER INSPECTION: '+('Fighting — few independent sources' if group['label']=='Fighting' else 'Low-motion Vandalism — label evidence may be unclear')+'</strong>')
        parts.append('<p>'+esc(row['flags'])+'</p><p>'+esc(row['possible_cross_split_overlap'])+'</p>')
        # The existing player checks containment and content hashes immediately before embedding.
        parts.extend(context_players(dict(plan,groups=[group])))
        for field,label in [('label_visible','Crime label visible in SELECTED clip itself'),('videos_watched','All available videos in this source group watched'),('incident_independence','Incident independent of the OTHER selected sources')]:
            parts.append(f'<label style="display:block;margin:8px">{esc(label)} <select data-field="{field}" aria-label="{esc(label)}">'+''.join(f'<option>{v}</option>' for v in OPTIONS)+'</select></label>')
        parts.append('<label>Reviewer <input data-field="reviewer" value="'+esc(row['reviewer'])+'"></label>')
        parts.append('<label style="display:block">Review notes / related source IDs <textarea data-field="review_notes" rows="4" style="width:95%">'+esc(row['review_notes'])+'</textarea></label></section>')
    parts.append('<button type="button" data-save>Save review decisions to Google Drive</button><p role="status" data-status>Not saved in this session.</p></div>')
    # Only generated identifiers reach JS; all reviewer text is escaped HTML and read via .value.
    parts.append('<script>(function(){const root=document.getElementById('+json.dumps(uid)+');const callback='+json.dumps(callback_name)+';'+r'''
const button=root.querySelector('[data-save]'),status=root.querySelector('[data-status]');
button.addEventListener('click',async()=>{
  if(!callback || !window.google?.colab?.kernel?.invokeFunction){
    status.textContent='Drive save unavailable here. Run the dashboard cell in your authorized Colab runtime; keep this form open.';return;
  }
  const rows=Array.from(root.querySelectorAll('[data-review-source]')).map(section=>{
    const row={source:section.dataset.reviewSource};
    section.querySelectorAll('[data-field]').forEach(control=>{row[control.dataset.field]=control.value;});return row;
  });
  button.disabled=true;status.textContent='Saving to a new Drive CSV...';
  try {
    const response=await google.colab.kernel.invokeFunction(callback,[rows],{});
    const result=response.data?.['application/json'];
    if(!result?.saved_csv)throw new Error('No save receipt returned; check Colab runtime connection.');
    status.textContent=result.message+' CSV: '+result.saved_csv;
  } catch(error){status.textContent='NOT SAVED: '+error.message+' Retry after fixing the runtime/path issue; keep your form open.';}
  finally {button.disabled=false;}
});})();</script>''')
    return ''.join(parts)


def register_drive_callback(context_directory, snapshot):
    """Only explicit button clicks invoke this authorized Colab-to-Drive writer."""
    from google.colab import output
    from IPython.display import JSON
    name='crimescope.review.'+uuid.uuid4().hex
    def save(decisions):
        return JSON(save_review_decisions(context_directory,snapshot,decisions))
    output.register_callback(name,save)
    return name
