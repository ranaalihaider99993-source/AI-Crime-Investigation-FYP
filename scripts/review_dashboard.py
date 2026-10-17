"""Consolidate saved evidence and user statements; no semantic video inference."""
import csv
import html
import json
import re
import uuid
from pathlib import Path
from scripts.review_paths import ensure_review_dataset, temporal_location
from scripts.data import file_hash
from scripts.smoke import read_csv, require

OBSERVATION_056 = 'A person inside a shop appears to pass items to another person; the scene appears to be at night.'


def source_056_observation(sources):
    matches = [s for s in sources if re.search(r'(?<!\d)056(?:_x264)?$', s, re.I)]
    return dict(source_reference='056', matched_source=matches[0] if len(matches)==1 else None,
                candidate_sources=matches, attribution='USER-REPORTED; NOT VERIFIED', text=OBSERVATION_056,
                resolution='Matched unique source number; confirm identity with user' if len(matches)==1 else
                'UNRESOLVED: source 056 absent or ambiguous; do not assign to a class/source automatically')


def consolidate_review(context_directory):
    directory = Path(context_directory)
    metadata = directory/'context_metadata.json'
    checklist = directory/'human_checklist.csv'
    saved = json.loads(metadata.read_text())
    current = ensure_review_dataset(directory)
    groups = {g['source']:g for g in current['groups']}
    require(len(groups)==12, 'Exactly 12 unique review sources required')
    human = read_csv(checklist)
    require(len(human)==12 and {r['source'] for r in human}==set(groups), 'Checklist sources missing, duplicated, or unexpected')
    for row in human:
        g = groups[row['source']]
        require(all(row[k]==g[k] for k in ['label','split']) and row['selected_clip']==g['selected'], 'Checklist identity/split changed')
        expected = '; '.join(c['canonical_clip'] for c in g['videos'] if c['context_only'])
        require(row['context_clips']==expected, 'Checklist context identities changed')
    human = {r['source']:r for r in human}
    observation = source_056_observation(groups)
    flags = {source:list(g['warnings']) for source,g in groups.items()}
    pairs = []
    # Exact bytes are technical evidence only; incident identity remains a human judgment.
    values = list(groups.values())
    for i,left in enumerate(values):
        for right in values[i+1:]:
            matches = [(a['filename'], b['filename']) for a in left['videos'] for b in right['videos'] if a['sha256']==b['sha256']]
            if matches:
                pairs.append(dict(left_source=left['source'], right_source=right['source'], kind='EXACT_BYTES',
                                  cross_split=left['split']!=right['split'], provenance='SHA-256 of verified context/selected files',
                                  detail=str(matches)))
    temporal_path = temporal_location(directory)
    temporal_status = 'NOT AVAILABLE; no visual similarity or motion analysis performed by this dashboard'
    inputs = dict(context_metadata_sha256=file_hash(metadata), checklist_sha256=file_hash(checklist))
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text())
        selected_hashes = {c['canonical_clip']:c['sha256'] for g in values for c in g['videos'] if not c['context_only']}
        require(temporal['canonical_split_signature']==saved['canonical_split_signature'] and
                {c['canonical_clip']:c['sha256'] for c in temporal['clips']}==selected_hashes,
                'Temporal evidence belongs to different candidate/content')
        inputs['temporal_review_sha256'] = file_hash(temporal_path)
        temporal_status = 'Existing temporal utility: '+temporal['technical_status']+'; heuristic motion/similarity only, no semantic understanding'
        for clip in temporal['clips']:
            require(clip['source'] in groups and clip['split']==groups[clip['source']]['split'], 'Temporal source/split differs')
            flags[clip['source']].extend('TEMPORAL HEURISTIC: '+w for w in clip.get('warnings',[]))
            if clip.get('error'):
                flags[clip['source']].append('TEMPORAL ERROR: '+clip['error'])
        for pair in temporal['pairs']:
            left,right = temporal['clips'][pair['left']],temporal['clips'][pair['right']]
            pairs.append(dict(left_source=left['source'],right_source=right['source'],kind=pair['kind'],
                              cross_split=left['split']!=right['split'],provenance='Existing temporal review (uncertain scene similarity)',detail=pair['detail']))
    # Only exact full source IDs in human notes can produce attributed pair links.
    # Free text and short/ambiguous numeric references remain unresolved warnings.
    for source,h in human.items():
        related = h.get('possible_related_sources','')
        for other in groups:
            if other != source and re.search(r'(?<![a-z0-9_])'+re.escape(other)+r'(?![a-z0-9_])', related, re.I):
                pairs.append(dict(left_source=source,right_source=other,kind='USER_REPORTED_POSSIBLE_RELATION',
                                  cross_split=groups[source]['split']!=groups[other]['split'],
                                  provenance='Human checklist; not independently verified',detail=related))
    rows = []
    for source,g in groups.items():
        h = human[source]
        warnings = flags[source]
        if any(c.get('reported_low_motion') for c in g['videos']):
            warnings.append('USER-REPORTED LOW MOTION: inspect original videos; not an automatic label judgment')
        label_status = h.get('selected_label_visible','PENDING')
        if label_status.strip().casefold() not in {'yes','true','confirmed','verified'}:
            warnings.append('LABEL REVIEW REQUIRED: selected-clip label absent, uncertain, or not yet reviewed (checklist: '+label_status+')')
        if h.get('label_visible_only_in_context','').strip().casefold() in {'yes','true'}:
            warnings.append('USER CHECKLIST: label may be visible only in context; selected clip needs verification')
        related = h.get('possible_related_sources','').strip()
        if related and related.casefold() not in {'pending','no','none','false'}:
            warnings.append('USER CHECKLIST: possible related sources: '+related+'; compare split assignments manually')
        findings = [p for p in pairs if source in (p['left_source'],p['right_source'])]
        if findings:
            warnings.append('POTENTIALLY RELATED FOOTAGE: investigate listed pairs; incident identity unverified')
        notes = '\n'.join(v for v in [h.get('notes',''),h.get('review_notes','')] if v)
        if observation['matched_source']==source:
            notes += ('\n' if notes else '')+'USER-REPORTED (not verified ground truth): '+OBSERVATION_056
        rows.append(dict(source=source,label=g['label'],split=g['split'],selected_clip=g['selected'],
                         clips='; '.join(c['role']+': '+c['filename'] for c in g['videos']),
                         temporal_relationship=saved['ordering'],flags=' | '.join(warnings),
                         possible_cross_split_overlap='FLAGGED; HUMAN VERIFICATION REQUIRED' if any(p['cross_split'] for p in findings) else
                         'NOT ESTABLISHED; absence of flags does not prove independence',
                         pair_findings=json.dumps(findings),review_notes=notes,watched_all=h.get('watched_all','PENDING'),
                         selected_label_visible=label_status,incident_independence=h.get('incident_independence','PENDING'),
                         reviewer=h.get('reviewer',''),human_approval='PENDING'))
    from scripts.checklist_sync import checklist_status
    return dict(rows=rows,pairs=pairs,observation_056=observation,inputs=inputs,checklist_status=checklist_status(list(human.values())),
                canonical_split_signature=saved['canonical_split_signature'],
                runtime_dataset_root=current['dataset_root'],recorded_dataset_root=saved['dataset_root'],
                technical_checks='PASS (identity checks only): candidate/context identities, file SHA-256, checklist identities and source-disjoint split assignments',
                technical_status='BLOCKED' if any(p['kind'] in {'EXACT_BYTES','exact_file_duplicate'} and p['cross_split'] for p in pairs) or temporal_status.startswith('Existing temporal utility: FAIL') else 'IDENTITY CHECKS PASS; HUMAN REVIEW PENDING',
                temporal_evidence=temporal_status,human_approval='PENDING',training_executed=False,
                source_mapping_reviewed=False,run_prototype_training=False,
                limitations='This dashboard reads metadata, hashes and saved heuristic findings; it has not watched or understood crime content. No semantic video-analysis method is implemented. All labels and incident independence require human verification. Context is not training data.')


def write_dashboard(context_directory, interactive=False, resume=False, analyze=False):
    report = consolidate_review(context_directory)
    output = Path(context_directory)/('dashboard-'+uuid.uuid4().hex[:12])
    output.mkdir(exist_ok=False)
    if analyze:
        from scripts.context_analysis import analyze_context
        plan=ensure_review_dataset(context_directory)
        report['automatic_analysis']=analyze_context(plan,output/'automatic_analysis')
        report['automatic_analysis_dir']=str(output/'automatic_analysis')
    (output/'review_report.json').write_text(json.dumps(report,indent=2))
    with (output/'review_report.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(report['rows'][0]))
        writer.writeheader()
        # Prevent spreadsheet formula execution from free-text user notes.
        for row in report['rows']:
            writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in row.items()})
    escape = lambda value:html.escape(str(value),quote=True)
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>CrimeScope human review</title>',
             '<style>body{font:16px sans-serif;margin:2em}section{border:2px solid #b76a00;padding:1em;margin:1em 0}td,th{border:1px solid #aaa;padding:.5em;vertical-align:top}pre{white-space:pre-wrap}</style></head><body>',
             '<h1>CrimeScope consolidated human review — PENDING</h1>',
             '<p>'+escape(report['technical_status'])+'</p><p>'+escape(report['technical_checks'])+'</p><p>'+escape(report['limitations'])+'</p>',
             '<p>'+escape(report['temporal_evidence'])+'</p>',
             '<p>Both training flags remain False. Use Notebook 02 source-context players to watch the original videos. This dashboard contains metadata, not video copies.</p>',
             '<h2>Current checklist status (human answers, not approval)</h2><pre>'+escape(json.dumps(report['checklist_status'],indent=2))+'</pre>',
             '<h2>User observation: source 056</h2><pre>'+escape(json.dumps(report['observation_056'],indent=2))+'</pre>',
             '<h2>Pair findings (not verified incidents)</h2><pre>'+escape(json.dumps(report['pairs'],indent=2))+'</pre>',
             '<p>Edit human_checklist.csv in the parent context folder, then rerun the dashboard cell. Previous dashboards and checklist notes are preserved.</p>']
    if analyze:
        evidence=report['automatic_analysis']
        parts.append('<h2>OpenCV automatic technical review: '+escape(evidence['technical_status'])+'</h2><p>'+escape(evidence['limitations'])+'</p><pre>'+escape(json.dumps(evidence['priorities'],indent=2))+'</pre>')
    for row in report['rows']:
        parts.append('<section><h2>'+escape(row['source'])+' · '+escape(row['label'])+' · '+escape(row['split'])+'</h2><table>')
        parts.extend('<tr><th>'+escape(k)+'</th><td><pre>'+escape(v)+'</pre></td></tr>' for k,v in row.items() if k not in {'source','label','split'})
        parts.append('</table></section>')
    parts.append('</body></html>')
    page=''.join(parts)
    if interactive:
        from scripts.review_interface import render_review_interface, register_drive_callback
        plan=ensure_review_dataset(context_directory)
        callback=register_drive_callback(context_directory,report)
        initial,restored_from={},None
        if resume:
            from scripts.review_launcher import resume_decisions
            initial,restored_from=resume_decisions(context_directory,report)
        interface=render_review_interface(report,plan,callback,initial,restored_from)
        from scripts.checklist_sync import render_sync_controls
        interface+=render_sync_controls(context_directory)
        # Keep detailed metadata below the form, collapsed for efficient video review.
        start=page.index('<body>')+len('<body>')
        page=page[:start]+interface+'<details><summary>Technical evidence and consolidated metadata</summary>'+page[start:].replace('</body>', '</details></body>')
    (output/'review_dashboard.html').write_text(page)
    return output
