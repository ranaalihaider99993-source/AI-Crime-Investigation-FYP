"""Conservative late fusion of existing CNN scores; no inferred visual events."""
import math
from scripts.data import CLASSES

METHOD = ('Exact-byte duplicate files count once. Average unique file scores within each '
          'modality, then equally average the available image/video modality means. '
          'No learned weights or probability calibration; rankings use all four classes.')

# Category-conditioned writing templates, not detections or learned narratives.
# The conditional requirements must never be replaced with asserted observations.
SCENARIO_TEMPLATES = {
    'Burglary': dict(title='Possible property-entry incident',
        explanation='If human review establishes unauthorized entry to a property, an entry-related incident could be considered. The Burglary score does not establish that entry happened.',
        uncertainty='Entry, authorization and intent are unverified; ordinary access remains possible.'),
    'Stealing': dict(title='Possible unauthorized removal of property',
        explanation='If human review establishes that property was taken without permission, a theft-related interpretation could be considered. The Stealing score does not show that any object was taken.',
        uncertainty='Ownership, permission and any transfer of property are unknown; lawful handling remains possible.'),
    'Fighting': dict(title='Possible physical confrontation',
        explanation='If human review establishes hostile physical contact, a confrontation-related interpretation could be considered. The Fighting score does not establish that contact or aggression occurred.',
        uncertainty='Contact, aggression and context are unverified; nonviolent activity remains possible.'),
    'Vandalism': dict(title='Possible property-damage incident',
        explanation='If human review establishes deliberate damage to property, a damage-related interpretation could be considered. The Vandalism score does not establish damage or intent.',
        uncertainty='Damage, cause and intent are unknown; accidental damage or an undamaged scene remain possible.')
}


def aggregate(records):
    if not records: raise ValueError('At least one analyzed file is required')
    groups = {}
    seen = set()
    for record in records:
        if record['kind'] not in {'image','video'}: raise ValueError('Unsupported modality')
        key = (record['kind'],record['sha256'])
        if key in seen: continue
        seen.add(key)
        scores = record['scores']
        if [s['label'] for s in scores] != CLASSES:
            raise ValueError('Incident class order mismatch')
        values = [s['score'] for s in scores]
        if any(type(v) not in {int,float} or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1)>1e-4:
            raise ValueError('Invalid incident score vector')
        groups.setdefault(record['kind'],[]).append(values)
    means = {kind:[sum(row[i] for row in rows)/len(rows) for i in range(4)] for kind,rows in groups.items()}
    combined = [sum(row[i] for row in means.values())/len(means) for i in range(4)]
    ranked = sorted(range(4),key=lambda i:-combined[i])
    weak = combined[ranked[0]] < .5
    close = combined[ranked[0]]-combined[ranked[1]] < .1
    scenarios = []
    for rank,i in enumerate(ranked[:3],1):
        contributions = '; '.join(f'{kind} mean {values[i]*100:.1f}%' for kind,values in means.items())
        template=SCENARIO_TEMPLATES[CLASSES[i]]
        scenarios.append(dict(rank=rank,category=CLASSES[i],relative_score=combined[i],
            title=template['title'],explanation=template['explanation'],uncertainty=template['uncertainty'],
            model_basis=f"Rank {rank}: {CLASSES[i]} has a {combined[i]*100:.1f}% combined relative category score. This supports the ranking only, not the hypothetical details.",
            modality_contributions=contributions,
            explanation_type='conditional category template; not an observed event',observed_events=[]))
    return dict(scenarios=scenarios,scores=[dict(label=c,score=combined[i]) for i,c in enumerate(CLASSES)],
        status='Inconclusive / Low confidence' if weak or close else 'Unverified category hypotheses',
        method=METHOD,modality_means={kind:[dict(label=c,score=values[i]) for i,c in enumerate(CLASSES)] for kind,values in means.items()},
        unique_files=sum(map(len,groups.values())),duplicates_ignored=len(records)-len(seen),validated=False,
        uncertainties=[
            'UNVALIDATED EXPERIMENTAL: percentages are relative scores, not probabilities that a crime occurred.',
            'Neither model has a Normal/Unknown class; non-crime evidence can still receive crime labels.',
            'The image model collapsed to one category in its tiny held-out experiment; both models are unvalidated.',
            'User grouping does not verify that files depict one incident. Upload order does not establish chronology.',
            'Related frames, re-encoded copies and shared training biases can be correlated; equal weighting is a heuristic.',
            'Abstaining from event reconstruction: no actions, chronology, identity or intent can be inferred from these scores.'],
        display_rule='Inconclusive when top score <50% or top-two margin <10 percentage points; uncalibrated heuristic.',
        ranking_note='Top 3 of 4 classes; percentages are not renormalized over the displayed scenarios.')
