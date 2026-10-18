"""One conservative narrative contract for video and image inputs.

Classification scores, upload order and frame order are never promoted to facts.
No free-form model text or synthetic scenario GRU output enters this pipeline.
"""


def build_narrative(modality, evidence):
    if modality not in {'video', 'images'} or not evidence:
        raise ValueError('A supported modality and evidence records are required')
    observations = []
    for item in evidence:
        observations.append(dict(evidence_id=item['id'], kind='technical_measurement',
                                 text=item['description']))
    return dict(
        schema_version=1, modality=modality, status='abstained',
        generator='deterministic evidence-bound renderer; not a trained language model',
        visible_observations=[], technical_observations=observations,
        possible_incident_sequence=dict(status='abstained', evidence_ids=[],
            text='No incident sequence is supported. Input/frame order does not establish an action or causal sequence.'),
        crime_hypothesis=dict(status='abstained', evidence_ids=[],
            text='Insufficient verified visual evidence to propose a crime hypothesis. Experimental category scores are not observations.'),
        uncertainties=[
            'No reviewed semantic image model or validated temporal action model is integrated.',
            'Visible objects, actions and relationships between inputs remain unverified.',
            'Identity, suspects, intent, motives and crime occurrence cannot be established.',
            'Human review and incident-disjoint evaluation are required.'],
        warning='UNVALIDATED EXPERIMENTAL — abstention is not evidence that no crime occurred.')
