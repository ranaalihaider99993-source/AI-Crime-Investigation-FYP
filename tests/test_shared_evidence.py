import hashlib
import io
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from scripts.evidence_narrative import build_narrative
from scripts.image_analysis import analyze_images
from scripts.image_evidence_training import ATTRIBUTES, ImageEvidenceCNN, validate_manifest, train


def fixture_manifest(tmp_path):
    """Synthetic colored squares test plumbing, never a crime/attribute benchmark."""
    rows = []
    for i in range(6):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (18, 12), (i*30, 60, 90)).save(path)
        rows.append(dict(image_id=str(i), path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_group=f'fixture-{i}', split=['train','val','test'][i//2],
            review_status='approved', reviewer='test-fixture-only', incident_independence='verified',
            provenance='synthetic unit test', license='test generated', labels={a:i%2 for a in ATTRIBUTES}))
    manifest = dict(schema_version=1, attributes=ATTRIBUTES, near_duplicate_audit='approved',
                    audit_reviewer='test-fixture-only', images=rows)
    path = tmp_path / 'fixture.json'; path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.mark.parametrize('modality', ['video','images'])
def test_both_modalities_abstain_without_semantic_evidence(modality):
    result = build_narrative(modality, [dict(id='input-1', description='Decoded input')])
    assert result['status'] == 'abstained'
    assert result['visible_observations'] == []
    assert result['possible_incident_sequence']['evidence_ids'] == []
    assert result['crime_hypothesis']['status'] == 'abstained'
    assert result['technical_observations'][0]['evidence_id'] == 'input-1'


def test_images_share_schema_not_upload_chronology():
    image = io.BytesIO(); Image.new('RGB', (10, 20)).save(image, format='PNG')
    result = analyze_images([image.getvalue()] * 2)
    assert result['narrative']['modality'] == 'images'
    assert 'does not establish' in result['narrative']['possible_incident_sequence']['text']


@pytest.mark.parametrize('change,match', [
    ('unreviewed','Unreviewed'), ('independence','Unreviewed'), ('leak','leakage'),
    ('hash','SHA-256'), ('duplicate','Duplicate'), ('escape','outside'), ('label','Attributes'),
    ('audit','audit'), ('coverage','support'), ('unknown','reviewed attribute')])
def test_training_gates(tmp_path, change, match):
    path, manifest = fixture_manifest(tmp_path)
    rows = manifest['images']
    if change == 'unreviewed': rows[0]['review_status'] = 'uncertain'
    if change == 'independence': rows[0]['incident_independence'] = 'pending'
    if change == 'leak': rows[2]['source_group'] = rows[0]['source_group']
    if change == 'hash': rows[0]['sha256'] = '0'*64
    if change == 'duplicate': rows[1]['path'] = rows[0]['path']; rows[1]['sha256'] = rows[0]['sha256']
    if change == 'escape': rows[0]['path'] = '../outside.png'
    if change == 'label': rows[0]['labels'][ATTRIBUTES[0]] = True
    if change == 'audit': manifest['near_duplicate_audit'] = 'pending'
    if change == 'coverage': rows[0]['labels'][ATTRIBUTES[0]] = None
    if change == 'unknown': rows[0]['labels'] = {a:None for a in ATTRIBUTES}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match): train(path, tmp_path, tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()


def test_cpu_training_plumbing_only(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    assert len(validate_manifest(path, tmp_path)) == 6
    output = tmp_path/'test-only-run'
    config = train(path, tmp_path, output, epochs=1)
    assert config['random_initialization'] and not config['deployment_approved']
    assert config['attributes'] == ATTRIBUTES
    assert (output/'test_metrics.json').exists()
    saved = torch.load(output/'image_evidence.pth', weights_only=True)
    model = ImageEvidenceCNN(); model.load_state_dict(saved['state_dict'])
    assert model(torch.zeros(1,3,128,128)).shape == (1,4)
    with pytest.raises(FileExistsError): train(path, tmp_path, output, epochs=1)


def test_empty_annotations_block_training(tmp_path):
    path = tmp_path/'empty.json'
    path.write_text(json.dumps(dict(schema_version=1, attributes=ATTRIBUTES,
        near_duplicate_audit='approved', audit_reviewer='fixture', images=[])))
    with pytest.raises(ValueError, match='No reviewed'): train(path, tmp_path, tmp_path/'blocked')
