import io
import math
from pathlib import Path
import pytest
from scripts.data import CLASSES
from scripts.incident_analysis import aggregate


def record(kind,digest,values):
    return dict(kind=kind,sha256=digest,scores=[dict(label=c,score=v) for c,v in zip(CLASSES,values)])


def test_equal_modality_means_and_duplicates_not_votes():
    image=record('image','a',[.7,.1,.1,.1])
    other=record('image','b',[.5,.2,.2,.1])
    video=record('video','v',[.1,.2,.3,.4])
    result=aggregate([image,other,video,image])
    assert [s['score'] for s in result['scores']]==pytest.approx([.35,.175,.225,.25])
    assert result['duplicates_ignored']==1 and result['unique_files']==3
    assert [s['category'] for s in result['scenarios']]==['Burglary','Vandalism','Fighting']
    assert sum(s['relative_score'] for s in result['scenarios'])<1
    assert result['status']=='Inconclusive / Low confidence'


@pytest.mark.parametrize('values',[[float('nan'),.2,.3,.5],[.8,.2,.1,.1],[-.1,.3,.4,.4]])
def test_reject_malformed_score_vectors(values):
    with pytest.raises(ValueError): aggregate([record('image','a',values)])


def test_scores_are_not_events_or_crime_probability():
    result=aggregate([record('video','x',[.8,.1,.05,.05])])
    assert result['status']=='Unverified category hypotheses'
    assert not result['validated']
    assert all('no visible act' in s['explanation'] for s in result['scenarios'])
    assert len(result['scenarios'])==3
    with pytest.raises(ValueError): aggregate([])
    bad=record('image','a',[.25]*4); bad['scores'].reverse()
    with pytest.raises(ValueError): aggregate([bad])


@pytest.fixture(scope='module')
def client():
    from web.app import create_app
    if not Path('models/experimental_image_cnn.pth').exists(): pytest.skip('Installed image checkpoint required')
    return create_app().test_client()


def image_file(): return (io.BytesIO(Path('artifacts/inference/real-frame.png').read_bytes()),'frame.png')
def video_file(): return (io.BytesIO(Path('artifacts/inference/sample.mp4').read_bytes()),'sample.mp4')


@pytest.mark.parametrize('mode',['images','videos','mixed'])
def test_real_incident_modes(client,mode):
    if not Path('artifacts/inference/real-frame.png').exists(): pytest.skip('Real sample required')
    files=([image_file(),image_file()] if mode=='images' else [video_file(),video_file()] if mode=='videos' else [image_file(),video_file()])
    response=client.post('/api/incidents/analyze',data={'evidence':files})
    assert response.status_code==200
    data=response.json
    assert len(data['scenarios'])==3 and len(data['files'])==2
    assert sum(s['score'] for s in data['scores'])==pytest.approx(1)
    assert not data['validated']
    if mode!='mixed': assert data['duplicates_ignored']==1
    else:
        means=data['modality_means']
        assert [s['score'] for s in data['scores']]==pytest.approx([(a['score']+b['score'])/2 for a,b in zip(means['image'],means['video'])])


def test_incident_rejection_and_no_partial_results(client):
    assert client.post('/api/incidents/analyze').status_code==400
    assert client.post('/api/incidents/analyze',data={'evidence':(io.BytesIO(b'x'),'bad.txt')}).status_code==415
    result=client.post('/api/incidents/analyze',data={'evidence':(io.BytesIO(b'not image'),'bad.jpg')})
    assert result.status_code==422 and 'scenarios' not in result.json
    assert client.post('/api/incidents/analyze',data={'evidence':[(io.BytesIO(b'x'),f'{i}.png') for i in range(7)]}).status_code==400
    assert client.post('/api/incidents/analyze',data={'evidence':[(io.BytesIO(b'x'),f'{i}.mp4') for i in range(4)]}).status_code==400


def test_simplified_home_and_legacy_routes(client):
    page=client.get('/').data
    assert b'Analyze Incident' in page and b'Advanced Details' in page
    assert b'id="video-file"' in client.get('/individual').data
    assert client.get('/api/model').status_code==200
    assert client.get('/api/images/model').json['available']
