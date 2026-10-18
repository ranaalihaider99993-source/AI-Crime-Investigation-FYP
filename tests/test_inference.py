import hashlib
import io
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from scripts.classifier import CrimeCNN, VideoDataset
from scripts.data import CLASSES
from scripts.inference import InferenceEngine, CHECKPOINT_SHA256
from scripts.video_frames import sample_video_frames
from web.app import create_app


@pytest.fixture
def sample_video(tmp_path):
    path=tmp_path/'sample.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,24))
    assert writer.isOpened()
    for index in range(5):writer.write(np.full((24,32,3),(index*20,50,150),dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def checkpoint(tmp_path):
    path=tmp_path/'fixture.pth'
    torch.manual_seed(1)
    torch.save(dict(state_dict=CrimeCNN().state_dict(),classes=CLASSES,
        config=dict(image_size=64,frames_per_clip=3),metadata=dict(validated_performance=False),epoch=1),path)
    return path


def test_exact_original_frame_pipeline(sample_video):
    actual,images,info=sample_video_frames(sample_video)
    cap=cv2.VideoCapture(str(sample_video));expected=[]
    for index in np.linspace(0,4,3).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read();assert ok
        rgb=cv2.cvtColor(cv2.resize(frame,(64,64)),cv2.COLOR_BGR2RGB)
        expected.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
    cap.release()
    assert torch.equal(actual,torch.stack(expected))
    assert info['sampled_indices']==[0,2,4] and actual.shape==(3,3,64,64)


def test_hash_and_class_order_fail_closed(checkpoint):
    with pytest.raises(ValueError,match='SHA-256'):InferenceEngine(checkpoint)
    saved=torch.load(checkpoint,weights_only=True);saved['classes']=list(reversed(CLASSES));torch.save(saved,checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='class order'):InferenceEngine(checkpoint,digest)


def test_prediction_uses_mean_logits_then_softmax(checkpoint,sample_video):
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    engine=InferenceEngine(checkpoint,digest)
    frames,_,_=sample_video_frames(sample_video)
    with torch.inference_mode():expected=engine.model(frames).mean(0).softmax(0).tolist()
    result=engine.predict(sample_video)
    assert [s['label'] for s in result['scores']]==CLASSES
    assert [s['score'] for s in result['scores']]==pytest.approx(expected)
    assert result['validated'] is False and len(result['frames'])==3


@pytest.mark.parametrize('change,message', [('preprocessing','preprocessing'),('architecture','size mismatch'),('nonfinite','non-finite')])
def test_checkpoint_contract_rejects_incompatible_weights(checkpoint,change,message):
    saved=torch.load(checkpoint,weights_only=True)
    if change=='preprocessing':saved['config']['image_size']=128
    elif change=='architecture':saved['state_dict']['net.0.weight']=torch.zeros(1,3,3,3)
    else:saved['state_dict']['net.0.weight'][0,0,0,0]=float('nan')
    torch.save(saved,checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises((ValueError,RuntimeError),match=message):InferenceEngine(checkpoint,digest)


@pytest.fixture
def app():
    path=Path('models/experimental_cnn.pth')
    if not path.exists():pytest.skip('Retrieve the approved experimental checkpoint for real inference tests')
    return create_app(path)


def test_upload_errors_and_warning(app,tmp_path):
    client=app.test_client()
    assert client.get('/').status_code==200
    assert b'UNVALIDATED EXPERIMENTAL' in client.get('/').data
    assert client.get('/api/model').json['checkpoint_sha256']==CHECKPOINT_SHA256
    assert client.post('/api/predict').status_code==400
    assert client.post('/api/predict',data={'video':(io.BytesIO(b'x'),'sample.txt')}).status_code==415
    assert client.post('/api/predict',data={'video':(io.BytesIO(b'bad video'),'sample.mp4')}).status_code==422
    app.config['MAX_CONTENT_LENGTH']=20
    assert client.post('/api/predict',data={'video':(io.BytesIO(b'x'*100),'sample.mp4')}).status_code==413


def test_real_saved_checkpoint_and_sample_end_to_end(app):
    path=Path('artifacts/inference/sample.mp4')
    if not path.exists():pytest.skip('Real sample not present')
    result=app.test_client().post('/api/predict',data={'video':(io.BytesIO(path.read_bytes()),'real-sample.mp4')})
    assert result.status_code==200
    payload=result.json
    assert payload['validated'] is False
    assert len(payload['scores'])==4 and len(payload['frames'])==3
    assert sum(row['score'] for row in payload['scores'])==pytest.approx(1)
    assert payload['model']['checkpoint_sha256']==CHECKPOINT_SHA256
