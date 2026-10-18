import hashlib
import io
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from scripts.experimental_image import WeakFrameCNN, fingerprint, duplicate_reason, extract_frames, train_frames


def test_duplicate_filter_exact_and_near():
    image=Image.new('RGB',(128,128),(100,50,80))
    prior=fingerprint(image); seen=[(prior,'a.png','train','one')]
    assert duplicate_reason(prior,seen)['kind']=='exact_preprocessed_pixels'
    assert duplicate_reason(fingerprint(Image.new('RGB',(128,128),(101,50,80))),seen)['kind'].startswith('near_duplicate')
    assert duplicate_reason(fingerprint(Image.new('RGB',(128,128),(220,20,20))),seen) is None


def test_authorization_and_gpu_gates(tmp_path):
    from scripts.experimental_image import run
    with pytest.raises(ValueError,match='authorization'): run(tmp_path,tmp_path,tmp_path)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='T4'): run(tmp_path,tmp_path,tmp_path,authorized=True)


def test_random_image_training_fixture_only(tmp_path):
    from scripts.data import CLASSES
    folder=tmp_path/'frames'; folder.mkdir()
    frames=[]
    for split in ['train','val','test']:
        for i,label in enumerate(CLASSES):
            name=f'{split}-{i}.png'
            Image.new('RGB',(128,128),(30*i+10,len(split)*20,40)).save(folder/name)
            frames.append(dict(file=name,sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest(),
                               split=split,source=f'{split}-{i}',label=label))
    output=tmp_path/'run'; output.mkdir()
    result=train_frames(frames,folder,output,epochs=1,device='cpu')
    assert result['device']=='cpu' and len(result['source']['confusion_matrix'])==4
    assert result['frame']['unsupported_classes']==[]
    saved=torch.load(output/'experimental_image_cnn.pth',weights_only=True)
    assert saved['config']['weak_labels'] and not saved['config']['validated']
    assert saved['config']['classes']==CLASSES


def test_checkpoint_pinning_and_real_image_scores():
    from scripts.image_inference import ImageInferenceEngine, IMAGE_CHECKPOINT_SHA256
    from scripts.image_evidence_training import preprocess
    path=Path('models/experimental_image_cnn.pth')
    if not path.exists(): pytest.skip('Verified trained checkpoint not installed')
    with pytest.raises(ValueError,match='SHA-256'): ImageInferenceEngine(path,'0'*64)
    engine=ImageInferenceEngine(path)
    stream=io.BytesIO(); Image.new('RGB',(320,240),(60,50,100)).save(stream,format='PNG')
    result=engine.predict(stream.getvalue())
    with torch.inference_mode(): expected=engine.model(preprocess(stream.getvalue()).unsqueeze(0)).softmax(1)[0]
    assert [r['score'] for r in result['scores']]==pytest.approx(expected.tolist())
    assert sum(r['score'] for r in result['scores'])==pytest.approx(1)
    assert not result['validated'] and result['model']['weak_labels']
    assert result['model']['checkpoint_sha256']==IMAGE_CHECKPOINT_SHA256


@pytest.mark.parametrize('change', ['classes','preprocessing','nonfinite'])
def test_image_contract_fail_closed(tmp_path,change):
    from scripts.image_inference import ImageInferenceEngine
    path=Path('models/experimental_image_cnn.pth')
    if not path.exists(): pytest.skip('Verified trained checkpoint not installed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if change=='classes': saved['config']['classes']=list(reversed(saved['config']['classes']))
    if change=='preprocessing': saved['config']['preprocessing']['size']=64
    if change=='nonfinite': saved['state_dict']['net.0.weight'][0,0,0,0]=float('nan')
    target=tmp_path/'bad.pth'; torch.save(saved,target)
    with pytest.raises(ValueError): ImageInferenceEngine(target,hashlib.sha256(target.read_bytes()).hexdigest())


def test_real_image_api_and_video_preserved():
    from web.app import create_app
    from scripts.inference import CHECKPOINT_SHA256
    if not Path('models/experimental_image_cnn.pth').exists(): pytest.skip('Image checkpoint missing')
    client=create_app().test_client()
    image=Path('artifacts/inference/real-frame.png')
    if not image.exists(): pytest.skip('Real image fixture missing')
    result=client.post('/api/images/analyze',data={'images':[(io.BytesIO(image.read_bytes()),'real.png') for _ in range(6)]})
    assert result.status_code==200 and result.json['image_model_available']
    assert len(result.json['images'])==6
    assert all('experimental_classification' in item for item in result.json['images'])
    assert result.json['narrative']['status']=='abstained'
    assert result.json['semantic_model_available'] is False
    assert client.get('/api/images/model').json['available']
    assert client.get('/api/model').json['checkpoint_sha256']==CHECKPOINT_SHA256
