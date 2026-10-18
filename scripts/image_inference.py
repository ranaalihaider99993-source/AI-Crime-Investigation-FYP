"""Pinned, CPU-only experimental image classification; not scene understanding."""
import hashlib
import threading
from pathlib import Path
import torch
from scripts.experimental_image import WeakFrameCNN
from scripts.image_evidence_training import PREPROCESSING, preprocess
from scripts.data import CLASSES

# Set only to the actual verified Colab artifact after successful training.
IMAGE_CHECKPOINT_SHA256 = 'e95b7ebf04510e78bbbde4e7716408cbdd1666291a7406ed42dcf26878e647a8'


class ImageInferenceEngine:
    def __init__(self,path,expected_sha256=IMAGE_CHECKPOINT_SHA256):
        path=Path(path); digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_sha256 or digest != expected_sha256:
            raise ValueError('Image checkpoint SHA-256 mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        config=saved['config']
        if (config.get('architecture')!='WeakFrameCNN-v1' or config.get('classes')!=CLASSES or
            config.get('preprocessing')!=PREPROCESSING or config.get('weak_labels') is not True or
            config.get('validated') is not False or config.get('random_initialization') is not True or
            config.get('pretrained') is not False):
            raise ValueError('Image checkpoint contract mismatch')
        self.model=WeakFrameCNN(); self.model.load_state_dict(saved['state_dict'],strict=True)
        if not all(torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError('Non-finite image weights')
        self.model.eval(); self.lock=threading.Lock()
        self.metadata=dict(checkpoint_sha256=digest,epoch=saved['epoch'],classes=CLASSES,
            architecture=config['architecture'],preprocessing=PREPROCESSING,validated=False,
            weak_labels=True,device='CPU',warning='UNVALIDATED EXPERIMENTAL — weak video-derived labels; scores are not crime probabilities.')

    def predict(self,data):
        image=preprocess(data).unsqueeze(0)
        with self.lock,torch.inference_mode(): scores=self.model(image).softmax(1)[0]
        if not torch.isfinite(scores).all(): raise ValueError('Non-finite image scores')
        return dict(prediction=CLASSES[int(scores.argmax())],scores=[dict(label=c,score=float(scores[i])) for i,c in enumerate(CLASSES)],
                    model=self.metadata,validated=False)
