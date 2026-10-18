"""Pinned video inference with abstaining evidence narratives; no review writes."""
import base64
import hashlib
import threading
from pathlib import Path
import cv2
import torch
from scripts.classifier import CrimeCNN
from scripts.data import CLASSES
from scripts.video_frames import sample_video_frames
from scripts.evidence_narrative import build_narrative

CHECKPOINT_SHA256 = 'a31794fd43f0a736d63702806a0c838af266b3536867a65da4cd74b4082b5ab6'
WARNING = 'UNVALIDATED EXPERIMENTAL — model scores are not evidence of a crime.'


class InferenceEngine:
    def __init__(self, checkpoint, expected_sha256=CHECKPOINT_SHA256):
        path=Path(checkpoint)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError('Checkpoint SHA-256 does not match the approved experimental model')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['classes'] != CLASSES or saved['classes'] != ['Burglary','Stealing','Fighting','Vandalism']:
            raise ValueError('Checkpoint class order mismatch')
        if saved['config']['image_size'] != 64 or saved['config']['frames_per_clip'] != 3:
            raise ValueError('Checkpoint preprocessing mismatch')
        if saved['metadata'].get('validated_performance') is not False:
            raise ValueError('Expected explicitly unvalidated experimental provenance')
        self.model=CrimeCNN().cpu()
        self.model.load_state_dict(saved['state_dict'],strict=True)
        if not all(torch.isfinite(v).all().item() for v in self.model.state_dict().values()):
            raise ValueError('Checkpoint contains non-finite weights')
        self.model.eval()
        self.lock=threading.Lock()
        self.metadata=dict(checkpoint_sha256=digest,checkpoint_epoch=saved['epoch'],classes=CLASSES,
            architecture='CrimeCNN: Conv 3→16→32→64, adaptive average pooling, Linear 64→4',
            preprocessing='3 uniformly spaced frames; OpenCV linear resize 64×64; BGR→RGB; CHW float32 /255',
            aggregation='Average frame logits, then softmax',device='CPU',validated=False,
            warning=WARNING,calibration='Uncalibrated relative scores; no Normal/Unknown class')

    def predict(self,path):
        frames,previews,details=sample_video_frames(path)
        with self.lock,torch.inference_mode():
            logits=self.model(frames).mean(0)
            scores=torch.softmax(logits,dim=0)
        if not torch.isfinite(scores).all():
            raise ValueError('Model returned non-finite scores')
        thumbnails=[]
        for image in previews:
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if not ok:raise ValueError('Cannot encode sampled frame')
            thumbnails.append('data:image/jpeg;base64,'+base64.b64encode(jpeg).decode())
        return dict(prediction=CLASSES[int(scores.argmax())],
                    scores=[dict(label=label,score=float(scores[i])) for i,label in enumerate(CLASSES)],
                    preprocessing=details,frames=thumbnails,model=self.metadata,
                    warning=WARNING,validated=False,
                    narrative=build_narrative('video', [dict(id=f'frame-{index}',
                        description=f'Decoded sampled frame at index {index}; RGB 64 × 64. No action annotation is available.')
                        for index in dict.fromkeys(details['sampled_indices'])]))
