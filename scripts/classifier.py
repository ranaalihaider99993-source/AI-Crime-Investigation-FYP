"""Randomly initialized CNN; deterministic temporal averaging, no pretrained assets."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from scripts.common import seed_all, experiment_metadata
from scripts.data import CLASSES, validate_manifest, split_signature
from scripts.video_frames import sample_video_frames


class CrimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                               nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                               nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,len(CLASSES)))
    def forward(self,x): return self.net(x)


class VideoDataset(Dataset):
    def __init__(self,manifest,split,size=64,frames=3):
        with open(manifest) as f: rows=list(csv.DictReader(f))
        validate_manifest(rows, require_splits=True, require_classes=True)
        self.rows=[r for r in rows if r['split']==split]
        if not self.rows: raise ValueError(f'Empty {split} split')
        self.size,self.frames=size,frames
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        try:
            images,_,_=sample_video_frames(r['path'],self.size,self.frames)
        except ValueError as exc:
            raise RuntimeError(f'Unreadable sampled frame in {r["path"]}: {exc}') from exc
        return images,CLASSES.index(r['label'])


def metrics(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=list(range(len(CLASSES))),average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),precision=p,recall=r,macro_f1=f,
                confusion_matrix=confusion_matrix(y,pred,labels=list(range(len(CLASSES)))).tolist(),
                per_class=classification_report(y,pred,labels=list(range(len(CLASSES))),target_names=CLASSES,output_dict=True,zero_division=0))


def epoch(model,loader,device,optimizer=None):
    model.train(optimizer is not None); total=0.; ys=[]; ps=[]
    with torch.set_grad_enabled(optimizer is not None):
        for x,y in loader:
            x,y=x.to(device),y.to(device); b,t,c,h,w=x.shape
            logits=model(x.reshape(b*t,c,h,w)).reshape(b,t,len(CLASSES)).mean(1)
            loss=nn.functional.cross_entropy(logits,y)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total+=loss.item()*b; ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).detach().cpu().tolist())
    return dict(loss=total/len(ys),**metrics(ys,ps))


def train(manifest,output,config):
    if config['epochs'] < 1 or config['patience'] < 1: raise ValueError('Positive epochs and patience required')
    seed_all(config['seed']); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    metadata=experiment_metadata(manifest,config['seed'])
    metadata['canonical_split_signature']=split_signature(manifest)
    (output/'experiment.json').write_text(json.dumps(metadata,indent=2))
    (output/'split_manifest.csv').write_text(Path(manifest).read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata['training_device']=str(device)
    metadata['gpu_name']=torch.cuda.get_device_name(device) if device.type=='cuda' else None
    (output/'experiment.json').write_text(json.dumps(metadata,indent=2))
    print('Actual training device:', device, metadata['gpu_name'])
    model=CrimeCNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    loaders={s:DataLoader(VideoDataset(manifest,s,config['image_size'],config['frames_per_clip']),batch_size=config['batch_size'],shuffle=s=='train',num_workers=0) for s in ['train','val']}
    best=float('inf'); stale=0; history=[]
    for i in range(config['epochs']):
        tr=epoch(model,loaders['train'],device,optimizer); val=epoch(model,loaders['val'],device)
        history.append(dict(epoch=i+1,train=tr,val=val)); print(i+1,tr['loss'],val['loss'])
        if val['loss'] < best:
            best=val['loss']; stale=0
            torch.save(dict(state_dict=model.state_dict(),config=config,metadata=metadata,classes=CLASSES,epoch=i+1,validation=val),output/'crime_classifier.pth')
        else: stale+=1
        (output/'classifier_history.json').write_text(json.dumps(history,indent=2))
        if stale>=config['patience']: break
    return history


def evaluate(manifest,checkpoint,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True); config=saved['config']
    if saved['classes'] != CLASSES:
        raise ValueError('Checkpoint class mapping differs; regenerate a four-class checkpoint from random initialization')
    if saved.get('metadata', {}).get('canonical_split_signature') != split_signature(manifest):
        raise ValueError('Canonical split membership differs from training checkpoint; do not reassign held-out sources')
    model=CrimeCNN(); model.load_state_dict(saved['state_dict'])
    result=epoch(model,DataLoader(VideoDataset(manifest,'test',config['image_size'],config['frames_per_clip']),batch_size=config['batch_size']),'cpu')
    Path(output).write_text(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('manifest'); parser.add_argument('--output',default='artifacts/classifier'); parser.add_argument('--config',default='configs/prototype.json')
    args=parser.parse_args(); train(args.manifest,args.output,json.loads(Path(args.config).read_text()))
