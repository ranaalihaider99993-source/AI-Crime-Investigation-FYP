"""Small GRU encoder-decoder with a training-only vocabulary and random weights."""
import csv
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts.common import seed_all, experiment_metadata

SPECIAL=['<pad>','<bos>','<eos>','<unk>']


def tokenize(text): return text.lower().split()


def load_pairs(path):
    with open(path) as f: rows=list(csv.DictReader(f))
    required={'source_id','split','evidence','scenario','provenance'}
    if not rows or not required.issubset(rows[0]): raise ValueError('Missing annotation columns or empty data')
    seen={}; pairs=set()
    for r in rows:
        if r['split'] not in {'train','val','test'} or not all(r[k].strip() for k in required): raise ValueError('Invalid annotation row')
        for key in [('source',r['source_id']),('evidence',r['evidence'].strip().lower())]:
            if key in seen and seen[key]!=r['split']: raise ValueError('Cross-split source/evidence leakage')
            seen[key]=r['split']
        pair=(r['evidence'],r['scenario'])
        if pair in pairs: raise ValueError('Duplicate pair')
        pairs.add(pair)
    return rows


def vocabulary(rows):
    return SPECIAL+sorted({w for r in rows if r['split']=='train' for k in ['evidence','scenario'] for w in tokenize(r[k])}-set(SPECIAL))


def encode(text,vocab,length=48):
    lookup={w:i for i,w in enumerate(vocab)}
    ids=[1]+[lookup.get(w,3) for w in tokenize(text)[:length-2]]+[2]
    return ids+[0]*(length-len(ids))


class ScenarioGRU(nn.Module):
    def __init__(self,vocab_size,embedding=32,hidden=64):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,embedding,padding_idx=0)
        self.encoder=nn.GRU(embedding,hidden,batch_first=True); self.decoder=nn.GRU(embedding,hidden,batch_first=True); self.output=nn.Linear(hidden,vocab_size)
    def context(self,src):
        lengths=(src!=0).sum(1).cpu()
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(src),lengths,batch_first=True,enforce_sorted=False)
        return self.encoder(packed)[1]
    def forward(self,src,target):
        decoded,_=self.decoder(self.embedding(target),self.context(src)); return self.output(decoded)
    @torch.no_grad()
    def generate(self,src,max_tokens=40):
        self.eval(); hidden=self.context(src); token=torch.ones((src.size(0),1),dtype=torch.long,device=src.device); result=[]
        for _ in range(max_tokens):
            value,hidden=self.decoder(self.embedding(token),hidden); scores=self.output(value[:,-1]); scores[:,[0,1]]=-float('inf')
            token=scores.argmax(-1,keepdim=True); result.append(token)
            if (token==2).all(): break
        return torch.cat(result,1)


def loader(rows,vocab,split):
    subset=[r for r in rows if r['split']==split]
    if not subset: raise ValueError(f'Empty {split}')
    return DataLoader(TensorDataset(torch.tensor([encode(r['evidence'],vocab) for r in subset]),torch.tensor([encode(r['scenario'],vocab) for r in subset])),batch_size=8,shuffle=split=='train')


def run_epoch(model,batches,device,optimizer=None):
    model.train(optimizer is not None); total=0.; tokens=0; correct=0
    with torch.set_grad_enabled(optimizer is not None):
        for src,tgt in batches:
            src,tgt=src.to(device),tgt.to(device); logits=model(src,tgt[:,:-1]); target=tgt[:,1:]; mask=target!=0
            loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=0)
            if optimizer is not None: optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            count=mask.sum().item(); total+=loss.item()*count; tokens+=count; correct+=((logits.argmax(-1)==target)&mask).sum().item()
    return {'loss':total/tokens,'teacher_forced_token_accuracy':correct/tokens,'tokens':tokens}


def train(path,output,epochs=5,patience=3,seed=42):
    if epochs < 1 or patience < 1: raise ValueError('Positive epochs and patience required')
    seed_all(seed); rows=load_pairs(path); vocab=vocabulary(rows); device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ScenarioGRU(len(vocab)).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001)
    output=Path(output); output.mkdir(parents=True,exist_ok=True); best=float('inf'); stale=0; history=[]
    (output/'experiment.json').write_text(json.dumps(experiment_metadata(path,seed),indent=2))
    for i in range(epochs):
        tr=run_epoch(model,loader(rows,vocab,'train'),device,opt); val=run_epoch(model,loader(rows,vocab,'val'),device)
        history.append(dict(epoch=i+1,train=tr,val=val)); print(history[-1])
        if val['loss']<best:
            best=val['loss']; stale=0
            torch.save(dict(state_dict=model.state_dict(),vocab=vocab,synthetic=all(r['provenance']=='synthetic' for r in rows),seed=seed),output/'scenario_gru.pth')
        else: stale+=1
        (output/'scenario_history.json').write_text(json.dumps(history,indent=2))
        if stale>=patience: break
    return history


def infer(checkpoint,evidence):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True); vocab=saved['vocab']; model=ScenarioGRU(len(vocab)); model.load_state_dict(saved['state_dict'])
    ids=model.generate(torch.tensor([encode(evidence,vocab)]))[0].tolist(); ids=ids[:ids.index(2)] if 2 in ids else ids
    return {'hypothesis':' '.join(vocab[i] for i in ids if i not in [0,1]),'synthetic_training':saved['synthetic'],'warning':'Unverified model output; human review required. No evidential or factual reliability established.'}


def evaluate(checkpoint,path):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True); rows=load_pairs(path); model=ScenarioGRU(len(saved['vocab'])); model.load_state_dict(saved['state_dict'])
    return run_epoch(model,loader(rows,saved['vocab'],'test'),'cpu')
