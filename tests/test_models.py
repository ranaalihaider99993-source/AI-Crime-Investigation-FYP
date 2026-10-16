import json
from pathlib import Path
import nbformat
import pytest
import torch
from scripts.classifier import CrimeCNN, epoch, train as train_cnn, evaluate as eval_cnn
from scripts.scenario import ScenarioGRU, train, infer, evaluate, load_pairs
from torch.utils.data import DataLoader, TensorDataset
from tests.test_data import test_inspection_and_split as make_test_videos


def test_forward_and_update():
    torch.set_num_threads(1)
    cnn=CrimeCNN(); x=torch.rand(5,2,3,16,16); y=torch.arange(5)
    before=cnn.net[-1].weight.detach().clone()
    result=epoch(cnn,DataLoader(TensorDataset(x,y),batch_size=5),'cpu',torch.optim.Adam(cnn.parameters()))
    assert result['loss']>0 and len(result['confusion_matrix'])==5
    assert not torch.equal(before,cnn.net[-1].weight)
    gru=ScenarioGRU(10)
    src=torch.tensor([[1,4,2,0],[1,5,6,2]])
    assert gru(src,src[:,:-1]).shape==(2,3,10)
    assert gru.generate(src,max_tokens=3).shape[0]==2


def test_training_and_checkpoints(tmp_path):
    torch.set_num_threads(1)
    make_test_videos(tmp_path)
    config=dict(seed=42,image_size=16,frames_per_clip=1,batch_size=5,learning_rate=.001,epochs=2,patience=1)
    history=train_cnn(tmp_path/'split.csv',tmp_path/'cnn',config)
    result=eval_cnn(tmp_path/'split.csv',tmp_path/'cnn/crime_classifier.pth',tmp_path/'cnn/test.json')
    assert len(history)==2 and sum(sum(row) for row in result['confusion_matrix'])==5
    pairs='annotations/synthetic_scenarios.csv'
    history=train(pairs,tmp_path/'gru',epochs=2)
    assert len(history)==2
    generated=infer(tmp_path/'gru/scenario_gru.pth','classification normal observation person standing')
    assert generated['synthetic_training'] is True
    assert evaluate(tmp_path/'gru/scenario_gru.pth',pairs)['tokens']>0


def test_notebooks():
    notebooks=list(Path('notebooks').glob('*.ipynb'))
    assert len(notebooks)==4
    for p in notebooks:
        book=nbformat.read(p,as_version=4); nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code': compile(cell.source,str(p),'exec')


def test_pair_leakage(tmp_path):
    path=tmp_path/'bad.csv'
    path.write_text('source_id,split,evidence,scenario,provenance\na,train,one,two,synthetic\na,val,three,four,synthetic\n')
    with pytest.raises(ValueError,match='leakage'): load_pairs(path)
