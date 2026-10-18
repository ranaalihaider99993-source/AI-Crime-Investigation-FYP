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
    cnn=CrimeCNN(); x=torch.rand(4,2,3,16,16); y=torch.arange(4)
    before=cnn.net[-1].weight.detach().clone()
    result=epoch(cnn,DataLoader(TensorDataset(x,y),batch_size=5),'cpu',torch.optim.Adam(cnn.parameters()))
    assert result['loss']>0 and len(result['confusion_matrix'])==4
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
    assert len(history)==2 and sum(sum(row) for row in result['confusion_matrix'])==4
    import csv
    from scripts.data import write_csv
    with (tmp_path/'split.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['split'] = {'train':'val', 'val':'train', 'test':'test'}[row['split']]
    write_csv(tmp_path/'changed.csv', rows, list(rows[0]))
    with pytest.raises(ValueError, match='membership differs'):
        eval_cnn(tmp_path/'changed.csv', tmp_path/'cnn/crime_classifier.pth', tmp_path/'should-not-exist.json')
    saved = torch.load(tmp_path/'cnn/crime_classifier.pth', weights_only=True)
    saved['classes'] = saved['classes'] + ['Normal']
    torch.save(saved, tmp_path/'legacy.pth')
    with pytest.raises(ValueError, match='class mapping differs'):
        eval_cnn(tmp_path/'split.csv', tmp_path/'legacy.pth', tmp_path/'should-not-exist.json')

    pairs='annotations/synthetic_scenarios.csv'
    history=train(pairs,tmp_path/'gru',epochs=2)
    assert len(history)==2
    generated=infer(tmp_path/'gru/scenario_gru.pth','classification normal observation person standing')
    assert generated['synthetic_training'] is True
    assert evaluate(tmp_path/'gru/scenario_gru.pth',pairs)['tokens']>0


def test_notebooks():
    notebooks=list(Path('notebooks').glob('*.ipynb'))
    assert {p.name for p in notebooks} == {
        '01_Dataset_Inspection.ipynb', '02_Crime_Classification_Training.ipynb',
        '03_Scenario_GRU_Training.ipynb', '04_Model_Evaluation.ipynb',
        '05_Reviewed_Image_Evidence.ipynb'}
    for p in notebooks:
        book=nbformat.read(p,as_version=4); nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code': compile(cell.source,str(p),'exec')


def test_pair_leakage(tmp_path):
    path=tmp_path/'bad.csv'
    path.write_text('source_id,split,evidence,scenario,provenance\na,train,one,two,synthetic\na,val,three,four,synthetic\n')
    with pytest.raises(ValueError,match='leakage'): load_pairs(path)


def test_notebook_preparation_keeps_training_disabled(tmp_path):
    from tests.test_data import test_duplicate_dataset_roots as make_mirrors
    make_mirrors(tmp_path)
    output = tmp_path/'notebook-output'
    output.mkdir()
    import shutil
    shutil.copytree(tmp_path/'report', output/'inspection')
    book = nbformat.read('notebooks/02_Crime_Classification_Training.ipynb', as_version=4)
    namespace = {'DATA': tmp_path/'dataset', 'OUTPUT': output}
    # Exercise the actual notebook preparation and guarded execution cells, not
    # Colab authentication, installation, Drive mounting, or GPU allocation.
    # Rendering widgets are not available in this CPU venv. Stub display only;
    # execute the actual preparation, guarded-launch and result code otherwise.
    namespace.update(display=lambda *args: None, Markdown=str, HTML=str, Image=lambda **kwargs: kwargs)
    for ident in ['automatic-preparation', 'temporal-selected-review', 'export-real-review', 'original-video-players', 'export-original-videos', 'source-context-players', 'consolidated-review-dashboard', 'reviewed-smoke-launch', 'actual-smoke-results']:
        source = next(c.source for c in book.cells if c.id == ident)
        source = source.replace('from IPython.display import display, Markdown, Image', '').replace('from IPython.display import HTML', '').replace('from IPython.display import display, HTML', '')
        exec(compile(source, '<notebook-cell>', 'exec'), namespace)
    assert namespace['SOURCE_MAPPING_REVIEWED'] is False
    assert namespace['RUN_PROTOTYPE_TRAINING'] is False
    assert namespace['AUTOMATIC_REPORT']['automatic_status'] == 'PASS'
    assert (output/'smoke_preparation/candidate_splits.csv').exists()
    assert namespace['SMOKE_RESULT'] is None
    assert namespace['DOWNLOAD_REAL_REVIEW'] is False
    assert namespace['EXPORT_ORIGINAL_VIDEOS'] is False
    assert namespace['DOWNLOAD_ORIGINAL_PACKAGE'] is False
    assert len(namespace['ORIGINAL_PLAN']['clips'])==12
    assert namespace['ORIGINAL_PACKAGE'] is None
    assert namespace['REAL_REVIEW_BUNDLE'].is_file()
    assert not (output/'classifier').exists()
