"""Evaluate held-out sources or paired synthetic scenarios with existing checkpoints."""
import argparse
import json
from pathlib import Path
from scripts.classifier import evaluate as cnn
from scripts.scenario import evaluate as gru


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model',choices=['cnn','gru']); p.add_argument('data'); p.add_argument('checkpoint'); p.add_argument('output')
    a=p.parse_args()
    result=cnn(a.data,a.checkpoint,a.output) if a.model=='cnn' else gru(a.checkpoint,a.data)
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
