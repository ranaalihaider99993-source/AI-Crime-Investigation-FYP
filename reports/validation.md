# Validation evidence

Executed in Codex Cloud using Python 3.12.14 and PyTorch 2.14.1+cpu.

- `python -m pytest -q`: **6 passed**. Tests cover actual temporary MJPEG video decoding,
  directories ending in `.mp4`, source grouping, disjoint source splits, unreadable-file
  handling, missing-class rejection, CNN and GRU forward passes, an optimizer weight update,
  two-epoch training loops, checkpoint reload, held-out evaluation, annotation leakage
  rejection, four notebook schemas, and compilation of every notebook code cell.
- `python -m compileall -q scripts tests`: passed.
- `pip check`: no broken requirements.
- Preprocessing and evaluation CLI help commands: passed.
- `reports/smoke_metrics.json`: actual two-epoch CPU smoke training and evaluation metrics.
  The CNN saw generated blank videos with arbitrary labels. The GRU saw 12 explicitly
  synthetic pairs. These results establish pipeline execution only. They provide **no
  evidence of real crime-classification or scenario-generation performance**.
- Cloud package versions are recorded in `requirements-cloud-tested.txt`.

## Not executed / remaining work

No full dataset download, real DCSASS statistics, annotation-schema verification, full
training, or Google Colab execution occurred. GPU performance and Drive mounting are
unverified here. No real accuracy is claimed. The Windows dataset path was not accessed.

In Colab, first execute notebook 01 to discover actual categories, video counts, label
files, and directory structure. Review all annotation candidates for bounding boxes and
paired narratives. Verify source mapping and visually related clips before enabling the
balanced prototype in notebook 02. Full training requires user confirmation. Notebook 03
only demonstrates the synthetic GRU pipeline. Use an adequately sized human-reviewed paired
dataset before evaluating investigative usefulness. Notebook 04 evaluates existing saved
checkpoints; missing files produce errors rather than invented results.

The source is committed locally; publishing the commit to the existing GitHub repository
is a separate step before Colab can clone these files. No checkpoints, datasets, videos,
credentials, or temporary test outputs are committed.
