# AI-Based Crime Investigation and Scenario Generation System Using Machine Learning and Blockchain

Final Year Project using Machine Learning and Blockchain.

This phase provides reproducible **from-scratch PyTorch training pipelines** and Google
Colab notebooks. The original repository description was “AI-Based Crime Investigation
and Scenario Generation System.” Frontend, live camera, object-detector training,
multimodal fusion, blockchain, and investigation reports are future work.
There are no pretrained weights, external AI inference calls, or fabricated accuracy results.

## Start in Google Colab

1. Ensure this commit is pushed to the existing GitHub repository (a local commit alone is not visible to Colab).
2. Go to **https://colab.research.google.com/**, click **File → Open notebook → GitHub**,
   enter `ranaalihaider99993-source/AI-Crime-Investigation-FYP`, and open
   **`notebooks/01_Dataset_Inspection.ipynb` first**.
3. Click **Runtime → Run all**, then authorize Google Drive mounting when prompted.
   GPU is optional for inspection. For training choose **Runtime → Change runtime type → T4 GPU**.
4. The notebook installs requirements, checks the device, and downloads
   `mateohervas/dcsass-dataset` using KaggleHub. It does not access a Windows drive or require a ZIP upload.
   If Kaggle requires authentication, use KaggleHub's supported login or secure Colab secret handling.
   Never paste credentials into committed cells.
5. Review the generated `inspection.json` and `manifest.csv` in
   `MyDrive/AI-Crime-Investigation-FYP/inspection`. Inspect **all** candidate annotation files;
   the notebook previews only the first 20. Reports contain observed counts, not assumed statistics.
6. Open `02_Crime_Classification_Training.ipynb`. Confirm class/source mapping and set
   `SOURCE_MAPPING_REVIEWED=True`. Enable `RUN_PROTOTYPE_TRAINING=True` only for the small prototype.
   Full training is **not approved or executed** in this task.
7. Run `03_Scenario_GRU_Training.ipynb` for the tiny explicitly synthetic neural pipeline test.
8. Run `04_Model_Evaluation.ipynb` after checkpoints exist to evaluate held-out data.

Run cells from top to bottom. Colab is interactive and Drive mounting requires your action.
If Drive mounting fails, resolve that error before training; the notebook does not silently
fall back to ephemeral storage in Colab. Outside Colab it uses ignored `artifacts/`.

## Repository layout

```text
notebooks/01_Dataset_Inspection.ipynb
notebooks/02_Crime_Classification_Training.ipynb
notebooks/03_Scenario_GRU_Training.ipynb
notebooks/04_Model_Evaluation.ipynb
scripts/data.py           # real-file inspection and grouped splitting
scripts/preprocess.py     # preprocessing CLI
scripts/classifier.py     # CNN, video sampling, training and evaluation
scripts/scenario.py       # GRU encoder-decoder, training and inference
scripts/evaluate.py       # held-out evaluation CLI
scripts/common.py         # reproducibility settings
configs/prototype.json
annotations/              # human annotation template + synthetic examples
models/                   # model documentation; weights ignored
reports/                  # validation evidence and remaining checks
tests/                    # temporary video and CPU model tests
requirements.txt
requirements-cloud-tested.txt
```

## Data and classifier design

DCSASS: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

Inspection checks `is_file()` before accepting video extensions, counts class-mapped and
unmapped videos, records extension counts and annotation candidates, and decodes the first
frame of each clip. This is not a full corruption scan: training raises an error if any
sampled frame is unreadable. Counts and actual annotation availability remain unknown until
notebook 01 executes against the dataset. Unmapped classes are excluded from classifier training.

Initial classes: **Burglary, Stealing, Fighting, Vandalism, Normal**. The three-layer CNN
starts with random weights and averages logits over three uniformly sampled RGB frames
(resized to 64×64 and scaled to [0,1]). It is an intentionally limited clip classifier,
not a temporal detector; category labels may include normal frames within crime clips.

Original `.mp4` parent directory names group segments. Flat `*_x264_N.mp4` names group by
`*_x264`. Unknown naming conventions require manually corrected source IDs. Filename grouping
does not prove source independence: review related videos and near duplicates before training.
The prototype uses one random representative clip per source, equal source counts per class,
up to 30 each. Each class needs at least three independent sources; otherwise it fails.
Groups split approximately 60/20/20 (train/validation/test), deterministically with seed 42.
Frames are decoded **after** splitting; the test split is never used for early stopping.

Training uses Adam, cross entropy, validation-loss early stopping, and best-checkpoint saving
as `crime_classifier.pth`. Epoch histories include real loss, accuracy, macro precision,
recall/F1, confusion matrices and per-class reports. The config and class order are saved
with the checkpoint. Record the dataset version, reviewed manifest and package versions
for each experiment. Exact reproducibility across GPU/platform/library versions is not guaranteed.

## Neural scenarios and detection feasibility

The randomly initialized embedding + GRU encoder-decoder uses teacher forcing during
training and greedy decoding for inference. Its vocabulary is learned only from training
pairs. The included synthetic examples validate execution, not useful scenario generation.
See `annotations/README.md` for the real paired annotation workflow. A sufficiently large,
human-reviewed evidence-to-scenario dataset is required before reliable generation can
be evaluated. Outputs can hallucinate or fail to express uncertainty and always need review.
This phase serializes hypothetical evidence as text; it does not implement visual/text fusion.

DCSASS bounding-box and narrative annotation availability is **unverified in this environment**.
Notebook 01 records annotation candidates for schema review. If boxes are unavailable,
a separate annotated detection dataset is required; do not fabricate boxes. No detector is trained.

## Checkpoint persistence

Colab writes best weights and epoch history under
`MyDrive/AI-Crime-Investigation-FYP/{classifier,scenario}`. Weights persist when Drive is
mounted; processes and the current epoch do not. Training reruns start from scratch and
can overwrite experiment outputs: use a new `OUTPUT` folder for a new experiment.
Optimizer-state continuation is not implemented. Saved models can be evaluated or used
for inference after reconnecting without retraining. Redownload DCSASS when needed; if its
cache path changes, regenerate/review the manifest with the same seed and **verify identical
source/test membership** before evaluation. Do not silently reassign the test set.

## Local CPU smoke tests

Python 3.12 was used in Codex. To reproduce this cloud CPU environment:

```bash
python3 -m venv /workspace/crime-venv
/workspace/crime-venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
/workspace/crime-venv/bin/pip install -r requirements-cloud-tested.txt
/workspace/crime-venv/bin/python -m pytest -q
/workspace/crime-venv/bin/python -m compileall -q scripts tests
```

`requirements-cloud-tested.txt` records installed versions for the cloud CPU run. Colab uses
`requirements.txt` ranges to preserve its GPU-compatible PyTorch installation. Export its
actual package versions alongside each real experiment before interpreting results.

CLI examples, after manually reviewing a real inspection manifest:

```bash
python -m scripts.preprocess inspect /path/to/dataset artifacts/inspection
python -m scripts.preprocess split artifacts/inspection/manifest.csv artifacts/splits.csv --per-class 30
python -m scripts.classifier artifacts/splits.csv --output artifacts/classifier
python -m scripts.evaluate cnn artifacts/splits.csv artifacts/classifier/crime_classifier.pth artifacts/test.json
python -m scripts.evaluate gru annotations/synthetic_scenarios.csv artifacts/scenario/scenario_gru.pth artifacts/gru-test.json
```

Do not run expensive full training without confirmation. Tests use generated temporary videos,
not real crime footage. See `reports/validation.md` for actual execution status.
