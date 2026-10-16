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
   `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/inspection`. Inspect **all** candidate annotation files;
   the notebook previews only the first 20. Reports contain observed counts, not assumed statistics.
6. If notebook 01 already produced the schema-v2 files, reopen the latest
   [notebook 02 in Colab](https://colab.research.google.com/github/ranaalihaider99993-source/AI-Crime-Investigation-FYP/blob/main/notebooks/02_Crime_Classification_Training.ipynb),
   choose a T4 runtime, and **Run all with both flags False**. Notebook 02 now reuses the
   saved inspection, automatically prepares a candidate split, and displays a PASS/FAIL
   report and selected-frame review sheet. It does not approve mappings or start training.
7. Review `smoke_preparation/validation.md`, `source_review.csv`, and `review_sheet.jpg`.
   Check class assignments, original-source conventions and related incidents across source
   IDs. If automatic checks fail or the mapping cannot be verified, leave both flags False.
8. Only after your review, set **both** `SOURCE_MAPPING_REVIEWED=True` and
   `RUN_PROTOTYPE_TRAINING=True` in the launch cell, then run that cell and the result cell.
   The GPU-required launcher permits only **2 epochs / 3 sources per class**, creates fresh
   random weights, and evaluates held-out sources. It rejects changed reports/data, missing
   GPUs and expanded training configurations. This 12-clip experiment checks execution only.
9. Actual checkpoint paths, accuracy, macro F1, confusion matrix and per-class results are
   printed after successful execution. Set `DOWNLOAD_CHECKPOINT=True` in the result cell
   to download `crime_classifier.pth` for later Django integration. Notebook 04 can repeat
   evaluation using the latest successfully completed smoke run.
10. Notebook 03 is an **optional, separate synthetic GRU pipeline test**. It is unnecessary
    for CNN training/evaluation and provides no real scenario-generation evidence.

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
scripts/smoke.py          # automatic report, human gate and bounded GPU launch
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

Current prototype classes: **Burglary, Stealing, Fighting, Vandalism**.
`Normal` and all other categories are unmapped and excluded from training for now.
The four-class CNN rejects checkpoints with the old five-class mapping. The three-layer CNN
starts with random weights and averages logits over three uniformly sampled RGB frames
(resized to 64×64 and scaled to [0,1]). It is an intentionally limited clip classifier,
not a temporal detector; category labels may include normal frames within crime clips.

Inspection strips repeated leading `DCSASS Dataset` wrappers case-insensitively and
lowercases the remaining relative clip key. Thus `DCSASS Dataset/...` and
`dcsass dataset/DCSASS Dataset/...` identify the same clip. It preserves category and
source directory components: equal segment basenames from different sources are distinct.
Among copies, it retains the shortest relative path, then case-insensitive/exact lexical
order. SHA-256 verifies every excluded copy; conflicting bytes stop inspection and write
`duplicate_conflict.json`. Hashing scans duplicate files fully and can take time; decoding
runs only for the retained copies. This is path-identity deduplication, not detection of
similar scenes stored under unrelated names. The new selected-clip preflight also rejects
byte-identical videos across different source IDs/splits, but cannot detect re-encoded duplicates. `excluded_duplicates.csv` preserves each
excluded path, selected path, canonical key, hash, and reason. No dataset files are deleted.

Original `.mp4` parent directory basenames identify source videos; flat
`*_x264_N.mp4` names group by `*_x264`. IDs are canonical lowercase, independent of dataset
root and category directories. Conflicting source labels, duplicate canonical/physical
clips, missing files, invalid class counts, legacy manifests, and cross-split source leakage
fail explicitly. Unknown naming conventions or near-duplicate incidents still need review.
Regenerate **all** old manifests and splits: schema v2 requires `canonical_clip`.

The prototype uses one seeded representative clip per original source, equal source counts
per class. Sources and clip candidates are sorted before seeded selection, so manifest row
order cannot change the split. Selection is capped at the least available readable class.
Each class needs at least three independent sources; otherwise it fails. The split policy
assigns `max(1, floor(0.2*n))` sources each to validation/test and the rest to training.
Frames are decoded after splitting; test data are not used for early stopping.

`candidate_splits.report.json` in notebook 02 (or `<split-file>.report.json` from the CLI)
records available readable clips/sources, exclusions, selected
counts **per class and per split**, and the actual source-intersection lists (all must be
empty). The notebook smoke uses 3 sources per class: **1 train / 1 validation / 1 test**,
12 clips total. If 8 sources per class are selected, counts are **6 / 1 / 1**. A held-out
Fighting clip from only one source cannot establish reliable performance. Enlarging other
classes does not create additional independent Fighting sources.

### User-reported DCSASS inspection (not independently measured here)

The user found **33,278 raw rows, 16,639 unique clips, and 520 canonical original sources**.
The duplicate dataset-root copies explain the doubled rows.

| Mapped category | Unique clips | Original sources |
| --- | ---: | ---: |
| Burglary | 1,504 | 47 |
| Stealing | 2,048 | 64 |
| Fighting | 256 | 8 |
| Vandalism | 928 | 29 |

The remaining **11,903 unique clips are unmapped**. These observations are not hard-coded
assertions about future Kaggle versions. Regenerated `inspection.json` reports the actual
current raw/unique counts, all-source counts, mapped counts, unmapped source/clip totals and
unmapped category-name candidates separately. No real-data accuracy is claimed.

Training uses Adam, cross entropy, validation-loss early stopping, and best-checkpoint saving
as `crime_classifier.pth`. Epoch histories include real loss, accuracy, macro precision,
recall/F1, confusion matrices and per-class reports. The config and class order are saved
with the checkpoint. A canonical split-membership hash prevents evaluation against changed
clip/source/label/split assignments, even if physical cache paths change. Record the dataset version, reviewed manifest and package versions
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

The notebook 02 launcher writes each reviewed run under
`MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/classifier/runs/<timestamp>-<id>/`.
The best checkpoint, epoch history, exact split snapshot, experiment metadata, review
attestation and actual test metrics are retained there. A `latest_completed.json` pointer
is updated **only after training and evaluation both succeed**. Interrupted/failed runs
cannot become the latest completed experiment. Earlier successful runs are not overwritten
or displayed as a new result after a failed invocation. Training always starts from scratch.
The legacy CLI trainer still writes directly to its specified output folder; use a new
folder to preserve previous CLI experiments. GRU outputs remain in the separate `scenario/` folder.
Drive retains files after disconnect; processes and the in-progress epoch do not persist.
Optimizer-state continuation is not implemented. Saved models can be evaluated or used
for inference after reconnecting without retraining. Redownload DCSASS when needed; if its
cache path changes, regenerate/review the manifest with the same seed and **verify identical
source/test membership** before evaluation. Do not silently reassign the test set. The new Drive subfolder isolates corrected
manifests and checkpoints from the old duplicated/five-class artifacts. In a reused Colab
runtime, notebooks 01/02 fast-forward a clean checkout and evict stale project module
imports; local checkout edits cause an error instead of being overwritten.

## What the automatic report does and does not verify

`smoke_preparation/validation.json` and `validation.md` separate **automatic PASS/FAIL**,
**human mapping review PENDING**, and **no training executed**. Preparation checks physical
paths against canonical IDs, unique clips, the four-class mapping and source counts,
saved duplicate-audit consistency, at least three readable recognized sources per class,
zero canonical source overlap, all sampled frame decoding and byte-identical leakage
among the selected clips. Very short clips reuse decoded frames for repeated sample indices.

The existing notebook 01 duplicate SHA-256 records are checked for consistency with counts,
current paths and file sizes; the full dataset is not rehashed. If its contents changed,
rerun notebook 01. Only the 12 selected clips are hashed again; their hashes, candidate split
and inspection artifacts are rechecked before training to bind review to unchanged inputs.
Unknown source naming fails automatically rather than assuming independence. The review
sheet helps visual inspection but cannot certify labels or identify all related incidents.
Preparation never sets an approval flag, saves trained weights or creates evaluation metrics.

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
python -m scripts.preprocess split artifacts/inspection/manifest.csv artifacts/splits.csv --per-class 3
python -m scripts.classifier artifacts/splits.csv --output artifacts/classifier
python -m scripts.evaluate cnn artifacts/splits.csv artifacts/classifier/crime_classifier.pth artifacts/test.json
python -m scripts.evaluate gru annotations/synthetic_scenarios.csv artifacts/scenario/scenario_gru.pth artifacts/gru-test.json
```

Do not run expensive full training without confirmation. Tests use generated temporary videos,
not real crime footage. See `reports/validation.md` for actual execution status.
