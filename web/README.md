# CrimeScope AI experimental inference

Local CPU web prototype using the existing **UNVALIDATED EXPERIMENTAL** checkpoint.
No training, scenario generation, review changes, or external AI service calls.

## Launch on Windows

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m web.app
```

Open **http://127.0.0.1:8000**. The server listens only on local loopback.
Use `--port 8001` or `--checkpoint <path>` if needed. Waitress serves the app;
Flask debug mode is disabled. Uploads have a 64 MB limit and are removed after
each request. Videos are not stored in the repository or sent to an AI service.

The approved checkpoint is already present locally as
`models/experimental_cnn.pth` (ignored by Git). On another machine, retrieve that
file from your authorized Google Drive folder:

`AI-Crime-Investigation-FYP/canonical-four-class-v2/experimental_cnn_smoke/20261018T052957Z-84209c36/experimental_cnn.pth`

Required SHA-256:
`a31794fd43f0a736d63702806a0c838af266b3536867a65da4cd74b4082b5ab6`.
Startup fails if the checksum, class order, preprocessing configuration,
architecture/state shapes, experimental provenance or finite weights fail checks.
There is no fallback to random or pretrained weights. Do not commit checkpoint
files, dataset clips, credentials, or Drive authentication tokens.

## Model contract

- Architecture: existing `CrimeCNN`; three convolutions (3→16→32→64), pooling,
  adaptive average pooling, and Linear 64→4.
- Class order: **Burglary, Stealing, Fighting, Vandalism**.
- Three frame indices from `numpy.linspace(0, frame_count-1, 3).astype(int)`.
- OpenCV linear resize to 64×64, BGR→RGB, CHW float32 divided by 255.
- Average the three frame logits; then apply softmax across the four categories.
- Training and inference share `scripts/video_frames.py`.
- CPU evaluation with `model.eval()` and `torch.inference_mode()`.

Scores are uncalibrated relative model outputs, not probabilities of crime.
The model has no Normal/Unknown class and always chooses one of four categories.
It does not determine incidents or identify people. Its training included seven
Uncertain labels by explicit authorization; three Rejected clips were excluded.
The two-clip experimental evaluation is not validated accuracy.

## Experimental result display

Only the highest-scoring category and its relative score are prominent by default.
The native **View all model scores** disclosure starts closed and exposes all four
scores, including the other three categories, when expanded. It closes again for
each new result.

The UI labels a result **Inconclusive / Low confidence** if **either** the top
softmax score is below **0.50 (50%)**, **or** the top-minus-runner-up margin is below
**0.10 (10 percentage points)**. Equality at either boundary passes that boundary;
both conditions must pass for the top category to be the main heading. Inconclusive
results still identify the highest-scoring category as secondary information.

These are conservative, display-only experimental cutoffs chosen to flag weak or
closely matched outputs. They have not been calibrated or validated against a
representative labeled dataset. Passing them does not establish confidence in a
crime, improve model accuracy, or confirm an incident. Percentages are relative
model scores, not true probabilities of crime. Raw backend outputs, CNN weights,
preprocessing and human review gates are unchanged.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_inference.py tests/test_models.py
node --test tests/test_score_policy.cjs
```

Tests verify original preprocessing parity, logit aggregation, checkpoint checks,
upload errors, the warning, and inference using the actual saved checkpoint and a
real sample when those ignored assets are present. Fixture-only tests use generated
videos/weights, and are not presented as real-data inference evidence.
See `reports/inference_demo.md` for actual browser and API evidence.
