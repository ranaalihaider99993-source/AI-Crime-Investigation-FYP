# Shared image/video scenario pipeline

**2026-10-18 experimental update:** image uploads now execute a separate pinned
weak-video-label four-class CNN when its checkpoint is installed. This is not a
trained attribute/action model. Shared narratives still abstain from scene and
crime claims. Actual training/evaluation limits and Drive checkpoint location
are recorded in `reports/experimental_image_training.md`. Earlier descriptions
of absent image checkpoints below refer to the reviewed semantic model, which
remains unavailable.

## Actual capabilities

Both API outputs now expose the same `narrative` schema and both website sections
render it: visible observations, technical evidence with input IDs, possible
incident sequence, crime hypothesis and uncertainties. At present both abstain
from semantic claims. This is an evidence-bound deterministic renderer, not a
trained scenario language model. No synthetic GRU output is used.

Video: the existing pinned experimental four-category CNN still runs unchanged.
Sampled frame indices are technical evidence only. Three static frames and an
unvalidated category score cannot establish actions, timing or a crime. No
validated action analysis has been added or claimed. Input images: local bounded
decoding, hashes, dimensions and previews work for 1–6 images; semantic image
analysis remains unavailable. Upload order never establishes chronology.

## Executable from-scratch image training foundation

`scripts/image_evidence_training.py` implements a new random-initialized
three-block CNN with a multilabel head for ordered attributes: `person_visible`,
`vehicle_visible`, `smoke_visible`, `flame_visible`. These are visible attributes,
not crime labels or actions. No pretrained weights or external inference calls.
Unknown targets use masked BCE. CPU workers=0, batch=4, threads=2. It writes a new
run folder only; it never overwrites existing runs. Best validation-loss weights
are evaluated once on the held-out test split. Outputs: checkpoint, SHA-256,
config, epoch history and per-attribute TP/FP/FN/precision/recall. The 0-logit
evaluation threshold is uncalibrated and is not a deployment threshold.

The validation gate verifies actual decoded files and hashes, approved review,
explicit incident independence, provenance/license, human near-duplicate audit,
source-group separation and positive/negative coverage for every split/attribute.
Exact duplicate image bytes are rejected. A human audit is still required for
visually duplicate frames and related incidents with different identifiers.
Two source groups per split is a minimum plumbing gate, not adequate scientific
coverage. No canonical video manifests or existing reviews are altered.

Manifest JSON schema:

```json
{
  "schema_version": 1,
  "attributes": ["person_visible", "vehicle_visible", "smoke_visible", "flame_visible"],
  "near_duplicate_audit": "pending",
  "audit_reviewer": "",
  "images": []
}
```

Each image row requires: `image_id`, root-relative `path`, original-byte `sha256`,
`source_group` (all related source/incident media share one group), `split`
(`train`, `val`, `test`), `review_status` (`approved`), `reviewer`,
`incident_independence` (`verified`), `provenance`, `license`, and `labels` with
each ordered attribute mapped to 0, 1 or null. Approval fields must reflect real
human decisions, never inferred programmatically. Empty examples are not data.

After real annotation and review, run in the existing repository checkout:

```sh
python -m scripts.image_evidence_training --manifest /path/image_evidence.json --root /path/images --output /path/new-experiment --epochs 5 --device cuda
```

Use the new Colab notebook to validate first on CPU and optionally save a unique
run directly to mounted Drive. No GPU, authentication or dataset download is
needed until reviewed data exist. No trained image weights are auto-integrated:
integration requires pinned checkpoint hashes, verified config, held-out results,
validation-selected abstention thresholds and explicit deployment review.

## Temporal/action and scenario training blockers

DCSASS category labels do not provide verified frame-level attributes, object
tracks, temporal actions, incident sequence or narrative targets. The existing
smoke run included uncertain labels and unresolved independence; it cannot be
promoted to validated evidence. Real scenario template is empty; available text
examples are explicitly synthetic. No reviewed semantic image checkpoint exists
in the local repository. Remote Drive contents have not been re-audited in this
task; no new Drive model/result write is claimed.

For genuine video analysis, annotate observable actions with start/end times,
source/incident group, uncertainty and review provenance. Require ordered
decoded frames with validated timestamps; use a separate randomly initialized
CNN+temporal model (e.g. GRU), not the clip-level crime classifier as an action
detector. Keep all related video segments, extracted frames and image variants
in the same split. Require normal/out-of-domain inputs and evaluate temporal
localization, per-action metrics and abstention on incident-disjoint media.

Only supported, evaluated attributes/actions may enter future visible
observations, with image/frame IDs and timestamps. Attributes alone never imply
a crime; sequence needs temporal evidence, and hypotheses need reviewed
evidence-to-hypothesis annotations. Any learned language model requires real
reviewed grounded targets and unsupported-claim evaluation. Until those gates
are met, the shared renderer explicitly abstains rather than fabricating text.

Next required work is human annotation/review and a cross-modal source-group
audit. No training on real data is justified by the currently available records.
