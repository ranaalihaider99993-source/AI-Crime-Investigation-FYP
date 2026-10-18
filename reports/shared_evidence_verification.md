# Shared video/image evidence workflow verification

Implemented and verified locally on 2026-10-18. One existing browser and one
Flask server were reused. No GPU runtime, pretrained weights, external AI
inference API, or dataset downloads were used.

## Results by capability

| Capability | Verified behavior | Remaining blocker |
|---|---|---|
| Video | Existing pinned CNN prediction, three frame previews, new shared structured narrative | Category model remains unvalidated; no reviewed temporal action data or validated action model |
| 1–6 images | Local previews, bounded decoding, dimensions, hashes, new shared structured narrative | No reviewed image attribute data or trained semantic image model |
| Narratives | Both workflows render observations, possible sequence, crime hypothesis and uncertainties; both abstain from unsupported claims | No real reviewed evidence-to-scenario targets; synthetic examples are not sufficient |
| Image training | Random CNN, reviewed manifest gates, masked BCE, validation checkpoint selection, held-out metrics and reproducibility artifacts implemented | Real data gates cannot currently pass |

## Tests and browser evidence

- 35 Python checks passed across the regression run and notebook recheck: shared schema, fail-closed data gates, fixture-only
  CPU training/checkpoint reload, image API, existing video/model regressions.
  The original four-notebook inventory assertion was updated for the new fifth
  notebook; its final validation and code-compilation check passed separately.
- Seven existing JavaScript score-policy tests passed.
- Real existing sample video: `Inconclusive / Low confidence`; narrative cites
  frame indices 0, 29, 59 as technical evidence and abstains from semantic claims.
- Two real sample-derived images: both 320×240; response reports two inputs and
  explicit lack of a semantic model. Image SHA-256:
  `851fe6ab3c5de04b9e09e15c0f79be50259f9fc9dbddb16fd50fb0015a234d99`.
- Desktop/mobile screenshots are saved alongside this report. Mobile actual
  bottom and horizontal overflow checked in the controlled browser.

The CPU fixture uses colored squares with test-only labels to verify code
execution. Its weights/metrics are temporary test artifacts, not a useful model,
not an evaluation of visual understanding, and not deployed or uploaded to Drive.

## Evidence for the training stop

`reports/experimental_cnn_smoke.json`: existing run is explicitly unvalidated,
with 2 Confirmed and 7 Uncertain clips; all 12 incident answers remain Pending in
that saved report. `annotations/scenarios_template.csv` is header-only, and
`annotations/synthetic_scenarios.csv` is explicitly synthetic. Local `models/`
contains only the existing experimental video checkpoint. These records do not
supply verified visible attributes, temporal actions or scenario ground truth.
Drive was not re-audited and no new real model/result save to Drive is claimed.

New `notebooks/05_Reviewed_Image_Evidence.ipynb` is unexecuted, disabled by default,
validates reviewed data on CPU first and writes only a new unique Drive run when
explicitly enabled with suitable inputs. It does not regenerate video splits,
download data or change production flags. Existing checkpoints/reviews preserved.

Next: create genuinely reviewed attribute and temporal-action annotations with
cross-modal incident/source groups and a visual duplicate audit. Existing Pending
reviews cannot be treated as approved. Then execute the independent Colab image
experiment, evaluate and calibrate abstention before any semantic integration.
