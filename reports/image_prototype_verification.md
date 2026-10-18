# Image workflow verification

Verified locally on 2026-10-18 using the existing Flask framework, CPU only,
one controlled browser, and sequential tests. No Colab session was started.

- Python image/API, video inference and model tests: **20 passed**.
- Existing JavaScript score-policy tests: **7 passed**.
- Browser multi-image upload: two previews decoded at natural width 320;
  response reported two 320×240 images with SHA-256 provenance and an evidence
  brief explicitly withholding scene/crime hypotheses.
- Browser single-image upload: one image inspected successfully.
- Mobile 390×844: no horizontal overflow; actual footer accessible at bottom.
- Desktop/mobile full-page screenshots: `image-prototype-desktop.png` and
  `image-prototype-mobile.png` alongside this report.
- Browser video regression: existing real sample uploaded and analyzed with
  three sampled input frames and the existing inconclusive display policy.

Test input: first decoded frame of the existing local sample video. Original
source clip `Burglary046_x264_15.mp4`, source group `burglary046_x264`; sample
video SHA-256 `7806099ee66ab2c582c317ce7c88149b1c63fedb96bcee10b8a3a0198a26256a`.
The frame is only an end-to-end test input, not a verified image annotation.
Dataset media are not committed.

No image semantic model was run. The generator is deterministic and reports
technical file observations only. No trained image checkpoint or reviewed
image attribute dataset was found locally. See `docs/image_prototype.md` for
the independent from-scratch training design and data requirements.

Existing video weights, review decisions, canonical manifests/splits and
production training flags were not changed. No training was performed.
