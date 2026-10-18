# Combined incident prototype verification

Verified locally 2026-10-18 with one Flask server and one controlled browser.
No training, Colab session, dataset changes or review writes.

- **53 Python tests passed** across incident fusion/API, existing image/video
  inference, image decoder, reviewed data gates and model/notebook regressions.
- **7 JavaScript score-policy tests passed**.
- Image-only browser submission: two identical real sample-derived PNGs;
  three ranked cards, duplicate deweighting and collapsed Advanced Details.
- Video-only browser submission: real sample MP4 plus a differently encoded
  15-frame AVI excerpt; both distinct byte streams analyzed successfully.
- Mixed browser submission: real sample-derived image + both videos; three
  files, both modalities represented. Recomputed equal-modality fusion error
  from displayed technical records: **0**.
- Mixed scores: Burglary **28.16285%**, Vandalism **24.68292%**, Stealing
  **24.62072%**, Fighting **22.53350%**. Top three displayed in that order;
  overall **Inconclusive / Low confidence**. These are uncalibrated relative
  scores, not crime probabilities.
- At 390×844: no horizontal overflow; actual footer reachable with Advanced
  Details both collapsed and expanded. Technical details collapsed by default.
- Screenshots: `incident-desktop.png`, `incident-mobile.png`.

The sample image and AVI are derived from the existing real sample video, not
independent incident evidence or new evaluation data. This tests integration
only. The tiny experimental model limitations remain unchanged.

Original video checkpoint SHA-256 remains
`a31794fd43f0a736d63702806a0c838af266b3536867a65da4cd74b4082b5ab6`;
image checkpoint remains
`e95b7ebf04510e78bbbde4e7716408cbdd1666291a7406ed42dcf26878e647a8`.

See `docs/incident_analysis.md` for aggregation, limits and interpretation.
Home page is the new combined flow; `/individual` preserves the original tools,
and all existing inference endpoints remain available. No useful crime-scene
accuracy or event reconstruction is established by these integration checks.
