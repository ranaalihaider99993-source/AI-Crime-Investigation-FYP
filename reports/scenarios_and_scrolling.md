# Readable scenarios and scrolling verification — 2026-10-18

Three ranked cards now provide a readable title, relative score, conditional
possible explanation, CNN category basis and scenario-specific uncertainty.
Conditional templates do not claim observed actions, suspects, motives or
timelines. No free-form AI model/API is used. Existing CNN outputs and score
aggregation are unchanged. These explanations are suitable for demonstrating
the prototype, not for establishing incident facts.

## Scrolling investigation

The controlled browser initially had pending native file-picker modals, which
block normal page interaction. After cancelling them, the baseline no-upload
page had visible HTML/body overflow, no nested scroll/clipping containers and
real wheel scrolling reached the footer (page 1000px at viewport height 720px).
A persistent CSS scroll lock could not be reproduced.

The stylesheet now explicitly keeps the document at automatic height with root
vertical scrolling and normal body flow. Incident upload spacing is more compact:
the no-upload desktop page is 881px, down from 1000px, without introducing a
viewport-height spacer or fixed-height page container. Empty preview/status
elements add no space. Responsive footer wrapping and minimum-width handling
avoid narrow-screen clipping. Native file-picker modal behavior is browser/OS
behavior; it is not represented as an application CSS fix.

## Actual validation

- **57 Python tests passed**, covering conditional narratives for all four
  categories, unchanged fusion, duplicate handling, invalid/no uploads, real
  image/video/mixed inference and model/data/notebook regressions.
- **7 existing JavaScript score-policy tests passed**.
- **20 real Chrome wheel-scroll assertions passed** using the existing browser:
  desktop 1280×720 and mobile viewport 390×844; image-only, two-video and mixed
  inputs; before upload, after selection, after results, and mixed results with
  Advanced Details expanded. All reached the actual document bottom with the
  footer visible, no horizontal overflow and no nested scroll/clipping container.
- Before upload: analysis disabled and results hidden. After selection: preview
  count matches inputs. After analysis: exactly three readable scenario cards,
  each with model basis and uncertainty, with technical details collapsed.
- Reusable browser check: `tests/incident_browser_checks.js` through the existing
  Playwright MCP `browser_run_code_unsafe` filename tool. It only interacts with
  the local page. Measurements: `scenario_scroll_checks.json`.
- Screenshots: `scenarios-scroll-1280.png`, `scenarios-scroll-390.png`.

QA media are the existing real sample video, its frame PNG and a re-encoded short
excerpt. They test integration only, not independent crime-model accuracy.
No model/review/split/training-flag changes, retraining or Colab session occurred.

Future actual event understanding requires reviewed visual/action annotations,
incident-independent evaluation and calibrated abstention. A learned narrative
model would additionally require reviewed evidence-to-narrative targets. Current
readable explanations are conditional templates over unvalidated category scores.
