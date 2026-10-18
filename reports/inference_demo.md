# Local inference prototype — verified execution evidence

Implemented and tested in the existing repository using the saved experimental
CNN. No retraining, pretrained model, AI API, scenario generation, canonical split
change or human-review modification was performed for inference.

## Launch

From the repository root, with the installed workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Open http://127.0.0.1:8000. The local QA server is stopped after testing; this command
starts it again. See [setup and model retrieval](../web/README.md) for another machine.
The checkpoint and real sample are intentionally ignored by Git, and are present
in this workspace. No authentication is required for local inference.

## Checkpoint contract verified

- SHA-256: `a31794fd43f0a736d63702806a0c838af266b3536867a65da4cd74b4082b5ab6`.
- Existing `CrimeCNN`, 23,844 parameters; strict state-dictionary keys and shapes.
- Four classes, in order: Burglary, Stealing, Fighting, Vandalism.
- Configuration: three 64×64 RGB frames; CHW float32 divided by 255.
- Uniform indices use NumPy linspace with integer truncation. Resizing uses
  OpenCV's default linear interpolation, matching the original training code.
- Mean frame logits followed by softmax; CPU eval/inference mode.
- Loaded with `weights_only=True`; mismatch and non-finite weights fail closed.
- Training and inference now call the same frame-preprocessing helper. Tests also
  independently reproduce the original preprocessing rather than just comparing
  two calls to the shared helper.

## Real-video end-to-end test

Retrieved only the checkpoint and `Burglary046_x264_15.mp4` from the authenticated
Colab session. The clip's SHA-256 matched the existing saved validation evidence:
`7806099ee66ab2c582c317ce7c88149b1c63fedb96bcee10b8a3a0198a26256a`.
The original sample is 35,759 bytes, 60 frames and approximately two seconds.

Actual HTTP multipart upload returned 200. Sampled frame indices: **0, 29, 59**.
The browser file chooser → upload → Analyze → result flow also completed.

| Model category | Uncalibrated relative score |
| --- | ---: |
| Burglary | 26.412579% |
| Stealing | 25.470093% |
| Fighting | 22.981463% |
| Vandalism | 25.135863% |

These are actual saved-model outputs, **not validated accuracy or probabilities
that a crime occurred**. The interface prominently states UNVALIDATED EXPERIMENTAL,
explains the lack of Normal/Unknown categories, and generates no crime scenario.

- [Desktop screenshot](inference_demo.png)
- [Mobile screenshot](inference_mobile.png)
- [Machine-readable response and provenance](inference_result.json)

Desktop and 390-pixel mobile layouts had no horizontal overflow. Three sampled
input images and four score bars rendered. An invalid MP4 produced a readable
error, hid stale results and allowed retry. Missing upload, unsupported extension,
corrupt video and upload-limit handling are also covered by API tests. Videos are
processed in a temporary directory and removed after the request.

## Tests

- **153 passed** in Colab Linux CPU, full regression suite (41.32 seconds).
- **13 passed** on local Windows Python 3.14, inference and existing model tests
  (47.73 seconds), including the actual retrieved checkpoint and real sample.
- JavaScript syntax, Python compilation, and Git whitespace checks passed.

Screenshots and response records are evidence of execution only. Existing
Uncertain/Rejected labels and Pending incident-independence decisions are preserved.
