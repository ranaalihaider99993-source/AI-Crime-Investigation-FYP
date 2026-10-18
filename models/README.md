Randomly initialized model definitions live in scripts/classifier.py and scripts/scenario.py.
Checkpoints are ignored. Colab saves best validation checkpoints to Google Drive.

The separate image classifier is `WeakFrameCNN` in `scripts/experimental_image.py`.
Install its saved `experimental_image_cnn.pth` at `models/experimental_image_cnn.pth`.
The Flask app checks SHA-256, architecture, classes, preprocessing and finite
weights before enabling it. If missing/invalid, image technical inspection still
works. Do not rename the video checkpoint to replace the image checkpoint.

Drive location (relative to `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2`):
`experimental_image_cnn/20261018T094503Z-bb078083/experimental_image_cnn.pth`.
SHA-256: `e95b7ebf04510e78bbbde4e7716408cbdd1666291a7406ed42dcf26878e647a8`.
This weak-label experimental model is not validated crime-scene understanding.
