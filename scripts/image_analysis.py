"""Bounded image inspection and deterministic evidence briefs; no semantic AI."""
import hashlib
import io
import warnings
from PIL import Image, ImageOps
from scripts.evidence_narrative import build_narrative

MAX_IMAGES = 6
MAX_BYTES = 8 * 1024**2
MAX_PIXELS = 16_000_000
WARNING = 'UNVALIDATED EXPERIMENTAL — no trained image understanding model is installed.'


def inspect_image(data, index):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Each image must be nonempty and at most 8 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in {'PNG', 'JPEG', 'WEBP'}:
                    raise ValueError('Only JPEG, PNG and WebP images are supported.')
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError('Each image must contain at most 16 million pixels.')
                if getattr(original, 'n_frames', 1) != 1:
                    raise ValueError('Animated images are not supported.')
                image = ImageOps.exif_transpose(original).convert('RGB')
                width, height = image.size
                # A bounded thumbnail is also the future training preprocessing input.
                image = ImageOps.pad(image, (128, 128), method=Image.Resampling.BILINEAR,
                                     color=(0, 0, 0))
                output = io.BytesIO()
                image.save(output, format='PNG')
                import base64
                preview = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
    except (OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Unreadable or unsafe image.') from error
    return dict(image_id=f'image-{index}', sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data), width=width, height=height, preview=preview)


def analyze_images(payloads):
    if not 1 <= len(payloads) <= MAX_IMAGES:
        raise ValueError('Choose between 1 and 6 images.')
    images = [inspect_image(data, i + 1) for i, data in enumerate(payloads)]
    narrative = build_narrative('images', [dict(id=item['image_id'], description=
        f"Decoded still image: {item['width']} × {item['height']} pixels; SHA-256 {item['sha256']}.") for item in images])
    return dict(warning=WARNING, semantic_model_available=False, images=images, narrative=narrative,
                preprocessing='EXIF orientation → RGB → bilinear aspect-preserving black pad 128×128; future tensor CHW float32 /255.',
                scenario=dict(
                    title='Experimental evidence brief — not a crime reconstruction',
                    observations=[f"{item['image_id']}: decoded still image, {item['width']} × {item['height']} pixels; SHA-256 recorded." for item in images],
                    hypotheses=['No scene or crime hypothesis is supported by this technical inspection.'],
                    uncertainty=['Objects, actions, people, chronology and relationships between images have not been determined.',
                                 'No suspects, motives, crime occurrence or incident independence can be inferred.',
                                 'Human visual review and a separately trained, evaluated image model are required.']))
