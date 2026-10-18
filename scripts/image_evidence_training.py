"""Independent, fail-closed, from-scratch multilabel image experiment.

No automatic dataset download, split creation, review edit or web integration.
Use a reviewed JSON manifest; see docs/shared_evidence_pipeline.md.
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
from scripts.image_analysis import inspect_image, MAX_BYTES

ATTRIBUTES = ['person_visible', 'vehicle_visible', 'smoke_visible', 'flame_visible']
PREPROCESSING = dict(size=128, color='RGB', resize='bilinear_aspect_pad_black',
                     orientation='EXIF_transpose', layout='CHW', scale='float32/255')


def preprocess(data):
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), (128, 128),
                             method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


class ImageEvidenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, len(ATTRIBUTES)))

    def forward(self, images):
        return self.net(images)


def validate_manifest(path, root):
    path, root = Path(path), Path(root).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('attributes') != ATTRIBUTES or manifest.get('schema_version') != 1:
        raise ValueError('Image attribute/schema contract mismatch')
    if manifest.get('near_duplicate_audit') != 'approved' or not manifest.get('audit_reviewer'):
        raise ValueError('A human near-duplicate audit is required')
    rows = manifest.get('images', [])
    if not rows:
        raise ValueError('No reviewed image data available')
    groups, hashes, ids = {}, {}, set()
    for row in rows:
        if (row.get('review_status') != 'approved' or not row.get('reviewer') or
            row.get('incident_independence') != 'verified' or not row.get('provenance') or
            not row.get('license') or not row.get('source_group')):
            raise ValueError('Unreviewed evidence, independence or provenance')
        split = row.get('split')
        if split not in {'train', 'val', 'test'}:
            raise ValueError('Explicit train/val/test split required')
        if not row.get('image_id') or row['image_id'] in ids:
            raise ValueError('Missing or duplicate image ID')
        ids.add(row['image_id'])
        file = (root / row['path']).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Image path is outside dataset root or missing')
        with file.open('rb') as stream:
            data = stream.read(MAX_BYTES + 1)
        inspect_image(data, 1)
        digest = hashlib.sha256(data).hexdigest()
        if digest != row.get('sha256'):
            raise ValueError('Image SHA-256 mismatch')
        if digest in hashes:
            raise ValueError('Duplicate image content; deduplicate before training')
        hashes[digest] = split
        group = row['source_group']
        if group in groups and groups[group] != split:
            raise ValueError('Cross-split incident/source leakage')
        groups[group] = split
        labels = row.get('labels', {})
        if set(labels) != set(ATTRIBUTES) or any(type(v) not in {int, type(None)} or v not in {None, 0, 1} for v in labels.values()):
            raise ValueError('Attributes require 0, 1 or null (unknown)')
        if all(v is None for v in labels.values()):
            raise ValueError('At least one reviewed attribute is required per image')
        row['_file'] = file
    for split in ['train', 'val', 'test']:
        subset = [r for r in rows if r['split'] == split]
        if len({r['source_group'] for r in subset}) < 2:
            raise ValueError('At least two independent source groups per split required (not a sufficiency guarantee)')
        for attribute in ATTRIBUTES:
            if not {0, 1}.issubset({r['labels'][attribute] for r in subset}):
                raise ValueError(f'{split}: missing reviewed positive/negative support for {attribute}')
    return rows


class EvidenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        data = row['_file'].read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Image changed after verification')
        labels = [row['labels'][a] for a in ATTRIBUTES]
        return preprocess(data), torch.tensor([v or 0 for v in labels], dtype=torch.float32), torch.tensor([v is not None for v in labels])


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, count = 0., 0
    tp, fp, fn = [torch.zeros(len(ATTRIBUTES), dtype=torch.long) for _ in range(3)]
    with torch.set_grad_enabled(optimizer is not None):
        for images, target, mask in loader:
            images, target, mask = images.to(device), target.to(device), mask.to(device)
            logits = model(images)
            if not mask.any(): continue
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')[mask].mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * mask.sum().item(); count += mask.sum().item()
            predicted = logits.detach() >= 0  # Reporting only, not a calibrated deployment threshold.
            truth = target.bool()
            tp += (predicted & truth & mask).sum(0).cpu()
            fp += (predicted & ~truth & mask).sum(0).cpu()
            fn += (~predicted & truth & mask).sum(0).cpu()
    return dict(masked_bce=loss_sum / max(count, 1), annotated_targets=count,
                per_attribute={a:dict(tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
                    precision=float(tp[i] / max(int(tp[i]+fp[i]), 1)),
                    recall=float(tp[i] / max(int(tp[i]+fn[i]), 1))) for i, a in enumerate(ATTRIBUTES)})


def train(manifest, root, output, epochs=5, device='cpu', seed=42):
    if not 1 <= epochs <= 100: raise ValueError('Epoch count must be 1–100')
    # Validate before creating an output directory or allocating a GPU model.
    rows = validate_manifest(manifest, root)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
    if device not in {'cpu', 'cuda'}: raise ValueError('Use cpu or cuda')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model = ImageEvidenceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loaders = {split: torch.utils.data.DataLoader(EvidenceDataset([r for r in rows if r['split']==split]),
               batch_size=4, shuffle=split=='train', num_workers=0) for split in ['train','val','test']}
    config = dict(architecture='ImageEvidenceCNN-v1', attributes=ATTRIBUTES, preprocessing=PREPROCESSING,
        random_initialization=True, pretrained=False, validated=False, deployment_approved=False,
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        seed=seed, epochs=epochs, device=device, learning_rate=.001, batch_size=4,
        action_analysis_available=False, narrative_model_trained=False)
    (output/'verified_manifest.json').write_bytes(Path(manifest).read_bytes())
    (output/'config.json').write_text(json.dumps(config, indent=2))
    best, history = float('inf'), []
    for number in range(epochs):
        record = dict(epoch=number+1, train=epoch(model, loaders['train'], device, optimizer),
                      val=epoch(model, loaders['val'], device))
        history.append(record)
        if record['val']['masked_bce'] < best:
            best = record['val']['masked_bce']
            torch.save(dict(state_dict=model.state_dict(), config=config, epoch=number+1), output/'image_evidence.pth')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    saved = torch.load(output/'image_evidence.pth', map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    (output/'test_metrics.json').write_text(json.dumps(epoch(model, loaders['test'], device), indent=2))
    (output/'checkpoint.sha256').write_text(hashlib.sha256((output/'image_evidence.pth').read_bytes()).hexdigest())
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    train(args.manifest, args.root, args.output, args.epochs, args.device)
