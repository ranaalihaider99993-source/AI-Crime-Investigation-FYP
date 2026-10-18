"""Shared training/inference preprocessing: uniform indices, RGB, CHW, [0,1]."""
import cv2
import numpy as np
import torch


def sample_video_frames(path, size=64, frames=3):
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count < 1:
            raise ValueError('Video cannot be decoded or has no indexed frames')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        indices = np.linspace(0, max(0, count-1), frames).astype(int).tolist()
        decoded, tensors, previews = {}, [], []
        for index in indices:
            if index not in decoded:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode sampled frame {index}')
                rgb = cv2.cvtColor(cv2.resize(frame, (size,size)), cv2.COLOR_BGR2RGB)
                decoded[index] = (torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255, rgb)
            tensor, rgb = decoded[index]
            tensors.append(tensor)
            previews.append(rgb)
        return torch.stack(tensors), previews, dict(frame_count=count, sampled_indices=indices,
            duration_seconds=count/fps if np.isfinite(fps) and fps>0 else None,
            size=[size,size], color='RGB', tensor_layout='TCHW', scale='float32 / 255')
    finally:
        capture.release()
