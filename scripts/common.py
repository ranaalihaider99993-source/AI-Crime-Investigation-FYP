import random
import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)


def experiment_metadata(input_path, seed):
    import hashlib
    import platform
    import subprocess
    from pathlib import Path
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        revision='unknown'
    return dict(seed=seed,input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                cuda=torch.version.cuda,git_commit=revision)
