"""Public 224-pixel three-network SIIM-ISIC retraining from DeGrave et al."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from backbone_resolution_study import download_file_curl
from inference_store import digest
from prove_ai_adae import _installed


NAME = "siim_retrained"
COMMIT = "98adb0e6ec9e181ffe2fec92fc178e84a0bf20dd"
SOURCE = f"https://raw.githubusercontent.com/suinleelab/derm_audit/{COMMIT}/models/siim_isic.py"
SOURCE_SHA = "a5bfc5661c85f79619065dfc68446b665f017bc31cbdcb0e8907b03a6a9e7407"
WEIGHTS = {
    "9c_b7ns_1e_224_ext_15ep_best_fold0.pth": (256728206, "d4deef8fe0e209ebabaeaf195cd3624a"),
    "9c_b6ns_1e_224_ext_15ep_best_fold1.pth": (164124631, "c8704226ba194e6488f0fa37416c00bb"),
    "9c_b5ns_1e_224_ext_15ep_best_fold2.pth": (114301894, "1fecdc65b87ddb88d4a8c0f804b86077"),
}
PROVENANCE = {"publication": "https://doi.org/10.1038/s41551-023-01160-9",
              "weights": "https://zenodo.org/records/10049217",
              "published_weights": {name: {"bytes": size, "md5": md5} for name, (size, md5) in WEIGHTS.items()},
              "code": SOURCE, "source_sha256": SOURCE_SHA, "dependency": "geffnet==1.0.2",
              "recipe": "Authors' evaluation transforms and eight views; image-only; probability of melanoma.",
              "eligibility": "The authors used the historical ISIC competition training data."
                             " Their stated training source was reviewed. A full per-image manifest was unavailable."}


def prepare(work):
    weights = work / "pretrained_classifiers"
    weights.mkdir(parents=True, exist_ok=True)
    source = download_file_curl(work / "siim_isic.py", SOURCE, expected_sha256=SOURCE_SHA)
    paths = {name: download_file_curl(
        weights / name, f"https://zenodo.org/records/10049217/files/{name}?download=1",
        expected_bytes=size, expected_md5=md5) for name, (size, md5) in WEIGHTS.items()}
    return source, {**PROVENANCE, "checkpoint_sha256": {name: digest(path) for name, path in paths.items()}}


def score(source, images, manifest, device, provenance):
    import torch
    from PIL import Image
    from torchvision import transforms

    if digest(source) != SOURCE_SHA or any(digest(source.parent / "pretrained_classifiers" / name) != sha
                                          for name, sha in provenance["checkpoint_sha256"].items()):
        raise RuntimeError("The published SIIM–ISIC code or weights changed after configuration.")
    if not _installed("geffnet==1.0.2"):
        subprocess.run([sys.executable, "-m", "pip", "install", "geffnet==1.0.2"], check=True)
    spec = importlib.util.spec_from_file_location("prove_ai_siim_isic_release", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = Path.cwd()
    force_safe_load = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    try:
        # The released class resolves its three immutable weights relative to cwd.
        os.chdir(source.parent)
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        model = module.SIIMISICClassifier().to(device).eval()
    finally:
        os.chdir(previous)
        if force_safe_load is None:
            os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = force_safe_load
    model.enable_augment()
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                    transforms.ToTensor(), transforms.Normalize(mean=.5, std=.5)])
    batches = []
    ids = manifest.image_id.tolist()
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            batch = []
            for case in ids[start:start+8]:
                with Image.open(images / f"{case}.jpg") as photo:
                    batch.append(transform(photo.convert("RGB")))
            batches.append(model(torch.stack(batch).to(device))[:, model.positive_index].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame({"image_id": ids, NAME: np.concatenate(batches)})
