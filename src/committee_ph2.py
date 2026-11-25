"""Run Somfai's released dermoscopic Committee-29 in a separate interpreter."""
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from acquisition_stress import pixel_hash
from backbone_resolution_study import download_file_curl
from inference_store import digest

COMMIT = "e860e1c37349f50c5e4b24b23155476dfd1dd72f"
WARMUP = "7021d63a49106e22c79b40564a7d39930e7b0f53"
MEGA = "https://mega.nz/#!voxngK5T!uKOZkXBM9BXXAcuOQybGkfbGymli9LCBy9FxUXS61bY"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/boliu0/melanoma-winning-models?datasetVersionNumber=1"
PROVE = "https://isic-archive.s3.amazonaws.com/dois/10.34970-576276/prove-ai.zip"
PROVE_SHA = "6619e608704826a337fcbc3ae640d32176eb46340e3bf1adb525406f9c32fd51"


def checkout(url, dest, commit):
    if not (dest / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", commit], check=True)


def competition_weights(root):
    """Stage the authors' shared Kaggle release once for both comparison models."""
    from nb23_kaggle_recovery import stage_weights
    return stage_weights(root)


def prepare(root):
    repo = root / "skin"
    checkout("https://github.com/semiquark1/skin.git", repo, COMMIT)
    checkout("https://github.com/ildoonet/pytorch-gradual-warmup-lr.git",
             repo / "contrib/pytorch-gradual-warmup-lr", WARMUP)
    envdir = root / "py37"
    python = envdir / "bin/python"
    if not python.is_file():
        setup = download_file_curl(root / "miniconda.sh",
            "https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh",
            expected_sha256="fc96109ea96493e31f70abbc5cae58e80634480c0686ab46924549ac41176812")
        subprocess.run(["bash", str(setup), "-b", "-u", "-p", str(envdir)], check=True)
    if not (envdir / "ready").exists():
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
        subprocess.run(pip + ["torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
                              "--extra-index-url", "https://download.pytorch.org/whl/cu117"], check=True)
        packages = ["numpy==1.21.1", "pandas==1.3.1", "scipy==1.4.1", "scikit-image==0.18.2",
                    "scikit-learn==0.24.2", "albumentations==1.0.3", "timm==0.4.12",
                    "tensorflow==2.2.0", "protobuf==3.20.3", "h5py==2.10.0", "PyYAML==6.0.1",
                    "resnest==0.0.5", "pretrainedmodels==0.7.4", "geffnet==1.0.2",
                    "opencv-python-headless==4.5.3.56", "matplotlib==3.4.2", "tqdm"]
        subprocess.run(pip + packages, check=True)
        subprocess.run([str(python), "-c", "import torch,tensorflow,timm; assert torch.ones(1,device='cuda').item()==1"], check=True)
        (envdir / "ready").touch()
    models = root / "models"
    if not (root / "weights_ready").exists():
        models.mkdir(exist_ok=True)
        archive = root / "models-v1.tar.gz"
        if not archive.is_file() or archive.stat().st_size != 3963477537:
            if not shutil.which("megadl") and not shutil.which("megatools"):
                subprocess.run(["apt-get", "update", "-qq"], check=True)
                subprocess.run(["apt-get", "install", "-qy", "megatools"], check=True)
            downloader = ["megadl"] if shutil.which("megadl") else ["megatools", "dl"]
            subprocess.run(downloader + ["--path", str(root), MEGA], check=True)
        if archive.stat().st_size != 3963477537:
            raise ValueError("Incomplete Committee-29 weight archive")
        with tarfile.open(archive) as tf:
            tf.extractall(root, filter="data")
        competition_weights(root)
        links = list(models.rglob("create-links"))
        if len(links) != 1:
            raise RuntimeError("Official Committee-29 create-links script missing or ambiguous")
        modeldir = links[0].parent
        if modeldir != models:
            raise ValueError("Unexpected dermoscopic Committee release layout")
        subprocess.run(["bash", "create-links"], cwd=modeldir, check=True)
        broken = [str(p) for p in modeldir.rglob("*") if p.is_symlink() and not p.exists()]
        if broken:
            raise ValueError(f"Incomplete Committee weights: {broken[:3]}")
        (root / "weights_ready").write_text(str(modeldir))
    modeldir = Path((root / "weights_ready").read_text())
    return repo, python, modeldir


def read_scores(path, ids):
    df = pd.read_csv(path).set_index("name", verify_integrity=True)
    if set(df.index) != set(ids) or not np.isfinite(df.pred).all() or not df.pred.between(0, 1).all():
        raise ValueError("Committee-29 returned missing, extra or invalid scores")
    return df.pred.reindex(ids)


def score_committee(bench, cal, ph2):
    root = bench.root / "committee"
    root.mkdir(exist_ok=True)
    manifest = pd.concat([cal.assign(source="prove"), ph2.assign(source="ph2")], ignore_index=True)
    if len(cal) != 71 or len(ph2) != 200 or manifest.image_id.duplicated().any():
        raise ValueError("Expected 71 shared calibration nevi and all 200 PH2 images")
    results = []
    prepared = None
    for start in range(0, len(manifest), 48):
        group = manifest.iloc[start:start+48]
        key = f"committee29_{start:04d}"
        path = bench.root / (key + ".csv")
        if not bench.drive.restore(f"{bench.run_id}__{path.name}", path):
            if prepared is None:
                prepared = prepare(root)
                repo, python, modeldir = prepared
                provenance = {str(p.relative_to(modeldir)): digest(p) for p in sorted(modeldir.rglob("*"))
                              if p.is_file() and not p.is_symlink()}
                bench.refs["committee_weights"] = bench.save_json("committee_weights", {
                    "files": provenance, "code_commit": COMMIT, "warmup_commit": WARMUP,
                    "release": MEGA, "competition_weights": KAGGLE,
                    "runtime": subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True).splitlines()})
            image_dir = root / "images"
            image_dir.mkdir(exist_ok=True)
            out = root / key
            out.mkdir(exist_ok=True)
            prove_zip = download_file_curl(root / "prove.zip", PROVE, expected_sha256=PROVE_SHA)
            with ZipFile(prove_zip) as prove, ZipFile(bench.ph2_zip) as ph:
                members = {PurePosixPath(s).stem.upper(): s for s in ph.namelist() if
                           PurePosixPath(s).parent.name.lower().endswith("_dermoscopic_image") and s.lower().endswith(".bmp")}
                for row in group.itertuples():
                    raw = (prove.read(f"images/{row.image_id}.jpg") if row.source == "prove"
                           else ph.read(members[row.image_id]))
                    with Image.open(io.BytesIO(raw)) as original:
                        rgb = ImageOps.exif_transpose(original).convert("RGB")
                        if pixel_hash(rgb) != row.pixel_sha256:
                            raise ValueError(f"Committee input mismatch: {row.image_id}")
                        rgb.save(image_dir / f"{row.image_id}.png")
            dsd = pd.DataFrame({"name": group.image_id, "path": group.image_id + ".png",
                                "target": 0, "fold": 0}).sort_values("name")
            dsd.to_csv(out / "dsd.csv", index=False)
            for stage in "bcde":
                subprocess.run([str(python), str(repo / "src/skin_proc.py"), "predict-eval-dir",
                                "--data-root", str(image_dir), "--model-dir", str(modeldir),
                                "--stages", stage, "--batch-size", "4", str(out)], cwd=repo, check=True,
                               env={**os.environ, "MPLBACKEND": "Agg",
                                    "PYTHONPATH": str(repo / "contrib/pytorch-gradual-warmup-lr"),
                                    "TF_FORCE_GPU_ALLOW_GROWTH": "true"})
            shutil.copyfile(out / "softmax_cm.csv", path)
            read_scores(path, group.image_id)
        results.append(read_scores(path, group.image_id))
        bench.refs[key] = bench.publish(path)
    if "committee_weights" not in bench.refs:
        weights = bench.root / "committee_weights.json"
        if not bench.drive.restore(f"{bench.run_id}__{weights.name}", weights):
            raise RuntimeError("Missing provenance for restored Committee-29 scores")
        bench.refs["committee_weights"] = bench.publish(weights)
    return pd.concat(results, verify_integrity=True)
