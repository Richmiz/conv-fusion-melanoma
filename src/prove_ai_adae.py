"""Run the pinned ISIC Research ADAE release on the PROVe-AI JPEGs."""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from backbone_resolution_study import download_file_curl
from inference_store import digest


REPOSITORY = "https://github.com/ISIC-Research/ADAE"
COMMIT = "d11ba1fa28ebd89d971801a3d6846498b87e843c"
FILES = {"predict.py": "e719e21cb6bb778572e1cce9951c24f5f94f1caf0ff2f0ae92b4dc36a13351a3",
         "melanoma_util.py": "9a014e5b678e6a74feef1dd77a014c27f4a680e7f100b80e45609f5a3f2a6ccd"}
WEIGHTS = "https://www.kaggle.com/datasets/boliu0/melanoma-winning-models"
DEPENDENCIES = ("geffnet==1.0.2", "resnest==0.0.6b20200823", "pretrainedmodels==0.7.4",
                "albumentations==1.4.24", "imagesize==1.4.1")


def prepare(work, weights):
    work.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        url = f"https://raw.githubusercontent.com/ISIC-Research/ADAE/{COMMIT}/{name}"
        download_file_curl(work / name, url, expected_sha256=sha)
    tree = ast.parse((work / "predict.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_used_models" for target in node.targets))
    families = tuple(ast.literal_eval(assignment.value))
    expected = [f"{family}_best_fold{fold}.pth" for family in families for fold in range(5)]
    files = {p.name: p for p in Path(weights).glob("*.pth")}
    missing = [name for name in expected if name not in files or files[name].stat().st_size == 0]
    if len(expected) != 90 or missing:
        raise RuntimeError(f"The complete ADAE release must contain all 90 weights. Missing {len(missing)}. {WEIGHTS}")
    versions = {name: {"sha256": digest(files[name]), "bytes": files[name].stat().st_size} for name in expected}
    return families, {"implementation": REPOSITORY, "commit": COMMIT, "source_files": FILES,
                      "weights_source": WEIGHTS, "weights": versions, "dependencies": DEPENDENCIES,
                      "aggregation": "ISIC Research release. Mean of 90 melanoma probabilities, eight views each.",
                      "inputs": "Dermoscopy plus release-defined age, sex, site and image/patient grouping features."}


def validate_family(table, ids, family):
    cols = [f"{family}_best_fold{fold}" for fold in range(5)]
    if (set(table.columns) != {"image_name", "target", *cols} or table.image_name.duplicated().any()
            or set(table.image_name) != set(ids)):
        raise RuntimeError(f"ADAE output does not cover every frozen case and fold for {family}.")
    prob = table[cols].to_numpy(dtype=float)
    if (not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1))
            or not np.allclose(prob.mean(axis=1), table.target, rtol=0, atol=1e-10)):
        raise RuntimeError(f"ADAE family probabilities are invalid for {family}.")
    return table.set_index("image_name").reindex(ids)[cols].to_numpy(dtype=float)


def score(work, weights, images, manifest, families, fingerprint, restore, publish):
    need = [spec for spec in DEPENDENCIES if not _installed(spec)]
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", *need], check=True)
    frame = manifest.rename(columns={"image_id": "image_name"})[[
        "image_name", "patient_id", "sex", "age_approx", "anatom_site_general", "width", "height"]]
    # The upstream imagesize helper swaps the width and height columns. Stock
    # classification does not consume these columns, so pass explicit dimensions.
    input_csv = work / "prove_ai.csv"
    frame.to_csv(input_csv, index=False)
    ids = frame.image_name.tolist()
    groups, refs = [], {}
    env = {**os.environ, "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}
    for family in families:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", family):
            raise ValueError("Unexpected ADAE family name.")
        path = work / f"adae_{family}.csv"
        if not restore(path):
            for fold in range(5):
                name = f"{family}_best_fold{fold}.pth"
                if digest(Path(weights) / name) != fingerprint["weights"][name]["sha256"]:
                    raise RuntimeError(f"ADAE weight changed after preflight: {name}")
            output = work / "inference" / family
            result = output / input_csv.stem / "sub_output_melanoma_average.csv"
            if result.is_file():
                result.unlink()
            subprocess.run([sys.executable, str(work / "predict.py"), "-d", str(images),
                            "-c", str(input_csv), "-M", str(weights), "-S", str(output),
                            "-m", family, "-e", "5", "-b", "4", "-w", "0", "-D", "cuda"],
                           cwd=work, env=env, check=True)
            if not result.is_file():
                raise RuntimeError(f"ADAE produced no ensemble file for {family}.")
            path.write_bytes(result.read_bytes())
        table = pd.read_csv(path, float_precision="round_trip")
        groups.append(validate_family(table, ids, family))
        refs[path.name] = publish(path)
        print(f"Verified ADAE families saved: {len(groups)}/{len(families)}", flush=True)
    values = np.column_stack(groups)
    if values.shape != (len(ids), 90):
        raise RuntimeError("ADAE requires ninety component scores per image.")
    return pd.DataFrame({"image_id": ids, "adae": values.mean(axis=1)}), refs


def _installed(spec):
    from importlib import metadata

    name, version = spec.split("==")
    try:
        return metadata.version(name) == version
    except metadata.PackageNotFoundError:
        return False
