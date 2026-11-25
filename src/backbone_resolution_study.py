"""Step 0 (NASNetMobile feature geometry) + Step 1 (ConvNeXt-Small 224px vs 384px).

Controlled, resumable, hash-verified pipeline designed for a single Colab L4 GPU:
  - Step 0: Isolated TensorFlow subprocess using exact Notebook 09 colour correction
            (`acquisition_stress.correct_rgb`) and strict NB09 replay verification
            (max prob diff <= 1e-4, 0 changed cutoff decisions) before analyzing
            norm-vs-direction score-shift decomposition on the 3 saved NASNetMobile seeds.
  - Step 1: Matched PyTorch/timm ConvNeXt-Small (convnext_small.fb_in1k, pure LayerNorm)
            at 224px vs 384px with IDENTICAL training image order, per-image augmentation
            draws, macro-batch update budget, Shades of Gray (p=6), class weights, 4x TTA,
            per-epoch immutable Drive recovery checkpoints, and locked <=10% FA cutoff.
  - Fair stress evaluation: Each degraded validation image is created ONCE from the
            native decoded photo, then resized from that shared degraded image to 224px
            and 384px before applying exact NB09 colour correction.
  - External evaluation: Locked-cutoff benchmarks on HIBA (750), DERM12345 (10,351),
            ISIC 2020 Labelled Nevi (5,772), ISIC 2020 Unspecified Benign (1,581),
            and ISIC 2020 Within-Patient AUROC (133 patients).
"""
import gc
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image, ImageOps, __version__ as pillow_version
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score, roc_curve

from inference_store import (
    CheckpointDriveStore,
    FolderReader,
    RemoteZip,
    digest,
    fetch_zip_member_bytes,
)
from step0_nasnet_features import (
    STEP0_STYLES,
    STEP0_STYLE_NAMES,
    clamp_u8,
    correct_rgb,
    decompose_score_shift,
    degrade_native_image,
    locked_cutoff,
    posthoc_cosine_preview,
    resize_and_correct_shared,
    summarize_feature_geometry,
    summary_metrics,
)

TRAIN_FOLDER = os.getenv("CONV_FUSION_TRAIN_FOLDER", "")
TRAIN_RUN = "3c9cf6562d30585daf4b"
TRAIN_RECEIPT_SHA256 = "38d20d324460c26c762520efc30ed7bd41051b84ad131c3c8f1b85c86d8cdfc5"
ARM = "shades_of_gray_aug"
NASNET_SEEDS = (17, 29, 43)

NB08_FOLDER = os.getenv("CONV_FUSION_NB08_FOLDER", "")
NB08_RUN = "7ac63bb0dd5624b151ef"
NB08_RECEIPT_SHA256 = "72268156670b4dd01b20208720cc4b29b527f2cab85e77e0647a5a7aa1beda52"

NB09_FOLDER = os.getenv("CONV_FUSION_NB09_FOLDER", "")
NB09_RUN = "04b1977ffbc8d4f7e6ef"
NB09_RECEIPT_SHA256 = "4aaf7f7276d14b053b303dcc2680c0fa03887b768c3833e4f26f1402c95a86a9"

OUTPUT_FOLDER = "Melanoma Step 0-1 Backbone and Resolution Study"
BACKBONE_NAME = "convnext_small.fb_in1k"
STUDY_SEED = 17
RESOLUTIONS = (224, 384)
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
WARMUP_LR = 1e-5
PEAK_LR = 5e-5
MIN_LR = 2e-6
WEIGHT_DECAY = 1e-2
DROP_PATH_RATE = 0.2
PARTIAL_AUC_MAX_FPR = 0.15
FALSE_ALARM_LIMIT = 0.10
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 41

S3_DOIS = "https://isic-archive.s3.amazonaws.com/dois/"
ISIC2020_ZIP_URL = "https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_JPEG.zip"
ISIC2020_ZIP_BYTES = 24707698022
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALL_STYLES = (
    "original",
    "contrast_lower",
    "contrast_low",
    "gamma_bright",
    "warmer",
    "cooler",
    "lower_resolution",
    "combined_low_contrast_low_res",
)
STYLE_NAMES = {
    "original": "Original",
    "contrast_lower": "Lower contrast (0.70x)",
    "contrast_low": "Low contrast (0.55x)",
    "gamma_bright": "Brighter exposure (gamma 0.75)",
    "warmer": "Warmer colour cast",
    "cooler": "Cooler colour cast",
    "lower_resolution": "Lower resolution (112px)",
    "combined_low_contrast_low_res": "Combined (0.55x + 112px)",
}
SHORT_STYLE_LABELS = {
    "original": "Original",
    "contrast_lower": "Lower\ncontrast",
    "contrast_low": "Low\ncontrast",
    "gamma_bright": "Brighter\n(g=0.75)",
    "warmer": "Warmer",
    "cooler": "Cooler",
    "lower_resolution": "Lower res\n(112px)",
    "combined_low_contrast_low_res": "Combined\n(0.55x+112)",
}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def pixel_hash(image):
    h = hashlib.sha256(f"{image.width}x{image.height}:RGB\0".encode("utf8"))
    h.update(image.tobytes())
    return h.hexdigest()


def partial_auroc(labels, scores, max_fpr=PARTIAL_AUC_MAX_FPR):
    """Normalized partial AUROC over FPR in [0, max_fpr] matching NB06 selection."""
    return float(roc_auc_score(np.asarray(labels, dtype=int), np.asarray(scores, dtype=np.float64), max_fpr=max_fpr))


def epoch_lr(epoch, total_epochs=EPOCHS):
    if epoch == 1:
        return WARMUP_LR
    if total_epochs <= 2:
        return PEAK_LR
    progress = (epoch - 2) / (total_epochs - 2)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (PEAK_LR - MIN_LR) * cosine


def make_epoch_augmentation_plan(n_train, seed, epoch):
    """Deterministic per-epoch image order & per-sample NB06 augmentation draws.

    Independent of image_size and micro_batch so 224px and 384px models receive the
    exact same training image sequence and exact same geometric/photometric draws.
    """
    rng = np.random.default_rng([int(seed), int(epoch)])
    order = rng.permutation(int(n_train))
    rot_k = rng.integers(0, 4, size=int(n_train))
    do_flip = rng.integers(0, 2, size=int(n_train))
    raw_gains = rng.uniform(0.88, 1.12, size=(int(n_train), 3))
    gains = (raw_gains / raw_gains.mean(axis=1, keepdims=True)).astype(np.float32)
    contrast_scale = rng.uniform(0.92, 1.08, size=int(n_train)).astype(np.float32)
    brightness_shift = rng.uniform(-6.0, 6.0, size=int(n_train)).astype(np.float32)
    return {
        "order": order,
        "rot_k": rot_k,
        "do_flip": do_flip,
        "gains": gains,
        "contrast_scale": contrast_scale,
        "brightness_shift": brightness_shift,
    }


def apply_epoch_augmentation_slice(batch, plan, pos_start, pos_end):
    """Apply pre-sampled per-image NB06 augmentations to positions [pos_start, pos_end)."""
    out = np.empty_like(batch)
    for local_i, pos in enumerate(range(pos_start, pos_end)):
        view = np.rot90(batch[local_i], k=int(plan["rot_k"][pos]))
        if int(plan["do_flip"][pos]):
            view = np.fliplr(view)
        f = view.astype(np.float32) * plan["gains"][pos]
        mean = f.mean(axis=(0, 1), keepdims=True)
        f = (f - mean) * plan["contrast_scale"][pos] + mean + plan["brightness_shift"][pos]
        out[local_i] = clamp_u8(f)
    return out


def align_external_reference_scores(df_scores, manifest, seed_key):
    """Filter saved external reference scores to seed & original appearance and verify 1-to-1 match."""
    sub = df_scores[df_scores["seed"].astype(str) == str(seed_key)].copy()
    if "appearance" in sub.columns:
        sub = sub[sub["appearance"].astype(str) == "original"].copy()
    if sub.empty:
        raise RuntimeError(f"No original-appearance reference scores found for seed {seed_key}.")
    if sub["image_id"].duplicated().any():
        dups = int(sub["image_id"].duplicated().sum())
        raise RuntimeError(f"Duplicate image_ids ({dups}) in external reference scores for seed {seed_key}.")
    expected_ids = set(manifest["image_id"].astype(str))
    actual_ids = set(sub["image_id"].astype(str))
    if len(sub) != len(manifest) or actual_ids != expected_ids:
        raise RuntimeError(
            f"External reference score count/ID mismatch for seed {seed_key}: "
            f"got {len(sub)} rows for {len(manifest)} manifest photos."
        )
    sub = sub.set_index("image_id").loc[manifest["image_id"]].reset_index()
    return sub["raw_score"].to_numpy(dtype=np.float64)


def validate_split(frame):
    required = {"source", "image_id", "lesion_id", "label", "group", "pixel_sha256", "split", "member"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Saved split is missing required columns.")
    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    if set(frame["split"]) != {"train", "validation"} or set(frame["label"]) != {0, 1}:
        raise RuntimeError("Unexpected split or label values.")
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "validation"]
    for col in ("image_id", "group", "pixel_sha256"):
        if set(train[col]) & set(val[col]):
            raise RuntimeError(f"Train/validation leakage detected on {col}.")
    if set(zip(train["source"], train["lesion_id"])) & set(zip(val["source"], val["lesion_id"])):
        raise RuntimeError("Train/validation share a source lesion_id.")
    return frame


def within_patient_auroc(patient_ids, labels, scores):
    """Mean within-patient AUROC across patients who have both melanoma and benign nevi."""
    df = pd.DataFrame({
        "patient_id": pd.Series(patient_ids, dtype=str),
        "label": np.asarray(labels, dtype=int),
        "score": np.asarray(scores, dtype=np.float64),
    })
    per_patient = []
    for pid, grp in df.groupby("patient_id", sort=True):
        if pid and grp["label"].nunique() == 2:
            per_patient.append({
                "patient_id": pid,
                "melanomas": int((grp["label"] == 1).sum()),
                "benign": int((grp["label"] == 0).sum()),
                "auroc": float(roc_auc_score(grp["label"], grp["score"])),
            })
    if not per_patient:
        return {"patients": 0, "mean_auroc": float("nan"), "median_auroc": float("nan"), "per_patient": []}
    vals = [x["auroc"] for x in per_patient]
    return {
        "patients": len(per_patient),
        "mean_auroc": float(np.mean(vals)),
        "median_auroc": float(np.median(vals)),
        "per_patient": per_patient,
    }


def paired_delta_ci(labels, groups, scores_a, cut_a, scores_b, cut_b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Paired cluster-bootstrap 95% CI for (model_b - model_a) on sensitivity, FA, matched sens, AUROC."""
    y = np.asarray(labels, dtype=int)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    g = np.asarray(groups, dtype=str)
    unique_g = np.unique(g)
    idx_by_g = [np.where(g == u)[0] for u in unique_g]
    rng = np.random.default_rng(seed)
    keys = ("Melanomas caught (%)", "Benign false alarms (%)", "Matched <=10% FA sens (%)", "AUROC")
    deltas = {k: [] for k in keys}
    for _ in range(draws):
        picks = rng.integers(0, len(unique_g), size=len(unique_g))
        idx = np.concatenate([idx_by_g[p] for p in picks])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ma = summary_metrics(yb, sa[idx], cut_a)
        mb = summary_metrics(yb, sb[idx], cut_b)
        for k in keys:
            deltas[k].append(mb[k] - ma[k])
    return {
        "valid_draws": len(deltas["AUROC"]),
        "groups": int(len(unique_g)),
        "delta_b_minus_a_95ci": {
            k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in deltas.items() if v
        },
    }


def download_file_curl(dest, url, expected_bytes=None, expected_sha256=None, expected_md5=None):
    """Resumable verified archive download via curl."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _valid(path):
        if not path.is_file():
            return False
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            return False
        if expected_sha256 is not None and digest(path, "sha256") != expected_sha256:
            return False
        if expected_md5 is not None and digest(path, "md5") != expected_md5:
            return False
        return True

    if _valid(dest):
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            cmd = [
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1",
                "--continue-at", "-", "--output", str(partial), url,
            ]
            if expected_bytes is not None:
                cmd.extend(["--max-filesize", str(expected_bytes + 1024)])
            subprocess.run(cmd, check=True)
            if _valid(partial):
                partial.replace(dest)
                return dest
            raise RuntimeError(f"Downloaded {dest.name} failed size/checksum verification.")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    return dest


def _to_imagenet_tensor(torch, batch_u8):
    arr = batch_u8.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(arr))


def ensure_timm():
    """Import torch and timm (installing timm cleanly on Colab if absent)."""
    import torch
    try:
        import timm
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.0"], check=True)
        import timm
    return torch, timm


def build_convnext_model(torch, timm, seed=STUDY_SEED, pretrained=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = timm.create_model(
        BACKBONE_NAME,
        pretrained=pretrained,
        num_classes=1,
        drop_path_rate=DROP_PATH_RATE,
    )
    return model


def extract_convnext_features_and_scores_4tta(torch, model, images_u8, device, batch_size=BATCH_SIZE):
    """Exact float32 4x-TTA penultimate features and raw logits for ConvNeXt."""
    model.eval()
    n = len(images_u8)
    feats_list, scores_list = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = images_u8[start:start + batch_size]
            views = (
                batch,
                np.flip(batch, axis=2),
                np.flip(batch, axis=1),
                np.rot90(batch, k=2, axes=(1, 2)),
            )
            f_acc, s_acc = None, None
            for v in views:
                x = _to_imagenet_tensor(torch, v).to(device, non_blocking=True)
                spatial = model.forward_features(x)
                penult = model.forward_head(spatial, pre_logits=True)
                logits = model.head.fc(penult).reshape(-1)
                f_np = penult.detach().float().cpu().numpy().astype(np.float64)
                s_np = logits.detach().float().cpu().numpy().astype(np.float64)
                f_acc = f_np if f_acc is None else f_acc + f_np
                s_acc = s_np if s_acc is None else s_acc + s_np
            feats_list.append(f_acc / 4.0)
            scores_list.append(s_acc / 4.0)
    return np.concatenate(feats_list, axis=0), np.concatenate(scores_list, axis=0)


def score_convnext_4tta(torch, model, images_u8, device, batch_size=BATCH_SIZE):
    _, scores = extract_convnext_features_and_scores_4tta(torch, model, images_u8, device, batch_size=batch_size)
    return scores


class BackboneResolutionStudy:
    """Orchestrates Step 0 (NASNetMobile feature geometry) + Step 1 (ConvNeXt 224 vs 384)."""

    def __init__(self, implementation_files, work="/content/melanoma-step0-step1"):
        from google.colab import auth
        from googleapiclient.discovery import build

        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        auth.authenticate_user()
        service = build("drive", "v3", cache_discovery=False)

        self.train_reader = FolderReader(service, TRAIN_FOLDER, self.work / "source_nb06")
        self.nb08_reader = FolderReader(service, NB08_FOLDER, self.work / "source_nb08")
        self.nb09_reader = FolderReader(service, NB09_FOLDER, self.work / "source_nb09")

        # 1. Verify Notebook 06 receipt, split, validation predictions, external manifests & scores
        nb06_receipt_path = self.train_reader.fetch(f"{TRAIN_RUN}__save_receipt.json")
        if digest(nb06_receipt_path) != TRAIN_RECEIPT_SHA256:
            raise RuntimeError("Notebook 06 save receipt changed.")
        nb06_receipt = json.loads(nb06_receipt_path.read_text(encoding="utf8"))
        val_report_path = self.train_reader.fetch_artifact(nb06_receipt["validation_report"])
        val_report = json.loads(val_report_path.read_text(encoding="utf8"))

        def _nb06_file(suffix):
            return next(x for x in nb06_receipt["files"] if x["name"].endswith(suffix))

        split_path = self.train_reader.fetch_artifact(_nb06_file("__pooled_ham_bcn_split.csv"))
        val_pred_path = self.train_reader.fetch_artifact(val_report["predictions"])
        hiba_manifest_path = self.train_reader.fetch_artifact(_nb06_file("__HIBA_images.csv"))
        derm_manifest_path = self.train_reader.fetch_artifact(_nb06_file("__DERM12345_images.csv"))
        hiba_scores_path = self.train_reader.fetch_artifact(_nb06_file("__HIBA_shades_of_gray_aug_scores.csv"))
        derm_scores_path = self.train_reader.fetch_artifact(_nb06_file("__DERM12345_shades_of_gray_aug_scores.csv"))

        # 2. Verify Notebook 08 receipt & ISIC 2020 cohort manifests + reference scores
        nb08_receipt_path = self.nb08_reader.fetch(f"{NB08_RUN}__durable_save_receipt.json")
        if digest(nb08_receipt_path) != NB08_RECEIPT_SHA256:
            raise RuntimeError("Notebook 08 receipt changed.")
        nb08_receipt = json.loads(nb08_receipt_path.read_text(encoding="utf8"))

        def _find_ref(obj, suffix):
            if isinstance(obj, dict):
                if isinstance(obj.get("name"), str) and obj["name"].endswith(suffix) and "sha256" in obj:
                    return obj
                for v in obj.values():
                    hit = _find_ref(v, suffix)
                    if hit is not None:
                        return hit
            elif isinstance(obj, list):
                for v in obj:
                    hit = _find_ref(v, suffix)
                    if hit is not None:
                        return hit
            return None

        isic_nevi_ref = _find_ref(nb08_receipt, "__spectrum_shades_of_gray_aug_nevi_cohort.csv")
        isic_unspec_ref = _find_ref(nb08_receipt, "__spectrum_shades_of_gray_aug_unspecified_cohort.csv")
        if isic_nevi_ref is None or isic_unspec_ref is None:
            raise RuntimeError("Could not locate verified ISIC 2020 cohort manifests in NB08 receipt.")
        isic_nevi_path = self.nb08_reader.fetch_artifact(isic_nevi_ref)
        isic_unspec_path = self.nb08_reader.fetch_artifact(isic_unspec_ref)

        # 3. Verify Notebook 09 receipt & stress policy + NB09 reference scores
        nb09_receipt_path = self.nb09_reader.fetch(f"{NB09_RUN}__durable_save_receipt.json")
        if digest(nb09_receipt_path) != NB09_RECEIPT_SHA256:
            raise RuntimeError("Notebook 09 receipt changed.")
        nb09_receipt = json.loads(nb09_receipt_path.read_text(encoding="utf8"))
        policy_ref = _find_ref(nb09_receipt, "__data_policy.json")
        nb09_summary_ref = _find_ref(nb09_receipt, "__report_summary.csv")
        nb09_by_seed_ref = _find_ref(nb09_receipt, "__report_by_seed.csv")
        if policy_ref is None or nb09_summary_ref is None:
            raise RuntimeError("Could not locate NB09 policy or summary in NB09 receipt.")
        self.policy = json.loads(self.nb09_reader.fetch_artifact(policy_ref).read_text(encoding="utf8"))
        self.nb09_summary = pd.read_csv(self.nb09_reader.fetch_artifact(nb09_summary_ref))
        self.nb09_by_seed = pd.read_csv(self.nb09_reader.fetch_artifact(nb09_by_seed_ref))
        self.nb09_score_refs = {}
        for style in ALL_STYLES:
            if style == "combined_low_contrast_low_res":
                continue
            for key in (str(STUDY_SEED), "ensemble"):
                tag = f"seed{key}" if key != "ensemble" else "ensemble"
                ref = _find_ref(nb09_receipt, f"__scores_{style}_{tag}.csv")
                if ref is not None:
                    self.nb09_score_refs[(style, key)] = ref

        # Load core tables
        self.split = validate_split(pd.read_csv(split_path, dtype=str, keep_default_na=False))
        self.train_manifest = self.split[self.split["split"] == "train"].sort_values(["source", "image_id"]).reset_index(drop=True)
        self.val_manifest = self.split[self.split["split"] == "validation"].sort_values(["source", "image_id"]).reset_index(drop=True)
        self.sources = val_report["protocol"]["image_sources"]
        self.external_specs = val_report["protocol"]["external_sets"]
        self.nasnet_models = {str(s): val_report["model_states"][ARM][str(s)]["selected"] for s in NASNET_SEEDS}
        self.class_weights = np.asarray(val_report["protocol"]["training_weights"], dtype=np.float32)

        saved_val = pd.read_csv(val_pred_path, dtype={"seed": str, "image_id": str}, keep_default_na=False, converters={"raw_score": float})
        saved_val = saved_val[(saved_val["arm"] == ARM) & (saved_val["appearance"] == "original")]
        self.nasnet_val_scores = {}
        for key in (*(str(s) for s in NASNET_SEEDS), "ensemble"):
            sub = saved_val[saved_val["seed"] == key].set_index("image_id").loc[self.val_manifest["image_id"]].reset_index()
            self.nasnet_val_scores[key] = sub["raw_score"].to_numpy(dtype=np.float64)
        compare = [x for x in val_report["comparison"] if x["arm"] == ARM and x["appearance"] == "original"]
        self.nasnet_cutoffs = {str(x["seed"]): float(x["cutoff"]) for x in compare}

        self.hiba_manifest = pd.read_csv(hiba_manifest_path, dtype=str, keep_default_na=False)
        self.hiba_manifest["label"] = self.hiba_manifest["label"].astype(int)
        self.derm_manifest = pd.read_csv(derm_manifest_path, dtype=str, keep_default_na=False)
        self.derm_manifest["label"] = self.derm_manifest["label"].astype(int)
        self.hiba_nasnet_scores = pd.read_csv(hiba_scores_path, dtype={"seed": str, "image_id": str, "appearance": str}, keep_default_na=False, converters={"raw_score": float})
        self.derm_nasnet_scores = pd.read_csv(derm_scores_path, dtype={"seed": str, "image_id": str, "appearance": str}, keep_default_na=False, converters={"raw_score": float})
        # Validate external reference alignment upfront so any schema mismatch fails immediately, not after training
        for ds_lbl, df_s, mf in (("HIBA", self.hiba_nasnet_scores, self.hiba_manifest),
                                 ("DERM12345", self.derm_nasnet_scores, self.derm_manifest)):
            for sk in (str(STUDY_SEED), "ensemble"):
                align_external_reference_scores(df_s, mf, sk)

        self.isic_nevi_manifest = pd.read_csv(isic_nevi_path, dtype=str, keep_default_na=False)
        self.isic_nevi_manifest["label"] = self.isic_nevi_manifest["label"].astype(int)
        self.isic_nevi_manifest["raw_score"] = self.isic_nevi_manifest["raw_score"].astype(float)
        self.isic_unspec_manifest = pd.read_csv(isic_unspec_path, dtype=str, keep_default_na=False)
        self.isic_unspec_manifest["label"] = self.isic_unspec_manifest["label"].astype(int)
        self.isic_unspec_manifest["raw_score"] = self.isic_unspec_manifest["raw_score"].astype(float)

        self.implementation_files = dict(implementation_files)
        self.protocol = {
            "study": "Step 0 (NASNetMobile feature norm vs direction) + Step 1 (ConvNeXt-Small 224 vs 384)",
            "source_receipts": {
                "nb06": TRAIN_RECEIPT_SHA256,
                "nb08": NB08_RECEIPT_SHA256,
                "nb09": NB09_RECEIPT_SHA256,
            },
            "implementation": {k: hashlib.sha256(v.encode("utf8")).hexdigest() for k, v in sorted(self.implementation_files.items())},
            "backbone": BACKBONE_NAME,
            "resolutions": list(RESOLUTIONS),
            "seed": STUDY_SEED,
            "cohorts": {"train": len(self.train_manifest), "validation": len(self.val_manifest)},
            "external_counts": {
                "HIBA": len(self.hiba_manifest),
                "DERM12345": len(self.derm_manifest),
                "ISIC2020_nevi": len(self.isic_nevi_manifest),
                "ISIC2020_unspecified": len(self.isic_unspec_manifest),
            },
            "training": {
                "optimizer": "AdamW",
                "warmup_lr": WARMUP_LR,
                "peak_lr": PEAK_LR,
                "min_lr": MIN_LR,
                "weight_decay": WEIGHT_DECAY,
                "drop_path_rate": DROP_PATH_RATE,
                "effective_batch_size": BATCH_SIZE,
                "shared_epoch_order_and_aug": True,
                "per_epoch_drive_recovery": True,
                "max_epochs": EPOCHS,
                "patience": PATIENCE,
                "checkpoint_selection": f"max validation partial AUROC (FPR <= {PARTIAL_AUC_MAX_FPR}), tie-break full AUROC",
                "class_weights": self.class_weights.tolist(),
                "preprocessing": "Exact NB09 mask-aware Shades of Gray (p=6, float64) + NB06 mild augmentation",
                "tta_views": 4,
            },
            "stress_policy": self.policy,
            "stress_generation": "Step 0 uses exact NB09 224px stress path for replay; Step 1 creates each degraded image once from the native decoded photo and resizes that shared degraded image to 224px and 384px.",
            "styles": list(ALL_STYLES),
            "false_alarm_limit": FALSE_ALARM_LIMIT,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pillow": pillow_version,
                "sklearn": sklearn_version,
                "matplotlib": matplotlib.__version__,
            },
        }
        self.run_id = canonical_hash(self.protocol)[:20]
        self.directory = self.work / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.work / "image_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = {}
        self.drive = CheckpointDriveStore(service, OUTPUT_FOLDER)
        self.drive.preflight(required_bytes=1024 * 1024**2)

        self.save_json("protocol", {"run_id": self.run_id, "protocol": self.protocol})
        code_path = self.directory / "implementation.zip"
        with ZipFile(code_path, "w") as archive:
            for name, contents in sorted(self.implementation_files.items()):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                archive.writestr(info, contents)
        self.publish("implementation", code_path)
        print(f"Run {self.run_id} | Private Drive folder: {self.drive.link}", flush=True)
        print(f"Verified splits: train={len(self.train_manifest):,}, val={len(self.val_manifest):,} "
              f"(mel={int(self.val_manifest.label.sum())}, benign={int((self.val_manifest.label==0).sum())}).", flush=True)
        print(f"Verified external cohorts & reference scores: HIBA={len(self.hiba_manifest):,}, DERM12345={len(self.derm_manifest):,}, "
              f"ISIC2020-Nevi={len(self.isic_nevi_manifest):,}, ISIC2020-Unspec={len(self.isic_unspec_manifest):,}.", flush=True)

    # ---- Durable Drive helpers ----

    def name(self, filename):
        return f"{self.run_id}__{filename}"

    def publish(self, logical, path):
        ref = self.drive.publish(Path(path), self.name(Path(path).name))
        self.drive.verify_receipt(ref)
        self.artifacts[logical] = ref
        return ref

    def save_json(self, logical, value):
        path = self.directory / f"{logical}.json"
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
        self.publish(logical, path)
        return path

    def save_df(self, logical, frame):
        path = self.directory / f"{logical}.csv"
        partial = path.with_suffix(".csv.part")
        frame.to_csv(partial, index=False, float_format="%.17g", lineterminator="\n")
        partial.replace(path)
        self.publish(logical, path)
        return path

    def restore_json(self, logical):
        path = self.directory / f"{logical}.json"
        if not self.drive.restore(self.name(path.name), path):
            return None
        self.publish(logical, path)
        return json.loads(path.read_text(encoding="utf8"))

    def restore_df(self, logical):
        path = self.directory / f"{logical}.csv"
        if not self.drive.restore(self.name(path.name), path):
            return None
        self.publish(logical, path)
        return pd.read_csv(path, dtype={"image_id": str, "seed": str}, keep_default_na=False)

    def save_figure(self, logical, figure):
        path = self.directory / f"{logical}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        self.publish(logical, path)
        plt.show()
        plt.close(figure)

    def stage_done(self, stage):
        receipt = self.restore_json(f"{stage}_receipt")
        if receipt is None:
            return False
        if receipt["run_id"] != self.run_id or receipt["stage"] != stage or not receipt["files"]:
            raise RuntimeError(f"Stage receipt {stage} belongs to another run or is empty.")
        for key, ref in receipt["files"].items():
            self.drive.verify_receipt(ref)
            self.artifacts[key] = ref
        return True

    def finish_stage(self, stage, required_keys):
        if not required_keys or any(k not in self.artifacts for k in required_keys):
            missing = [k for k in required_keys if k not in self.artifacts]
            raise RuntimeError(f"Stage {stage} missing saved outputs: {missing}")
        self.save_json(f"{stage}_receipt", {
            "run_id": self.run_id,
            "stage": stage,
            "files": {k: self.artifacts[k] for k in required_keys},
        })

    # ---- Source HAM+BCN image cache (224, 384, and verified native validation JPEGs) ----

    def _source_cache_paths(self):
        return {
            "train_sog_224": self.cache_dir / "train_sog_224.npy",
            "train_sog_384": self.cache_dir / "train_sog_384.npy",
            "val_raw_224": self.cache_dir / "val_raw_224.npy",
            "val_sog_224": self.cache_dir / "val_sog_224.npy",
            "val_sog_384": self.cache_dir / "val_sog_384.npy",
            "val_native_dir": self.cache_dir / "val_native_jpegs",
            "done": self.cache_dir / "source_cache_verified.json",
        }

    def prepare_source_images(self):
        paths = self._source_cache_paths()
        if paths["done"].is_file() and all(paths[k].is_file() for k in ("train_sog_224", "train_sog_384", "val_raw_224", "val_sog_224", "val_sog_384")):
            meta = json.loads(paths["done"].read_text(encoding="utf8"))
            if (meta.get("train_n") == len(self.train_manifest) and meta.get("val_n") == len(self.val_manifest)
                    and paths["val_native_dir"].is_dir()
                    and sum(1 for _ in paths["val_native_dir"].glob("*.jpg")) == len(self.val_manifest)):
                print("Reusing verified local HAM+BCN 224px, 384px, and native validation caches.", flush=True)
                return

        archives_dir = self.work / "archives"
        ham_zip = download_file_curl(
            archives_dir / "ISIC2018_Task3_Training_Input.zip",
            self.sources["HAM"]["url"],
            expected_bytes=int(self.sources["HAM"]["bytes"]),
        )
        bcn_zip = download_file_curl(
            archives_dir / "BCN20000.zip",
            self.sources["BCN"]["url"],
            expected_bytes=int(self.sources["BCN"]["bytes"]),
            expected_md5=self.sources["BCN"].get("md5"),
        )
        print("Verified HAM10000 and BCN20000 archives. Building 224px, 384px, and native validation caches...", flush=True)
        paths["val_native_dir"].mkdir(parents=True, exist_ok=True)

        with ZipFile(ham_zip) as z_ham, ZipFile(bcn_zip) as z_bcn:
            zmap = {"HAM": z_ham, "BCN": z_bcn}

            def _build_split(manifest, split_name):
                n = len(manifest)
                sog224 = np.lib.format.open_memmap(paths[f"{split_name}_sog_224"].with_suffix(".part.npy"), mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
                sog384 = np.lib.format.open_memmap(paths[f"{split_name}_sog_384"].with_suffix(".part.npy"), mode="w+", dtype=np.uint8, shape=(n, 384, 384, 3))
                raw224 = None
                if split_name == "val":
                    raw224 = np.lib.format.open_memmap(paths["val_raw_224"].with_suffix(".part.npy"), mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
                for idx, row in enumerate(manifest.itertuples(index=False)):
                    raw_bytes = zmap[row.source].read(row.member)
                    with Image.open(io.BytesIO(raw_bytes)) as photo:
                        orig = ImageOps.exif_transpose(photo).convert("RGB")
                    if pixel_hash(orig) != row.pixel_sha256:
                        raise RuntimeError(f"Decoded pixel hash mismatch for {row.image_id}.")
                    r224 = np.asarray(orig.resize((224, 224), Image.Resampling.BILINEAR))
                    r384 = np.asarray(orig.resize((384, 384), Image.Resampling.BILINEAR))
                    sog224[idx] = correct_rgb(r224)
                    sog384[idx] = correct_rgb(r384)
                    if split_name == "val":
                        raw224[idx] = r224
                        jp = paths["val_native_dir"] / f"{row.image_id}.jpg"
                        jpart = jp.with_suffix(".jpg.part")
                        jpart.write_bytes(raw_bytes)
                        jpart.replace(jp)
                    if (idx + 1) % 1500 == 0 or (idx + 1) == n:
                        print(f"  [{split_name}] decoded & verified {idx + 1:,}/{n:,} photos.", flush=True)
                sog224.flush(); del sog224
                sog384.flush(); del sog384
                paths[f"{split_name}_sog_224"].with_suffix(".part.npy").replace(paths[f"{split_name}_sog_224"])
                paths[f"{split_name}_sog_384"].with_suffix(".part.npy").replace(paths[f"{split_name}_sog_384"])
                if split_name == "val":
                    raw224.flush(); del raw224
                    paths["val_raw_224"].with_suffix(".part.npy").replace(paths["val_raw_224"])

            _build_split(self.train_manifest, "train")
            _build_split(self.val_manifest, "val")

        paths["done"].write_text(json.dumps({
            "train_n": len(self.train_manifest),
            "val_n": len(self.val_manifest),
        }, indent=2) + "\n", encoding="utf8")
        print("HAM+BCN 224px, 384px, and native validation caches ready.", flush=True)

    def _shared_stress_cache_path(self, style, image_size):
        return self.cache_dir / f"val_shared_stress_{style}_{int(image_size)}.npy"

    def ensure_shared_stress_caches(self):
        """Create every degraded validation image ONCE from the native photo, then resize to 224 and 384."""
        self.prepare_source_images()
        paths = self._source_cache_paths()
        all_exist = all(
            self._shared_stress_cache_path(style, sz).is_file()
            for style in ALL_STYLES for sz in RESOLUTIONS
        )
        if all_exist:
            print("Reusing shared native-degraded validation stress caches (224px and 384px).", flush=True)
            return

        n = len(self.val_manifest)
        print(f"Building shared native-degraded validation stress caches for {len(ALL_STYLES)} appearances...", flush=True)
        for style in ALL_STYLES:
            p224 = self._shared_stress_cache_path(style, 224)
            p384 = self._shared_stress_cache_path(style, 384)
            if p224.is_file() and p384.is_file():
                continue
            mm224 = np.lib.format.open_memmap(p224.with_suffix(".part.npy"), mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
            mm384 = np.lib.format.open_memmap(p384.with_suffix(".part.npy"), mode="w+", dtype=np.uint8, shape=(n, 384, 384, 3))
            for idx, row in enumerate(self.val_manifest.itertuples(index=False)):
                with Image.open(paths["val_native_dir"] / f"{row.image_id}.jpg") as photo:
                    orig = ImageOps.exif_transpose(photo).convert("RGB")
                if pixel_hash(orig) != row.pixel_sha256:
                    raise RuntimeError(f"Native validation cache changed for {row.image_id}.")
                degraded = degrade_native_image(np.asarray(orig, dtype=np.uint8), style, self.policy)
                mm224[idx] = resize_and_correct_shared(degraded, 224)
                mm384[idx] = resize_and_correct_shared(degraded, 384)
            mm224.flush(); del mm224
            mm384.flush(); del mm384
            p224.with_suffix(".part.npy").replace(p224)
            p384.with_suffix(".part.npy").replace(p384)
            print(f"  [Shared stress] {STYLE_NAMES[style]} ready for both 224px and 384px.", flush=True)

    # ---- Step 0: Isolated TensorFlow NASNetMobile feature geometry check ----

    def run_step0_nasnet_features(self):
        if self.stage_done("step0"):
            summary = self.restore_df("step0_nasnet_feature_summary")
            head_params = self.restore_json("step0_nasnet_head_params")
            self.restore_json("step0_nasnet_scores")
            print("Restored verified Step 0 (NASNetMobile feature geometry) from Drive.", flush=True)
            display(self._step0_display_table(summary))
            return summary, head_params

        self.prepare_source_images()
        paths = self._source_cache_paths()
        model_paths = {s: str(self.train_reader.fetch_artifact(ref)) for s, ref in self.nasnet_models.items()}

        cfg_path = self.directory / "step0_config.json"
        out_head = self.directory / "step0_nasnet_head_params.json"
        out_csv = self.directory / "step0_nasnet_feature_summary.csv"
        out_scores = self.directory / "step0_nasnet_scores.json"
        cfg = {
            "val_raw_224_path": str(paths["val_raw_224"]),
            "labels": self.val_manifest["label"].astype(int).tolist(),
            "policy": self.policy,
            "styles": list(STEP0_STYLES),
            "cutoffs": {k: float(v) for k, v in self.nasnet_cutoffs.items()},
            "reference_scores": {k: v.tolist() for k, v in self.nasnet_val_scores.items()},
            "model_paths": model_paths,
            "out_head_params": str(out_head),
            "out_summary_csv": str(out_csv),
            "out_ensemble_scores": str(out_scores),
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf8")

        print("Running Step 0 in isolated TensorFlow subprocess (exact NB09 colour correction & replay check)...", flush=True)
        script_path = Path(__file__).resolve().parent / "step0_nasnet_features.py"
        self._invoke_step0_subprocess(script_path, cfg_path)

        head_params = json.loads(out_head.read_text(encoding="utf8"))
        summary = pd.read_csv(out_csv)
        self.publish("step0_nasnet_head_params", out_head)
        self.publish("step0_nasnet_feature_summary", out_csv)
        self.publish("step0_nasnet_scores", out_scores)

        fig = self._plot_step0_summary(summary)
        self.save_figure("step0_nasnet_feature_plot", fig)
        self.finish_stage("step0", [
            "step0_nasnet_head_params",
            "step0_nasnet_feature_summary",
            "step0_nasnet_scores",
            "step0_nasnet_feature_plot",
        ])

        print("\n=== Step 0 Result: Saved NASNetMobile Head Parameters (3 Seeds) ===")
        for s in sorted((k for k in head_params if k != "ensemble_replay"), key=int):
            hp = head_params[s]
            print(f"  Seed {s}: bias b = {hp['bias']:+.4f} | ||w|| = {hp['weight_l2_norm']:.4f} | "
                  f"median w^T h (mel / benign) = {hp['median_dot_orig_mel']:+.3f} / {hp['median_dot_orig_benign']:+.3f} | "
                  f"replay prob_diff = {hp['max_prob_diff']:.2e}, changed_decisions = {hp['changed_cutoff_decisions']}")
        if "ensemble_replay" in head_params:
            er = head_params["ensemble_replay"]
            print(f"  Ensemble replay: prob_diff = {er['max_prob_diff']:.2e}, changed_decisions = {er['changed_cutoff_decisions']}")
        display(self._step0_display_table(summary))
        return summary, head_params

    def _invoke_step0_subprocess(self, script_path, cfg_path):
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(cfg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            tail = "".join(lines[-30:]).strip()
            raise RuntimeError(f"Step 0 subprocess failed with exit code {proc.returncode}.\n{tail}")

    def _step0_display_table(self, summary):
        cols = [
            "Appearance", "Median ||h|| ratio (mel)", "Median cos(h0, h1) (mel)",
            "Median dz total (mel)", "Median dz norm-term (mel)", "Median dz dir-term (mel)",
            "Median norm share of |dz| (mel)", "Linear fixed sens (%)", "Posthoc cosine fixed sens (%)",
        ]
        agg = summary.groupby(["Style", "Appearance"], sort=False)[cols[1:]].mean().reset_index()
        order = {s: i for i, s in enumerate(STEP0_STYLES)}
        agg["_o"] = agg["Style"].map(order)
        return agg.sort_values("_o").drop(columns=["_o", "Style"]).round(3)

    def _plot_step0_summary(self, summary):
        agg = summary.groupby(["Style", "Appearance"], sort=False).mean(numeric_only=True).reset_index()
        agg = agg[agg["Style"] != "original"].copy()
        order = {s: i for i, s in enumerate(STEP0_STYLES)}
        agg["_o"] = agg["Style"].map(order)
        agg = agg.sort_values("_o")
        labels = [SHORT_STYLE_LABELS.get(s, s) for s in agg["Style"]]
        x = np.arange(len(labels))
        w = 0.34

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
        ax1.bar(x - w / 2, agg["Median dz norm-term (mel)"], width=w, label="Norm-scaling term", color="#2b6cb0")
        ax1.bar(x + w / 2, agg["Median dz dir-term (mel)"], width=w, label="Direction-rotation term", color="#dd6b20")
        ax1.axhline(0.0, color="black", linewidth=0.8)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
        ax1.set_ylabel("Median melanoma logit shift (dz)")
        ax1.set_title("Step 0: NASNetMobile exact dz decomposition (3-seed mean)")
        ax1.legend()

        ax2.plot(x, agg["Linear fixed sens (%)"], "o-", label="Original linear head (fixed cutoff)", linewidth=2)
        ax2.plot(x, agg["Posthoc cosine fixed sens (%)"], "s--", label="Post-hoc L2 cosine preview (fixed cutoff)", linewidth=2)
        ax2.plot(x, agg["Linear matched <=10% FA (%)"], "^:", label="Linear head (matched <=10% FA)", linewidth=1.8)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Melanomas caught (%)")
        ax2.set_title("Does removing feature-norm scaling stabilize fixed cutoff?")
        ax2.legend(fontsize=8)
        return fig

    # ---- Step 1: Controlled ConvNeXt-Small (224px vs 384px) training with per-epoch Drive recovery ----

    def _find_latest_drive_epoch_state(self, image_size):
        """Scan immutable per-epoch state manifests on private Drive from highest completed epoch down."""
        for ep in range(EPOCHS, 0, -1):
            state = self.restore_json(f"convnext_{image_size}_epoch{ep:02d}_state")
            if state is not None:
                if state.get("run_id") != self.run_id or int(state.get("image_size", -1)) != int(image_size):
                    raise RuntimeError(f"Epoch {ep} recovery state belongs to another run or image size.")
                return state
        return None

    def _verify_drive_recovery_roundtrip(self, torch, image_size):
        """Verify before epoch 1 that a saved checkpoint restores cleanly into a fresh local path."""
        probe_path = self.directory / f"convnext_{image_size}_recovery_probe.pt"
        payload = {"run_id": self.run_id, "image_size": int(image_size), "probe": True}
        torch.save(payload, probe_path)
        ref = self.publish(f"convnext_{image_size}_recovery_probe", probe_path)
        fresh_path = self.directory / f"convnext_{image_size}_recovery_probe_restored.pt"
        if fresh_path.exists():
            fresh_path.unlink()
        if not self.drive.restore(ref["name"], fresh_path):
            raise RuntimeError("Drive checkpoint restore probe failed before training started.")
        if digest(fresh_path) != ref["sha256"]:
            raise RuntimeError("Restored Drive checkpoint probe hash mismatch.")
        restored = torch.load(fresh_path, map_location="cpu", weights_only=False)
        if restored != payload:
            raise RuntimeError("Restored Drive checkpoint probe payload mismatch.")
        fresh_path.unlink()
        print(f"Verified fresh-session Drive checkpoint restore probe for ConvNeXt @{image_size}px.", flush=True)

    def train_convnext(self, image_size):
        image_size = int(image_size)
        stage = f"train_convnext_{image_size}"
        if self.stage_done(stage):
            meta = self.restore_json(f"convnext_{image_size}_selection")
            self.restore_df(f"convnext_{image_size}_history")
            self.restore_df(f"convnext_{image_size}_val_scores")
            self._restore_model_weights(image_size)
            print(f"Restored verified ConvNeXt-Small @{image_size}px (best epoch {meta['best_epoch']}, "
                  f"val AUROC {meta['val_auroc']:.4f}, locked cutoff {meta['locked_cutoff']:.4f}).", flush=True)
            return meta

        self.prepare_source_images()
        paths = self._source_cache_paths()
        torch, timm = ensure_timm()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and not os.environ.get("ALLOW_CPU_TEST"):
            raise RuntimeError("Connect a Colab GPU before training ConvNeXt-Small.")

        train_sog = np.load(paths[f"train_sog_{image_size}"], mmap_mode="r")
        val_sog = np.load(paths[f"val_sog_{image_size}"], mmap_mode="r")
        y_train = self.train_manifest["label"].to_numpy(dtype=np.int64)
        y_val = self.val_manifest["label"].to_numpy(dtype=np.int64)

        micro_batch = 32 if image_size <= 224 else 16
        use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

        model = build_convnext_model(torch, timm, seed=STUDY_SEED, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        history = []
        start_epoch = 1
        best_score = (-1.0, -1.0)
        best_epoch = 0
        no_improve = 0
        early_stopped = False
        best_weights_path = None
        best_weights_ref = None

        # Check private Drive for the latest completed immutable epoch recovery state (works in a fresh session!)
        recovered_state = self._find_latest_drive_epoch_state(image_size)
        if recovered_state is None:
            self._verify_drive_recovery_roundtrip(torch, image_size)
        if recovered_state is not None:
            rec_ep = int(recovered_state["epoch"])
            rec_ref = recovered_state["recovery_ref"]
            best_weights_ref = recovered_state["best_weights_ref"]
            self.drive.verify_receipt(rec_ref)
            self.drive.verify_receipt(best_weights_ref)

            rec_local = self.directory / f"convnext_{image_size}_epoch{rec_ep:02d}_recovery.pt"
            if not self.drive.restore(rec_ref["name"], rec_local):
                raise RuntimeError(f"Failed to restore Drive recovery checkpoint {rec_ref['name']}.")
            best_weights_path = self.directory / f"convnext_{image_size}_best_epoch{int(recovered_state['best_epoch']):02d}.pt"
            if not self.drive.restore(best_weights_ref["name"], best_weights_path):
                raise RuntimeError(f"Failed to restore Drive best checkpoint {best_weights_ref['name']}.")

            ckpt = torch.load(rec_local, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            history = list(recovered_state["history"])
            start_epoch = rec_ep + 1
            best_score = tuple(recovered_state["best_score"])
            best_epoch = int(recovered_state["best_epoch"])
            no_improve = int(recovered_state["no_improve"])
            early_stopped = bool(recovered_state.get("early_stopped", False))
            print(f"Restored Drive recovery checkpoint for ConvNeXt-Small @{image_size}px at epoch {rec_ep} "
                  f"(best epoch {best_epoch}); {'finalizing' if early_stopped else f'continuing at epoch {start_epoch}'}.", flush=True)

        t_start = time.monotonic()
        n_train = len(y_train)
        if not early_stopped:
            for epoch in range(start_epoch, EPOCHS + 1):
                lr = epoch_lr(epoch, EPOCHS)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Exact shared epoch order & augmentation plan (identical for 224px and 384px)
                plan = make_epoch_augmentation_plan(n_train, STUDY_SEED, epoch)
                order = plan["order"]
                model.train()
                loss_sum, seen = 0.0, 0
                optimizer_steps = 0
                t_ep = time.monotonic()

                # Exact macro-batch loop: identical effective batch-32 gradient & update count for 224 and 384
                for macro_start in range(0, n_train, BATCH_SIZE):
                    macro_end = min(macro_start + BATCH_SIZE, n_train)
                    macro_len = macro_end - macro_start
                    optimizer.zero_grad(set_to_none=True)
                    macro_loss_sum = 0.0
                    for sub_start in range(macro_start, macro_end, micro_batch):
                        sub_end = min(sub_start + micro_batch, macro_end)
                        idx = order[sub_start:sub_end]
                        raw_batch = np.asarray(train_sog[idx])
                        aug_batch = apply_epoch_augmentation_slice(raw_batch, plan, sub_start, sub_end)
                        x = _to_imagenet_tensor(torch, aug_batch).to(device, non_blocking=True)
                        y_b = torch.from_numpy(y_train[idx].astype(np.float32)).to(device, non_blocking=True)
                        w_b = torch.from_numpy(self.class_weights[y_train[idx]]).to(device, non_blocking=True)

                        if use_bf16:
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                logits = model(x).reshape(-1)
                                loss_chunk = (bce(logits, y_b) * w_b).mean() * (len(idx) / float(macro_len))
                        else:
                            logits = model(x).reshape(-1)
                            loss_chunk = (bce(logits, y_b) * w_b).mean() * (len(idx) / float(macro_len))

                        if not torch.isfinite(loss_chunk):
                            raise RuntimeError(f"Non-finite training loss at ConvNeXt @{image_size}px epoch {epoch}.")
                        loss_chunk.backward()
                        macro_loss_sum += float(loss_chunk.detach().item()) * macro_len
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer_steps += 1
                    loss_sum += macro_loss_sum
                    seen += macro_len

                val_scores = score_convnext_4tta(torch, model, val_sog, device, batch_size=micro_batch)
                val_pauc = partial_auroc(y_val, val_scores, max_fpr=PARTIAL_AUC_MAX_FPR)
                val_auroc = float(roc_auc_score(y_val, val_scores))
                val_cut = locked_cutoff(y_val, val_scores, fpr_limit=FALSE_ALARM_LIMIT)
                val_m = summary_metrics(y_val, val_scores, val_cut)
                ep_sec = time.monotonic() - t_ep

                rec = {
                    "epoch": epoch,
                    "lr": lr,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": loss_sum / max(seen, 1),
                    "val_pauc_15": val_pauc,
                    "val_auroc": val_auroc,
                    "val_sens_at_10fa": val_m["Melanomas caught (%)"],
                    "val_fa_at_cut": val_m["Benign false alarms (%)"],
                    "epoch_seconds": ep_sec,
                }
                history.append(rec)
                metric_pair = (val_pauc, val_auroc)
                improved = metric_pair > best_score
                if improved:
                    best_score = metric_pair
                    best_epoch = epoch
                    no_improve = 0
                    best_weights_path = self.directory / f"convnext_{image_size}_best_epoch{epoch:02d}.pt"
                    torch.save({
                        "model": model.state_dict(), "epoch": epoch,
                        "image_size": image_size, "backbone": BACKBONE_NAME,
                    }, best_weights_path)
                    best_weights_ref = self.publish(f"convnext_{image_size}_best_epoch{epoch:02d}", best_weights_path)
                else:
                    no_improve += 1

                early_stopped = (no_improve >= PATIENCE)

                # Publish & verify immutable per-epoch recovery checkpoint, cumulative history, and state on Drive
                rec_path = self.directory / f"convnext_{image_size}_epoch{epoch:02d}_recovery.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_score": list(best_score),
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "history": history,
                }, rec_path)
                rec_ref = self.publish(f"convnext_{image_size}_epoch{epoch:02d}_recovery", rec_path)
                hist_key = f"convnext_{image_size}_epoch{epoch:02d}_history"
                self.save_df(hist_key, pd.DataFrame(history))
                self.save_json(f"convnext_{image_size}_epoch{epoch:02d}_state", {
                    "run_id": self.run_id,
                    "image_size": image_size,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score": list(best_score),
                    "no_improve": no_improve,
                    "early_stopped": early_stopped,
                    "recovery_ref": rec_ref,
                    "best_weights_ref": best_weights_ref,
                    "history_ref": self.artifacts[hist_key],
                    "history": history,
                })

                print(f"  [ConvNeXt-{image_size}] ep {epoch:02d}/{EPOCHS} | steps={optimizer_steps} | lr={lr:.1e} | "
                      f"loss={rec['train_loss']:.4f} | val pAUC15={val_pauc:.4f} | AUROC={val_auroc:.4f} | "
                      f"sens@10%FA={rec['val_sens_at_10fa']:.1f}% | {ep_sec:.1f}s {'*BEST*' if improved else ''} [Drive saved]", flush=True)

                if early_stopped:
                    print(f"  [ConvNeXt-{image_size}] Early stopping at epoch {epoch} (best epoch {best_epoch}).", flush=True)
                    break

        if best_weights_path is None or best_weights_ref is None:
            raise RuntimeError(f"No selected checkpoint recorded for ConvNeXt @{image_size}px.")

        # Reload verified best checkpoint and lock validation scores + cutoff
        best_state = torch.load(best_weights_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        val_scores = score_convnext_4tta(torch, model, val_sog, device, batch_size=micro_batch)
        cutoff = locked_cutoff(y_val, val_scores, fpr_limit=FALSE_ALARM_LIMIT)
        val_m = summary_metrics(y_val, val_scores, cutoff)

        final_best_path = self.directory / f"convnext_{image_size}_best.pt"
        final_best_path.write_bytes(best_weights_path.read_bytes())
        self.publish(f"convnext_{image_size}_weights", final_best_path)
        self.save_df(f"convnext_{image_size}_history", pd.DataFrame(history))
        val_df = self.val_manifest[["image_id", "source", "lesion_id", "group", "label"]].copy()
        val_df["model"] = f"convnext_small_{image_size}"
        val_df["seed"] = str(STUDY_SEED)
        val_df["raw_score"] = val_scores
        self.save_df(f"convnext_{image_size}_val_scores", val_df)

        selection = {
            "backbone": BACKBONE_NAME,
            "image_size": image_size,
            "seed": STUDY_SEED,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "val_pauc_15": best_score[0],
            "val_auroc": val_m["AUROC"],
            "locked_cutoff": cutoff,
            "val_metrics_at_locked_cutoff": val_m,
            "training_minutes": (time.monotonic() - t_start) / 60.0,
            "weights_sha256": digest(final_best_path),
            "best_epoch_weights_ref": best_weights_ref,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        self.save_json(f"convnext_{image_size}_selection", selection)
        self.finish_stage(stage, [
            f"convnext_{image_size}_weights",
            f"convnext_{image_size}_history",
            f"convnext_{image_size}_val_scores",
            f"convnext_{image_size}_selection",
        ])

        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"ConvNeXt-Small @{image_size}px locked: epoch {best_epoch}, AUROC {val_m['AUROC']:.4f}, "
              f"caught {val_m['Melanomas caught']}/{val_m['Melanomas']} ({val_m['Melanomas caught (%)']:.1f}%), "
              f"FA {val_m['Benign false alarms']}/{val_m['Benign']} ({val_m['Benign false alarms (%)']:.2f}%).", flush=True)
        return selection

    def _restore_model_weights(self, image_size):
        path = self.directory / f"convnext_{image_size}_best.pt"
        if not self.drive.restore(self.name(path.name), path):
            raise RuntimeError(f"Missing saved ConvNeXt @{image_size}px weights on Drive.")
        self.publish(f"convnext_{image_size}_weights", path)
        return path

    def _load_trained_convnext(self, image_size):
        torch, timm = ensure_timm()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = self._restore_model_weights(image_size)
        model = build_convnext_model(torch, timm, seed=STUDY_SEED, pretrained=False).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        return torch, model, device

    # ---- Step 1: Shared native-degraded acquisition stress test (8 appearances) + feature geometry ----

    def evaluate_convnext_stress(self, image_size):
        image_size = int(image_size)
        stage = f"stress_convnext_{image_size}"
        if self.stage_done(stage):
            stress_df = self.restore_df(f"stress_summary_convnext_{image_size}")
            feat_df = self.restore_df(f"feature_summary_convnext_{image_size}")
            print(f"Restored verified stress & feature check for ConvNeXt-Small @{image_size}px.", flush=True)
            return stress_df, feat_df

        self.ensure_shared_stress_caches()
        sel = self.restore_json(f"convnext_{image_size}_selection")
        if sel is None:
            raise RuntimeError(f"Train ConvNeXt @{image_size}px before running its stress check.")
        cutoff = float(sel["locked_cutoff"])
        labels = self.val_manifest["label"].to_numpy(dtype=int)

        torch, model, device = self._load_trained_convnext(image_size)
        micro_batch = 32 if image_size <= 224 else 16
        w = model.head.fc.weight.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        b = float(model.head.fc.bias.detach().float().cpu().numpy().reshape(-1)[0])

        stress_rows, feat_rows = [], []
        required_keys = []
        h_orig = None

        for style in ALL_STYLES:
            arr = np.load(self._shared_stress_cache_path(style, image_size), mmap_mode="r")
            h_cur, scores = extract_convnext_features_and_scores_4tta(torch, model, arr, device, batch_size=micro_batch)
            recon_err = float(np.max(np.abs((h_cur @ w + b) - scores)))
            if recon_err > 1e-4:
                raise RuntimeError(f"ConvNeXt-{image_size} linear head mismatch ({style}): {recon_err:.3e}")

            if style == "original":
                h_orig = h_cur

            score_key = f"scores_convnext_{image_size}_{style}"
            sdf = self.val_manifest[["image_id", "source", "lesion_id", "group", "label"]].copy()
            sdf["model"] = f"convnext_small_{image_size}"
            sdf["style"] = style
            sdf["raw_score"] = scores
            self.save_df(score_key, sdf)
            required_keys.append(score_key)

            m = summary_metrics(labels, scores, cutoff)
            stress_rows.append({"Model": f"ConvNeXt-Small @{image_size}px", "Style": style, "Appearance": STYLE_NAMES[style], **m})
            if style in STEP0_STYLES:
                feat_rows.append(summarize_feature_geometry(
                    f"ConvNeXt-Small @{image_size}px", style, labels, cutoff, h_orig, h_cur, w, b,
                ))
            print(f"  [Stress ConvNeXt-{image_size}] {STYLE_NAMES[style]}: fixed {m['Melanomas caught (%)']:.1f}% "
                  f"(FA {m['Benign false alarms (%)']:.2f}%), matched {m['Matched <=10% FA sens (%)']:.1f}%, AUROC {m['AUROC']:.4f}", flush=True)

        stress_df = pd.DataFrame(stress_rows)
        feat_df = pd.DataFrame(feat_rows)
        self.save_df(f"stress_summary_convnext_{image_size}", stress_df)
        self.save_df(f"feature_summary_convnext_{image_size}", feat_df)
        required_keys.extend([f"stress_summary_convnext_{image_size}", f"feature_summary_convnext_{image_size}"])
        self.finish_stage(stage, required_keys)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return stress_df, feat_df

    # ---- Step 1: Locked-cutoff external development benchmark (HIBA, DERM12345, ISIC 2020) ----

    def _prepare_external_archives(self):
        ext_dir = self.work / "external_archives"
        ext_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name in ("HIBA", "DERM12345"):
            spec = self.external_specs[name]
            url = S3_DOIS + spec["stem"] + ".zip"
            out[name] = download_file_curl(
                ext_dir / f"{name}.zip", url,
                expected_bytes=int(spec["bytes"]),
                expected_sha256=spec["sha256"],
            )
        return out

    def _ensure_isic2020_jpegs(self):
        """Fetch only the 6,772 unique ISIC 2020 JPEGs (nevi + unspecified) via bounded S3 Range reads."""
        jpeg_dir = self.work / "isic2020_jpegs"
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        needed_ids = sorted(set(self.isic_nevi_manifest["image_id"]) | set(self.isic_unspec_manifest["image_id"]))
        missing = [iid for iid in needed_ids if not (jpeg_dir / f"{iid}.jpg").is_file()]
        if not missing:
            print(f"Reusing {len(needed_ids):,} cached ISIC 2020 JPEGs.", flush=True)
            return jpeg_dir

        print(f"Fetching {len(missing):,}/{len(needed_ids):,} ISIC 2020 JPEGs via bounded S3 Range reads...", flush=True)
        with ZipFile(RemoteZip(ISIC2020_ZIP_URL, ISIC2020_ZIP_BYTES)) as zf:
            by_stem = {Path(info.filename).stem: info for info in zf.infolist()}

        absent = [iid for iid in missing if iid not in by_stem]
        if absent:
            raise RuntimeError(f"{len(absent)} ISIC 2020 images absent from archive (e.g. {absent[:3]}).")

        def _fetch_one(iid):
            raw = fetch_zip_member_bytes(ISIC2020_ZIP_URL, by_stem[iid], ISIC2020_ZIP_BYTES)
            p = jpeg_dir / f"{iid}.jpg"
            part = p.with_suffix(".jpg.part")
            part.write_bytes(raw)
            part.replace(p)
            return iid

        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_fetch_one, iid) for iid in missing]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 1000 == 0 or done == len(missing):
                    print(f"  [ISIC2020] fetched {done:,}/{len(missing):,} JPEGs.", flush=True)
        return jpeg_dir

    def _score_zip_external(self, torch, model, device, zip_path, manifest, image_size, batch_size):
        scores = []
        with ZipFile(zip_path) as zf:
            buf = []
            for idx, row in enumerate(manifest.itertuples(index=False)):
                with Image.open(io.BytesIO(zf.read(row.member))) as photo:
                    orig = ImageOps.exif_transpose(photo).convert("RGB")
                if pixel_hash(orig) != row.pixel_sha256:
                    raise RuntimeError(f"Pixel hash mismatch for external image {row.image_id}.")
                resized = np.asarray(orig.resize((image_size, image_size), Image.Resampling.BILINEAR))
                buf.append(correct_rgb(resized))
                if len(buf) == batch_size or (idx + 1) == len(manifest):
                    scores.append(score_convnext_4tta(torch, model, np.stack(buf), device, batch_size=len(buf)))
                    buf.clear()
                if (idx + 1) % 2500 == 0 or (idx + 1) == len(manifest):
                    print(f"    scored {idx + 1:,}/{len(manifest):,}", flush=True)
        return np.concatenate(scores)

    def _score_isic2020_external(self, torch, model, device, jpeg_dir, image_ids, image_size, batch_size):
        scores_by_id = {}
        buf, id_buf = [], []
        for idx, iid in enumerate(image_ids):
            with Image.open(jpeg_dir / f"{iid}.jpg") as photo:
                orig = ImageOps.exif_transpose(photo).convert("RGB")
            resized = np.asarray(orig.resize((image_size, image_size), Image.Resampling.BILINEAR))
            buf.append(correct_rgb(resized))
            id_buf.append(iid)
            if len(buf) == batch_size or (idx + 1) == len(image_ids):
                vec = score_convnext_4tta(torch, model, np.stack(buf), device, batch_size=len(buf))
                for k, v in zip(id_buf, vec):
                    scores_by_id[k] = float(v)
                buf.clear(); id_buf.clear()
            if (idx + 1) % 1500 == 0 or (idx + 1) == len(image_ids):
                print(f"    scored {idx + 1:,}/{len(image_ids):,} ISIC 2020 photos", flush=True)
        return scores_by_id

    def evaluate_external(self, image_size):
        image_size = int(image_size)
        stage = f"external_convnext_{image_size}"
        if self.stage_done(stage):
            ext_summary = self.restore_df(f"external_summary_convnext_{image_size}")
            print(f"Restored verified external benchmark for ConvNeXt-Small @{image_size}px.", flush=True)
            return ext_summary

        sel = self.restore_json(f"convnext_{image_size}_selection")
        if sel is None:
            raise RuntimeError(f"Train ConvNeXt @{image_size}px before external evaluation.")
        cutoff = float(sel["locked_cutoff"])
        ext_zips = self._prepare_external_archives()
        isic_dir = self._ensure_isic2020_jpegs()
        torch, model, device = self._load_trained_convnext(image_size)
        micro_batch = 32 if image_size <= 224 else 16

        rows = []
        required_keys = []
        for ds_name, manifest in (("HIBA", self.hiba_manifest), ("DERM12345", self.derm_manifest)):
            print(f"  [External ConvNeXt-{image_size}] Scoring {ds_name} ({len(manifest):,} photos)...", flush=True)
            vec = self._score_zip_external(torch, model, device, ext_zips[ds_name], manifest, image_size, micro_batch)
            sdf = manifest[["image_id", "label", "unit"]].copy()
            sdf["raw_score"] = vec
            key = f"external_scores_convnext_{image_size}_{ds_name}"
            self.save_df(key, sdf)
            required_keys.append(key)
            m = summary_metrics(manifest["label"], vec, cutoff)
            rows.append({"Model": f"ConvNeXt-Small @{image_size}px", "Cohort": ds_name, "Locked cutoff": cutoff, **m, "Within-patient AUROC": float("nan")})

        unique_isic_ids = sorted(set(self.isic_nevi_manifest["image_id"]) | set(self.isic_unspec_manifest["image_id"]))
        print(f"  [External ConvNeXt-{image_size}] Scoring ISIC 2020 ({len(unique_isic_ids):,} unique photos)...", flush=True)
        isic_map = self._score_isic2020_external(torch, model, device, isic_dir, unique_isic_ids, image_size, micro_batch)

        for tag, cname, manifest in (
            ("ISIC2020_nevi", "ISIC 2020 (Labelled nevi)", self.isic_nevi_manifest),
            ("ISIC2020_unspecified", "ISIC 2020 (Unspecified benign)", self.isic_unspec_manifest),
        ):
            vec = np.array([isic_map[iid] for iid in manifest["image_id"]], dtype=np.float64)
            sdf = manifest[["image_id", "patient_id", "label"]].copy()
            sdf["raw_score"] = vec
            key = f"external_scores_convnext_{image_size}_{tag}"
            self.save_df(key, sdf)
            required_keys.append(key)
            m = summary_metrics(manifest["label"], vec, cutoff)
            wp = within_patient_auroc(manifest["patient_id"], manifest["label"], vec)["mean_auroc"] if tag == "ISIC2020_nevi" else float("nan")
            rows.append({"Model": f"ConvNeXt-Small @{image_size}px", "Cohort": cname, "Locked cutoff": cutoff, **m, "Within-patient AUROC": wp})

        ext_summary = pd.DataFrame(rows)
        self.save_df(f"external_summary_convnext_{image_size}", ext_summary)
        required_keys.append(f"external_summary_convnext_{image_size}")
        self.finish_stage(stage, required_keys)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        display(ext_summary.round(3))
        return ext_summary

    # ---- Unified Report, Paired CIs, and Publication Plots ----

    def _load_nasnet_stress_scores(self, style, seed_key):
        if style in STEP0_STYLES:
            s0 = self.restore_json("step0_nasnet_scores")
            if s0 is not None and style in s0 and seed_key in s0[style]:
                return np.asarray(s0[style][seed_key], dtype=np.float64)
        ref = self.nb09_score_refs.get((style, seed_key))
        if ref is None:
            return None
        df = pd.read_csv(self.nb09_reader.fetch_artifact(ref), dtype={"image_id": str}, keep_default_na=False)
        df = df.set_index("image_id").loc[self.val_manifest["image_id"]].reset_index()
        return df["raw_score"].to_numpy(dtype=np.float64)

    def build_report(self):
        for stage in ("step0", "train_convnext_224", "train_convnext_384",
                      "stress_convnext_224", "stress_convnext_384",
                      "external_convnext_224", "external_convnext_384"):
            if not self.stage_done(stage):
                raise RuntimeError(f"Complete stage {stage} before building the final report.")

        labels_val = self.val_manifest["label"].to_numpy(dtype=int)
        groups_val = self.val_manifest["group"].to_numpy(dtype=str)
        sel224 = self.restore_json("convnext_224_selection")
        sel384 = self.restore_json("convnext_384_selection")
        cut224, cut384 = float(sel224["locked_cutoff"]), float(sel384["locked_cutoff"])
        cut_nas17, cut_nas_ens = float(self.nasnet_cutoffs[str(STUDY_SEED)]), float(self.nasnet_cutoffs["ensemble"])

        # 1. Combined Feature Geometry Table
        f_nas = self.restore_df("step0_nasnet_feature_summary")
        f_224 = self.restore_df("feature_summary_convnext_224")
        f_384 = self.restore_df("feature_summary_convnext_384")
        f_nas_mean = f_nas.groupby(["Style", "Appearance"], sort=False).mean(numeric_only=True).reset_index()
        f_nas_mean.insert(0, "Model", "NASNetMobile (3-seed mean)")
        feat_combined = pd.concat([f_nas_mean, f_224, f_384], ignore_index=True)
        self.save_df("report_feature_geometry", feat_combined)

        # 2. Combined 8-Condition Acquisition Stress Comparison
        stress_rows = []
        stress_cis = []
        for style in ALL_STYLES:
            s_nas17 = self._load_nasnet_stress_scores(style, str(STUDY_SEED))
            s_nas_ens = self._load_nasnet_stress_scores(style, "ensemble")
            s_224 = self.restore_df(f"scores_convnext_224_{style}")["raw_score"].to_numpy(dtype=np.float64)
            s_384 = self.restore_df(f"scores_convnext_384_{style}")["raw_score"].to_numpy(dtype=np.float64)
            for mlabel, vec, cut in (
                (f"NASNetMobile (seed {STUDY_SEED})", s_nas17, cut_nas17),
                ("NASNetMobile (3-seed ens)", s_nas_ens, cut_nas_ens),
                ("ConvNeXt-Small @224px (seed 17)", s_224, cut224),
                ("ConvNeXt-Small @384px (seed 17)", s_384, cut384),
            ):
                if vec is None:
                    continue
                m = summary_metrics(labels_val, vec, cut)
                stress_rows.append({"Model": mlabel, "Style": style, "Appearance": STYLE_NAMES[style], **m})
            if s_nas17 is not None:
                stress_cis.append({"Style": style, "Comparison": "ConvNeXt-224 minus NASNet-seed17",
                                   **paired_delta_ci(labels_val, groups_val, s_nas17, cut_nas17, s_224, cut224)})
            stress_cis.append({"Style": style, "Comparison": "ConvNeXt-384 minus ConvNeXt-224",
                               **paired_delta_ci(labels_val, groups_val, s_224, cut224, s_384, cut384)})
        stress_table = pd.DataFrame(stress_rows)
        self.save_df("report_stress_comparison", stress_table)
        self.save_json("report_stress_intervals", stress_cis)

        # 3. Cross-Clinic External Comparison (Val, HIBA, DERM12345, ISIC2020-Nevi, ISIC2020-Unspec)
        ext_rows = []
        ext_cis = []
        cohort_specs = [
            ("Pooled Val (HAM+BCN)", self.val_manifest, self.val_manifest["group"].to_numpy(dtype=str),
             self.nasnet_val_scores[str(STUDY_SEED)], self.nasnet_val_scores["ensemble"],
             self.restore_df("convnext_224_val_scores")["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("convnext_384_val_scores")["raw_score"].to_numpy(dtype=np.float64), None),
            ("HIBA", self.hiba_manifest, self.hiba_manifest["unit"].to_numpy(dtype=str),
             align_external_reference_scores(self.hiba_nasnet_scores, self.hiba_manifest, str(STUDY_SEED)),
             align_external_reference_scores(self.hiba_nasnet_scores, self.hiba_manifest, "ensemble"),
             self.restore_df("external_scores_convnext_224_HIBA")["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_384_HIBA")["raw_score"].to_numpy(dtype=np.float64), None),
            ("DERM12345", self.derm_manifest, self.derm_manifest["unit"].to_numpy(dtype=str),
             align_external_reference_scores(self.derm_nasnet_scores, self.derm_manifest, str(STUDY_SEED)),
             align_external_reference_scores(self.derm_nasnet_scores, self.derm_manifest, "ensemble"),
             self.restore_df("external_scores_convnext_224_DERM12345")["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_384_DERM12345")["raw_score"].to_numpy(dtype=np.float64), None),
            ("ISIC 2020 (Labelled nevi)", self.isic_nevi_manifest, self.isic_nevi_manifest["patient_id"].to_numpy(dtype=str),
             None, self.isic_nevi_manifest["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_224_ISIC2020_nevi")["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_384_ISIC2020_nevi")["raw_score"].to_numpy(dtype=np.float64),
             self.isic_nevi_manifest["patient_id"].to_numpy(dtype=str)),
            ("ISIC 2020 (Unspecified benign)", self.isic_unspec_manifest, self.isic_unspec_manifest["patient_id"].to_numpy(dtype=str),
             None, self.isic_unspec_manifest["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_224_ISIC2020_unspecified")["raw_score"].to_numpy(dtype=np.float64),
             self.restore_df("external_scores_convnext_384_ISIC2020_unspecified")["raw_score"].to_numpy(dtype=np.float64), None),
        ]

        for cname, mf, grps, s_n17, s_nens, s_224, s_384, wp_pids in cohort_specs:
            y = mf["label"].to_numpy(dtype=int)
            for mlabel, vec, cut in (
                (f"NASNetMobile (seed {STUDY_SEED})", s_n17, cut_nas17),
                ("NASNetMobile (3-seed ens)", s_nens, cut_nas_ens),
                ("ConvNeXt-Small @224px (seed 17)", s_224, cut224),
                ("ConvNeXt-Small @384px (seed 17)", s_384, cut384),
            ):
                if vec is None:
                    continue
                m = summary_metrics(y, vec, cut)
                wp = within_patient_auroc(wp_pids, y, vec)["mean_auroc"] if wp_pids is not None else float("nan")
                ext_rows.append({"Cohort": cname, "Model": mlabel, **m, "Within-patient AUROC": wp})
            if s_nens is not None:
                ext_cis.append({"Cohort": cname, "Comparison": "ConvNeXt-224 minus NASNet-ens",
                                **paired_delta_ci(y, grps, s_nens, cut_nas_ens, s_224, cut224)})
                ext_cis.append({"Cohort": cname, "Comparison": "ConvNeXt-384 minus NASNet-ens",
                                **paired_delta_ci(y, grps, s_nens, cut_nas_ens, s_384, cut384)})
            ext_cis.append({"Cohort": cname, "Comparison": "ConvNeXt-384 minus ConvNeXt-224",
                            **paired_delta_ci(y, grps, s_224, cut224, s_384, cut384)})

        ext_table = pd.DataFrame(ext_rows)
        self.save_df("report_external_comparison", ext_table)
        self.save_json("report_external_intervals", ext_cis)

        fig_feat = self._plot_feature_comparison(feat_combined)
        self.save_figure("report_feature_decomposition_fig", fig_feat)
        fig_stress = self._plot_stress_comparison(stress_table)
        self.save_figure("report_stress_comparison_fig", fig_stress)
        fig_ext = self._plot_external_comparison(ext_table)
        self.save_figure("report_external_comparison_fig", fig_ext)

        nas_lc = f_nas_mean[f_nas_mean["Style"] == "contrast_low"].iloc[0]
        c224_lc = f_224[f_224["Style"] == "contrast_low"].iloc[0]
        c384_lc = f_384[f_384["Style"] == "contrast_low"].iloc[0]
        verdict = {
            "run_id": self.run_id,
            "step0_gate_nasnet_low_contrast": {
                "norm_ratio_mel": float(nas_lc["Median ||h|| ratio (mel)"]),
                "cos_sim_mel": float(nas_lc["Median cos(h0, h1) (mel)"]),
                "dz_norm_mel": float(nas_lc["Median dz norm-term (mel)"]),
                "dz_dir_mel": float(nas_lc["Median dz dir-term (mel)"]),
                "norm_share_of_abs_dz_mel": float(nas_lc["Median norm share of |dz| (mel)"]),
                "linear_fixed_sens_pct": float(nas_lc["Linear fixed sens (%)"]),
                "posthoc_cosine_fixed_sens_pct": float(nas_lc["Posthoc cosine fixed sens (%)"]),
            },
            "step1_convnext_low_contrast_feature_check": {
                "convnext_224_norm_ratio_mel": float(c224_lc["Median ||h|| ratio (mel)"]),
                "convnext_224_fixed_sens_pct": float(c224_lc["Linear fixed sens (%)"]),
                "convnext_384_norm_ratio_mel": float(c384_lc["Median ||h|| ratio (mel)"]),
                "convnext_384_fixed_sens_pct": float(c384_lc["Linear fixed sens (%)"]),
            },
            "note": "Review paired CIs, original-quality retention, and feature decomposition before deciding whether Step 2 needs a cosine head, contrast/resolution augmentation, and/or cutoff-focused loss.",
        }
        self.save_json("report_verdict", verdict)
        self.finish_stage("report", [
            "report_feature_geometry",
            "report_stress_comparison",
            "report_stress_intervals",
            "report_external_comparison",
            "report_external_intervals",
            "report_feature_decomposition_fig",
            "report_stress_comparison_fig",
            "report_external_comparison_fig",
            "report_verdict",
        ])

        print("\n=== STEP 0 + STEP 1: FEATURE GEOMETRY (Low Contrast & Lower Resolution) ===")
        display(feat_combined[feat_combined["Style"].isin(["contrast_low", "gamma_bright", "lower_resolution", "combined_low_contrast_low_res"])][
            ["Model", "Appearance", "Median ||h|| ratio (mel)", "Median cos(h0, h1) (mel)",
             "Median dz norm-term (mel)", "Median dz dir-term (mel)", "Linear fixed sens (%)", "Posthoc cosine fixed sens (%)"]
        ].round(3))

        print("\n=== STEP 1: ACQUISITION STRESS TEST (8 Appearances, Locked Source Cutoffs) ===")
        display(stress_table[["Model", "Appearance", "Melanomas caught (%)", "Benign false alarms (%)", "Matched <=10% FA sens (%)", "AUROC"]].round(3))

        print("\n=== STEP 1: CROSS-CLINIC EXTERNAL BENCHMARK (Locked Source Cutoffs) ===")
        display(ext_table[["Cohort", "Model", "Melanomas caught (%)", "Benign false alarms (%)", "Matched <=10% FA sens (%)", "AUROC", "Within-patient AUROC"]].round(3))
        return verdict

    def _plot_feature_comparison(self, feat_combined):
        sub = feat_combined[feat_combined["Style"].isin(["contrast_low", "gamma_bright", "lower_resolution", "combined_low_contrast_low_res"])].copy()
        styles = ["contrast_low", "gamma_bright", "lower_resolution", "combined_low_contrast_low_res"]
        models = ["NASNetMobile (3-seed mean)", "ConvNeXt-Small @224px", "ConvNeXt-Small @384px"]
        x = np.arange(len(styles))
        w = 0.25
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
        for i, m in enumerate(models):
            part = sub[sub["Model"] == m].set_index("Style").reindex(styles)
            ax1.bar(x + (i - 1) * w, part["Median ||h|| ratio (mel)"], width=w, label=m)
            ax2.bar(x + (i - 1) * w, part["Median norm share of |dz| (mel)"] * 100.0, width=w, label=m)
        ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax1.set_xticks(x); ax1.set_xticklabels([SHORT_STYLE_LABELS[s] for s in styles], fontsize=8.5)
        ax1.set_ylabel("Median ||h_stress|| / ||h_orig|| (melanomas)")
        ax1.set_title("Feature-norm shrinkage: NASNetMobile vs. LayerNorm ConvNeXt")
        ax1.legend(fontsize=8)
        ax2.set_xticks(x); ax2.set_xticklabels([SHORT_STYLE_LABELS[s] for s in styles], fontsize=8.5)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Norm share of |dz| (%)")
        ax2.set_title("Share of melanoma logit shift caused by norm vs. direction")
        ax2.legend(fontsize=8)
        return fig

    def _plot_stress_comparison(self, stress_table):
        x = np.arange(len(ALL_STYLES))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")
        for mlabel, style_fmt in (
            ("NASNetMobile (3-seed ens)", "o--"),
            ("ConvNeXt-Small @224px (seed 17)", "s-"),
            ("ConvNeXt-Small @384px (seed 17)", "^-"),
        ):
            sub = stress_table[stress_table["Model"] == mlabel].set_index("Style").reindex(ALL_STYLES)
            ax1.plot(x, sub["Melanomas caught (%)"], style_fmt, label=mlabel, linewidth=2)
            ax2.plot(x, sub["Matched <=10% FA sens (%)"], style_fmt, label=mlabel, linewidth=2)
        for ax, title, ylabel in (
            (ax1, "Fixed locked cutoff: Melanomas caught (%)", "Melanomas caught (%)"),
            (ax2, "Matched <=10% false alarms: Melanomas caught (%)", "Melanomas caught (%)"),
        ):
            ax.set_xticks(x); ax.set_xticklabels([SHORT_STYLE_LABELS[s] for s in ALL_STYLES], fontsize=8)
            ax.set_ylim(0, 100); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
        return fig

    def _plot_external_comparison(self, ext_table):
        cohorts = ["Pooled Val (HAM+BCN)", "HIBA", "DERM12345", "ISIC 2020 (Labelled nevi)", "ISIC 2020 (Unspecified benign)"]
        short_c = ["Val\n(HAM+BCN)", "HIBA\n(750)", "DERM12345\n(10,351)", "ISIC 2020\n(Nevi)", "ISIC 2020\n(Unspec)"]
        models = ["NASNetMobile (3-seed ens)", "ConvNeXt-Small @224px (seed 17)", "ConvNeXt-Small @384px (seed 17)"]
        x = np.arange(len(cohorts))
        w = 0.25
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6), layout="constrained")
        for i, m in enumerate(models):
            sub = ext_table[ext_table["Model"] == m].set_index("Cohort").reindex(cohorts)
            ax1.bar(x + (i - 1) * w, sub["Melanomas caught (%)"], width=w, label=m)
            ax2.bar(x + (i - 1) * w, sub["AUROC"], width=w, label=m)
        ax1.set_xticks(x); ax1.set_xticklabels(short_c, fontsize=8.5)
        ax1.set_ylim(0, 105); ax1.set_ylabel("Melanomas caught at locked cutoff (%)")
        ax1.set_title("Locked-cutoff sensitivity across clinics")
        ax1.legend(fontsize=8)
        ax2.set_xticks(x); ax2.set_xticklabels(short_c, fontsize=8.5)
        ax2.set_ylim(0.65, 1.0); ax2.set_ylabel("AUROC")
        ax2.set_title("Discrimination (AUROC) across clinics")
        ax2.legend(fontsize=8)
        return fig

    def finalize_and_release(self):
        required_stages = (
            "step0", "train_convnext_224", "train_convnext_384",
            "stress_convnext_224", "stress_convnext_384",
            "external_convnext_224", "external_convnext_384", "report",
        )
        for stage in required_stages:
            if not self.stage_done(stage):
                raise RuntimeError(f"Stage {stage} is incomplete; runtime will stay connected.")
        for ref in self.artifacts.values():
            self.drive.verify_receipt(ref)
        for ref in self.nasnet_models.values():
            self.train_reader.verify_artifact(ref)

        final_receipt = {
            "run_id": self.run_id,
            "protocol_sha256": canonical_hash(self.protocol),
            "folder_id": self.drive.folder_id,
            "source_models": self.nasnet_models,
            "files": self.artifacts,
            "completed_stages": list(required_stages),
        }
        self.save_json("durable_save_receipt", final_receipt)
        self.drive.verify_receipt(self.artifacts["durable_save_receipt"])
        print(f"All Step 0 + Step 1 outputs, checkpoints, figures, and hashes verified on Drive: {self.drive.link}", flush=True)
        from google.colab import runtime
        runtime.unassign()
