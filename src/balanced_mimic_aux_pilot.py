"""Notebook 21: From-ImageNet Source-Balanced ConvNeXt-224 + Auxiliary Mimic Head.

Controlled 1-seed (seed=17) two-arm experiment starting from ImageNet-1k
`convnext_small.fb_in1k` (NOT the MoleMax-biased NB10 checkpoint):
  - Arm 1 (`balanced_mel_nv`): Source-balanced MEL vs. NV across
    HAM-MoleMax, HAM-other, BCN, MSK-1/2, and Sydney (RPAH), 15 epochs
    with cosine LR and calibration-only early stopping (pAUC15, patience=5).
  - Arm 2 (`balanced_aux_mimics`): Identical MEL/NV batches + auxiliary 5-class
    head (NV, MEL, BKL, DF, VASC; weight=0.25) trained on MEL/NV + 577 verified
    training mimics. Mimics never feed negative BCE into the primary binary head,
    keeping the MEL-vs-NV operating point unshifted.
"""
import gc
import hashlib
import io
import json
import math
import platform
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import sklearn
from IPython.display import display
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve
from urllib.request import Request, urlopen

from acquisition_stress import canonical_hash, correct_rgb, metrics, pixel_hash
from backbone_resolution_study import (
    BACKBONE_NAME,
    BATCH_SIZE,
    DROP_PATH_RATE,
    TRAIN_FOLDER,
    TRAIN_RECEIPT_SHA256,
    TRAIN_RUN,
    WEIGHT_DECAY,
    _to_imagenet_tensor,
    apply_epoch_augmentation_slice,
    build_convnext_model,
    download_file_curl,
    make_epoch_augmentation_plan,
    partial_auroc,
    score_convnext_4tta,
    validate_split,
)
from benign_mimic_training_pilot import source_cutoff
from inference_store import FolderReader, digest
from paired_view_data import Photos, RepeatIndex
from paired_view_store import ProjectRunStore
from resolution_augmentation_pilot import MICRO_BATCH, SEED, SIZE, calibration_roles

TITLE = "21 ImageNet Source-Balanced + Aux Mimics"
ARMS = ("balanced_mel_nv", "balanced_aux_mimics")
MAX_EPOCHS = 15
PATIENCE = 5
WARMUP_LR = 1e-5
PEAK_LR = 5e-5
MIN_LR = 2e-6
AUX_WEIGHT = 0.25
MIMICS_PER_BATCH = 8
STRATUM_CAP = 3.0
BOOTSTRAP_DRAWS = 1000
TIMM_VERSION = "1.0.22"
SUBTYPE_TO_ID = {"NV": 0, "MEL": 1, "BKL": 2, "DF": 3, "VASC": 4}


def cosine_lr_15(epoch, total_epochs=MAX_EPOCHS):
    if epoch == 1:
        return WARMUP_LR
    if total_epochs <= 2:
        return PEAK_LR
    progress = (epoch - 2) / (total_epochs - 2)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * progress))


def fetch_url_bytes(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def assign_ham_stratum(image_ids, ham_meta):
    ds_map = dict(zip(ham_meta["image_id"], ham_meta["dataset"]))
    out = []
    for iid in image_ids:
        ds = ds_map.get(iid, "")
        if ds == "vidir_molemax":
            out.append("HAM_molemax")
        elif ds:
            out.append("HAM_other")
        else:
            raise KeyError(f"Missing HAM dataset metadata for {iid}")
    return out


def compute_balanced_sample_weights(strata, labels, cap=STRATUM_CAP):
    """Capped inverse-sqrt stratum weights, normalized per class so MEL:NV = 1:1."""
    s = np.asarray(strata, dtype=str)
    y = np.asarray(labels, dtype=int)
    weights = np.ones(len(y), dtype=np.float64)
    for cls in (0, 1):
        mask_cls = (y == cls)
        unique_s, counts = np.unique(s[mask_cls], return_counts=True)
        # Inverse-sqrt relative to the largest stratum in this diagnosis
        max_c = float(np.max(counts))
        raw_map = {st: min(float(cap), math.sqrt(max_c / float(c))) for st, c in zip(unique_s, counts)}
        for st, w_val in raw_map.items():
            weights[mask_cls & (s == st)] = w_val
        # Normalize so total class weight is balanced (mean weight per class = 0.5 * N / N_cls)
        weights[mask_cls] *= (len(y) / (2.0 * float(np.sum(weights[mask_cls]))))
    return weights.astype(np.float32)


def compute_mimic_subtype_weights(subtypes):
    """Balanced weights across BKL / DF / VASC for auxiliary mimic sampling."""
    st = np.asarray(subtypes, dtype=str)
    unique_s, counts = np.unique(st, return_counts=True)
    max_c = float(np.max(counts))
    w_map = {k: min(3.0, math.sqrt(max_c / float(c))) for k, c in zip(unique_s, counts)}
    probs = np.array([w_map[k] for k in st], dtype=np.float64)
    return probs / probs.sum()


class ConvNeXtDualHead(object):
    """Lightweight wrapper around timm ConvNeXt-Small adding an optional 5-class aux head."""

    @staticmethod
    def build(torch, timm, seed=SEED, pretrained=True, num_aux=5):
        import torch.nn as nn

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        base = build_convnext_model(torch, timm, seed=seed, pretrained=pretrained)

        class _Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                in_features = backbone.head.fc.in_features
                self.aux_head = nn.Linear(in_features, num_aux)

            def forward_penult(self, x):
                spatial = self.backbone.forward_features(x)
                return self.backbone.forward_head(spatial, pre_logits=True)

            def forward(self, x):
                penult = self.forward_penult(x)
                return self.backbone.head.fc(penult).reshape(-1), self.aux_head(penult)

        return _Model(base)


def score_dual_4tta(torch, model, pixels, device, batch_size=BATCH_SIZE):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start:start + batch_size]
            views = (
                batch,
                np.flip(batch, axis=2),
                np.flip(batch, axis=1),
                np.rot90(batch, k=2, axes=(1, 2)),
            )
            acc = None
            for v in views:
                x = _to_imagenet_tensor(torch, v).to(device, non_blocking=True)
                logits, _ = model(x)
                s = logits.detach().float().cpu().numpy().astype(np.float64)
                acc = s if acc is None else acc + s
            out.append(acc / 4.0)
    return np.concatenate(out, axis=0)


class BalancedMimicAuxPilot:
    """From-ImageNet source-balanced ConvNeXt-224 with & without auxiliary benign-mimic head."""

    title = TITLE

    def __init__(self, service, bundle, code, work="/content/balanced_mimic_aux"):
        import timm
        import torch

        if timm.__version__ != TIMM_VERSION:
            raise RuntimeError(f"Requires timm {TIMM_VERSION}, found {timm.__version__}.")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Select a BF16-capable GPU (e.g., Colab L4).")
        self.torch, self.timm = torch, timm
        self.device = torch.device("cuda:0")
        self.bundle, self.code = Path(bundle), Path(code)
        self.run_id = digest(self.bundle)[:20]
        self.root = Path(work) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

        self.drive = ProjectRunStore(service, self.run_id, title=self.title)
        self.drive.claim_writer()

        # Verify NB06 archive specs
        nb06 = FolderReader(service, TRAIN_FOLDER, self.root / "nb06")
        nb06_rec = json.loads(nb06.fetch(f"{TRAIN_RUN}__save_receipt.json").read_bytes())
        if digest(self.root / "nb06" / f"{TRAIN_RUN}__save_receipt.json") != TRAIN_RECEIPT_SHA256:
            raise RuntimeError("NB06 save receipt mismatch.")
        val_report = json.loads(nb06.fetch_artifact(nb06_rec["validation_report"]).read_bytes())
        self.sources = val_report["protocol"]["image_sources"]

        # Load frozen split & metadata from bundle
        self.split = validate_split(pd.read_csv(self.code / "pooled_ham_bcn_split.csv", dtype=str, keep_default_na=False))
        self.ham_meta = pd.read_csv(self.code / "ham_metadata.csv", dtype=str, keep_default_na=False)
        self.mimic_split = pd.read_csv(self.code / "mimic_split.csv", dtype=str, keep_default_na=False)
        self.site_checks = pd.read_csv(self.code / "site_image_checks.csv", dtype=str, keep_default_na=False)
        self.prove_ids = pd.read_csv(self.code / "prove_ids.csv", dtype=str, keep_default_na=False)
        self.source_predictions = pd.read_csv(self.code / "source_predictions.csv", keep_default_na=False, float_precision="round_trip")

        self.base_train = self.split[self.split.split.eq("train")].sort_values(["source", "image_id"]).reset_index(drop=True)
        self.validation = self.split[self.split.split.eq("validation")].sort_values(["source", "image_id"]).reset_index(drop=True)
        self.roles = calibration_roles(self.validation)
        self.cal_mask = self.roles["role"].eq("calibration").to_numpy()
        self.eval_mask = self.roles["role"].eq("evaluation").to_numpy()

        self.protocol = {
            "task": "From-ImageNet source-balanced ConvNeXt-224 vs. Auxiliary Benign-Mimic Head",
            "bundle_sha256": digest(self.bundle),
            "backbone": BACKBONE_NAME,
            "initialization": "ImageNet-1k pretrained (NOT MoleMax-biased NB10 checkpoint)",
            "seed": SEED,
            "image_size": SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "selection_split": "calibration_only (625 cases; 937 eval cases untouched)",
            "selection_metric": "val_pauc_15 (FPR<=0.15) tie-break val_auroc",
            "lr_schedule": {"warmup": WARMUP_LR, "peak": PEAK_LR, "min": MIN_LR},
            "batch_size": BATCH_SIZE,
            "micro_batch": MICRO_BATCH,
            "weight_decay": WEIGHT_DECAY,
            "drop_path_rate": DROP_PATH_RATE,
            "stratum_cap": STRATUM_CAP,
            "aux_weight": AUX_WEIGHT,
            "mimics_per_batch": MIMICS_PER_BATCH,
            "arms": list(ARMS),
            "software": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "timm": timm.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pillow": PIL.__version__,
                "sklearn": sklearn.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        self.protocol_sha = canonical_hash(self.protocol)
        self.refs = {}

        receipt = self.restore_json("durable_save_receipt")
        self.complete = receipt is not None
        if receipt:
            if (receipt["run_id"], receipt["protocol_sha256"]) != (self.run_id, self.protocol_sha):
                raise RuntimeError("Completed run belongs to a different protocol.")
            for ref in receipt["files"].values():
                self.drive.verify_receipt(ref)
            print("Run already completed and verified on Drive:", self.drive.link)
            return

        self.drive.quota(1024 * 1024**2)
        self.refs["protocol"] = self.save_json("protocol", {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "protocol": self.protocol})
        self.refs["bundle"] = self.publish(self.bundle)
        self.refs["roles"] = self.save_csv("validation_roles", self.roles)
        print(f"Run {self.run_id} ready | Drive: {self.drive.link}", flush=True)

    def publish(self, path):
        self.drive.check_writer()
        ref = self.drive.publish(Path(path), f"{self.run_id}__{Path(path).name}")
        self.drive.verify_receipt(ref)
        return ref

    def save_json(self, stem, value):
        path = self.root / f"{stem}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return self.publish(path)

    def save_csv(self, stem, df):
        path = self.root / f"{stem}.csv"
        df.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        return self.publish(path)

    def restore_json(self, stem):
        path = self.root / f"{stem}.json"
        if not self.drive.restore(f"{self.run_id}__{path.name}", path):
            return None
        return json.loads(path.read_bytes())

    def restore_file(self, ref):
        prefix = self.run_id + "__"
        path = self.root / ref["name"].removeprefix(prefix)
        self.drive.verify_receipt(ref)
        if not self.drive.restore(ref["name"], path) or digest(path) != ref["sha256"]:
            raise RuntimeError(f"Drive file mismatch for {ref['name']}.")
        return path

    def _save_ckpt(self, prefix, payload):
        part = self.root / f"{prefix}-{uuid.uuid4().hex}.part"
        with part.open("xb") as f:
            self.torch.save(payload, f)
        dest = self.root / f"{prefix}-{digest(part)[:20]}.pt"
        part.replace(dest)
        return self.publish(dest)

    def prepare_images(self):
        cache_meta = self.root / "prepared_caches.json"
        if cache_meta.is_file():
            meta = json.loads(cache_meta.read_bytes())
            if meta.get("protocol_sha") == self.protocol_sha and all((self.root / k).is_file() for k in meta["files"]):
                self.train_table = pd.read_csv(self.root / "train_manifest.csv", dtype=str, keep_default_na=False)
                self.train_table["label"] = self.train_table["label"].astype(int)
                self.train_table["sample_weight"] = self.train_table["sample_weight"].astype(float)
                self.mimic_train_table = pd.read_csv(self.root / "mimic_train_manifest.csv", dtype=str, keep_default_na=False)
                self.mimic_eval_table = pd.read_csv(self.root / "mimic_eval_manifest.csv", dtype=str, keep_default_na=False)
                self.images = {k: np.load(self.root / fn, mmap_mode="r") for k, fn in meta["arrays"].items()}
                print(f"Reusing verified caches: train_mel_nv={len(self.train_table):,}, mimic_train={len(self.mimic_train_table):,}.", flush=True)
                return

        archives_dir = self.root / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        for src_name in ("HAM", "BCN"):
            spec = self.sources[src_name]
            download_file_curl(archives_dir / f"{src_name}.zip", spec["url"], expected_bytes=int(spec["bytes"]), expected_md5=spec.get("md5"))

        # Screen MSK/Sydney site candidates against HAM/BCN + PROVe-AI using native perceptual hashes
        from audit import perceptual_hashes

        passing = self.site_checks[self.site_checks.passes_available_checks.eq("True")].copy()
        index = RepeatIndex(passing.set_index("image_id").phash.to_dict())
        overlap_hits = []

        ham_zip = ZipFile(archives_dir / "HAM.zip")
        bcn_zip = ZipFile(archives_dir / "BCN.zip")
        zmap = {"HAM": ham_zip, "BCN": bcn_zip}

        def _decode_source(source, member, expected_sha):
            with Image.open(BytesIO(zmap[source].read(member))) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            if pixel_hash(rgb) != expected_sha:
                raise RuntimeError(f"Pixel hash mismatch for {source}/{member}")
            overlap_hits.extend((cand, f"{source}:{member}") for cand in sorted(index.matches(perceptual_hashes(rgb))))
            return correct_rgb(np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))

        # Decode HAM+BCN train & val, plus mimics
        n_base = len(self.base_train)
        val_arr = np.lib.format.open_memmap(self.root / "val.npy", mode="w+", dtype="uint8", shape=(len(self.validation), SIZE, SIZE, 3))
        for i, r in enumerate(self.validation.itertuples(index=False)):
            val_arr[i] = _decode_source(r.source, r.member, r.pixel_sha256)
        val_arr.flush()

        mimic_train_df = self.mimic_split[self.mimic_split.split.eq("train")].sort_values(["source", "image_id"]).reset_index(drop=True)
        mimic_eval_df = self.mimic_split[self.mimic_split.split.isin(["development", "test"])].sort_values(["split", "source", "image_id"]).reset_index(drop=True)

        mimic_tr_arr = np.lib.format.open_memmap(self.root / "mimic_train.npy", mode="w+", dtype="uint8", shape=(len(mimic_train_df), SIZE, SIZE, 3))
        for i, r in enumerate(mimic_train_df.itertuples(index=False)):
            mimic_tr_arr[i] = _decode_source(r.source, r.member, r.pixel_sha256)
        mimic_tr_arr.flush()

        mimic_ev_arr = np.lib.format.open_memmap(self.root / "mimic_eval.npy", mode="w+", dtype="uint8", shape=(len(mimic_eval_df), SIZE, SIZE, 3))
        for i, r in enumerate(mimic_eval_df.itertuples(index=False)):
            mimic_ev_arr[i] = _decode_source(r.source, r.member, r.pixel_sha256)
        mimic_ev_arr.flush()

        # Check base train + build source strata
        ham_mask = self.base_train["source"].eq("HAM")
        base_strata = np.where(ham_mask, "", "BCN")
        ham_strata = assign_ham_stratum(self.base_train.loc[ham_mask, "image_id"].tolist(), self.ham_meta)
        base_strata[ham_mask.to_numpy()] = ham_strata

        base_imgs = []
        for i, r in enumerate(self.base_train.itertuples(index=False)):
            base_imgs.append(_decode_source(r.source, r.member, r.pixel_sha256))
            if (i + 1) % 2000 == 0:
                print(f"  Decoded {i + 1:,}/{n_base:,} HAM+BCN training photos...", flush=True)
        ham_zip.close(); bcn_zip.close()

        # Also check PROVe-AI ids if archive available via site_expansion_v1 / S3 (here passing already screened against PROVe/ISIC2020/HIBA)
        linked_patients = set(passing.loc[passing.image_id.isin({c for c, _ in overlap_hits}), "patient_id"])
        clean_sites = passing[~passing.patient_id.isin(linked_patients)].sort_values(["site", "image_id"]).reset_index(drop=True)
        print(f"Added clean biopsy-confirmed hospital cases: {len(clean_sites):,} "
              f"(excluded {len(passing) - len(clean_sites)} overlap-linked).", flush=True)

        def _fetch_site(row):
            raw = fetch_url_bytes(row.url)
            if hashlib.sha256(raw).hexdigest() != row.jpeg_sha256:
                raise RuntimeError(f"JPEG hash mismatch for {row.image_id}")
            with Image.open(BytesIO(raw)) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            if pixel_hash(rgb) != row.pixel_sha256:
                raise RuntimeError(f"Pixel hash mismatch for {row.image_id}")
            return correct_rgb(np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))

        with ThreadPoolExecutor(6) as pool:
            site_imgs = list(pool.map(_fetch_site, clean_sites.itertuples(index=False)))

        n_total = n_base + len(clean_sites)
        train_arr = np.lib.format.open_memmap(self.root / "train.npy", mode="w+", dtype="uint8", shape=(n_total, SIZE, SIZE, 3))
        for i, img in enumerate(base_imgs):
            train_arr[i] = img
        for i, img in enumerate(site_imgs, start=n_base):
            train_arr[i] = img
        train_arr.flush()

        base_df = pd.DataFrame({
            "image_id": self.base_train["image_id"],
            "source": self.base_train["source"],
            "stratum": base_strata,
            "label": self.base_train["label"].astype(int),
        })
        site_df = pd.DataFrame({
            "image_id": clean_sites["image_id"],
            "source": clean_sites["site"],
            "stratum": clean_sites["site"],
            "label": clean_sites["diagnosis"].map({"melanoma": 1, "atypical_nevus": 0}).astype(int),
        })
        self.train_table = pd.concat([base_df, site_df], ignore_index=True)
        self.train_table["sample_weight"] = compute_balanced_sample_weights(
            self.train_table["stratum"], self.train_table["label"], cap=STRATUM_CAP
        )
        self.mimic_train_table = mimic_train_df
        self.mimic_eval_table = mimic_eval_df

        self.train_table.to_csv(self.root / "train_manifest.csv", index=False)
        self.mimic_train_table.to_csv(self.root / "mimic_train_manifest.csv", index=False)
        self.mimic_eval_table.to_csv(self.root / "mimic_eval_manifest.csv", index=False)
        self.refs["train_manifest"] = self.save_csv("train_manifest", self.train_table)

        arrays = {"train": "train.npy", "val": "val.npy", "mimic_train": "mimic_train.npy", "mimic_eval": "mimic_eval.npy"}
        cache_meta.write_text(json.dumps({
            "protocol_sha": self.protocol_sha,
            "arrays": arrays,
            "files": ["train_manifest.csv", "mimic_train_manifest.csv", "mimic_eval_manifest.csv", *arrays.values()],
        }), encoding="utf8")
        self.images = {k: np.load(self.root / fn, mmap_mode="r") for k, fn in arrays.items()}
        print("Stratum summary (train):")
        display(self.train_table.groupby(["stratum", "label"])["sample_weight"].agg(["count", "mean"]).round(3))

    def _latest_epoch(self, arm):
        for ep in range(MAX_EPOCHS, 0, -1):
            st = self.restore_json(f"{arm}_epoch{ep:02d}")
            if st is not None:
                if (st["run_id"], st["protocol_sha256"], st["arm"], st["epoch"]) != (self.run_id, self.protocol_sha, arm, ep):
                    raise RuntimeError("Epoch checkpoint belongs to another run.")
                for role in ("selected", "recovery", "history"):
                    self.drive.verify_receipt(st["files"][role])
                return st
        return {"epoch": 0}

    def train(self, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}")
        done_flag = self.restore_json(f"{arm}_complete")
        if done_flag is not None:
            print(f"{arm}: already trained and verified (best epoch {done_flag['best_epoch']}).", flush=True)
            return done_flag

        torch = self.torch
        model = ConvNeXtDualHead.build(torch, self.timm, seed=SEED, pretrained=True).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)

        state = self._latest_epoch(arm)
        history = []
        best_score = (-1.0, -1.0)
        best_epoch = 0
        best_ref = None
        no_improve = 0
        start_epoch = 1

        if state["epoch"]:
            rec = torch.load(self.restore_file(state["files"]["recovery"]), map_location="cpu", weights_only=False)
            model.load_state_dict(rec["model"], strict=True)
            optimizer.load_state_dict(rec["optimizer"])
            history = list(rec["history"])
            best_score = tuple(rec["best_score"])
            best_epoch = int(rec["best_epoch"])
            best_ref = rec["best_ref"]
            no_improve = int(rec["no_improve"])
            start_epoch = int(state["epoch"]) + 1
            print(f"Resuming {arm} from epoch {start_epoch} (best epoch {best_epoch}).", flush=True)

        y_train = self.train_table["label"].to_numpy(dtype=np.int64)
        w_train = self.train_table["sample_weight"].to_numpy(dtype=np.float32)
        y_cal = self.roles.loc[self.cal_mask, "label"].to_numpy(dtype=int)
        val_cal_imgs = np.asarray(self.images["val"][self.cal_mask])

        if arm == "balanced_aux_mimics":
            mimic_y = self.mimic_train_table["label"].map(SUBTYPE_TO_ID).to_numpy(dtype=np.int64)
            mimic_probs = compute_mimic_subtype_weights(self.mimic_train_table["label"])

        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            lr = cosine_lr_15(epoch, MAX_EPOCHS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            plan = make_epoch_augmentation_plan(len(self.train_table), SEED, epoch)
            mimic_rng = np.random.default_rng([SEED + 5000, epoch])
            model.train()
            t0 = time.monotonic()
            loss_sum, seen = 0.0, 0

            for macro_start in range(0, len(self.train_table), BATCH_SIZE):
                macro_end = min(macro_start + BATCH_SIZE, len(self.train_table))
                macro_len = macro_end - macro_start
                optimizer.zero_grad(set_to_none=True)

                for pos in range(macro_start, macro_end, MICRO_BATCH):
                    stop = min(pos + MICRO_BATCH, macro_end)
                    ids = plan["order"][pos:stop]
                    pixels = apply_epoch_augmentation_slice(self.images["train"][ids], plan, pos, stop)
                    x = _to_imagenet_tensor(torch, pixels).to(self.device, non_blocking=True)
                    y_b = torch.as_tensor(y_train[ids], dtype=torch.float32, device=self.device)
                    w_b = torch.as_tensor(w_train[ids], dtype=torch.float32, device=self.device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        bin_logits, aux_logits = model(x)
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, y_b, reduction="none")
                        loss_bin = (bce * w_b).sum() / float(macro_len)
                        if arm == "balanced_aux_mimics":
                            aux_y = torch.as_tensor(y_train[ids], dtype=torch.long, device=self.device)
                            loss_aux_main = torch.nn.functional.cross_entropy(aux_logits, aux_y) * (len(ids) / float(macro_len))
                            loss = loss_bin + AUX_WEIGHT * loss_aux_main
                        else:
                            loss = loss_bin
                    loss.backward()
                    loss_sum += float(loss.detach()) * float(macro_len)

                if arm == "balanced_aux_mimics":
                    m_ids = mimic_rng.choice(len(self.mimic_train_table), size=MIMICS_PER_BATCH, replace=True, p=mimic_probs)
                    m_pix = np.asarray(self.images["mimic_train"][m_ids])
                    # Simple dihedral flip/rot90 for mimics
                    for mi in range(len(m_pix)):
                        m_pix[mi] = np.rot90(m_pix[mi], k=int(mimic_rng.integers(0, 4)))
                        if mimic_rng.integers(0, 2):
                            m_pix[mi] = np.fliplr(m_pix[mi])
                    mx = _to_imagenet_tensor(torch, m_pix).to(self.device, non_blocking=True)
                    my = torch.as_tensor(mimic_y[m_ids], dtype=torch.long, device=self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, m_aux_logits = model(mx)
                        m_loss = AUX_WEIGHT * torch.nn.functional.cross_entropy(m_aux_logits, my)
                    m_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                seen += macro_len

            cal_scores = score_dual_4tta(torch, model, val_cal_imgs, self.device, batch_size=BATCH_SIZE)
            cal_pauc = partial_auroc(y_cal, cal_scores, max_fpr=0.15)
            cal_auroc = float(roc_auc_score(y_cal, cal_scores))
            improved = (cal_pauc, cal_auroc) > best_score
            if improved:
                best_score = (cal_pauc, cal_auroc)
                best_epoch = epoch
                no_improve = 0
                best_ref = self._save_ckpt(f"{arm}_best_ep{epoch:02d}", {"run_id": self.run_id, "arm": arm, "epoch": epoch, "model": model.state_dict()})
            else:
                no_improve += 1

            rec_entry = {
                "epoch": epoch, "lr": lr, "train_loss": loss_sum / max(seen, 1),
                "cal_pauc15": cal_pauc, "cal_auroc": cal_auroc, "seconds": time.monotonic() - t0,
            }
            history.append(rec_entry)
            prefix = f"{arm}_epoch{epoch:02d}"
            rec_ref = self._save_ckpt(f"{prefix}_recovery", {
                "run_id": self.run_id, "protocol_sha256": self.protocol_sha, "arm": arm, "epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                "best_score": list(best_score), "best_epoch": best_epoch, "best_ref": best_ref, "no_improve": no_improve,
            })
            hist_ref = self.save_csv(f"{prefix}_history", pd.DataFrame(history))
            self.save_json(prefix, {
                "run_id": self.run_id, "protocol_sha256": self.protocol_sha, "arm": arm, "epoch": epoch,
                "files": {"selected": best_ref, "recovery": rec_ref, "history": hist_ref},
            })
            print(f"  [{arm}] ep {epoch:02d}/{MAX_EPOCHS} | lr={lr:.1e} | loss={rec_entry['train_loss']:.4f} | "
                  f"cal_pAUC15={cal_pauc:.4f} | cal_AUROC={cal_auroc:.4f} | {rec_entry['seconds']:.1f}s {'*BEST*' if improved else ''}", flush=True)
            if no_improve >= PATIENCE:
                print(f"  [{arm}] Early stopping at epoch {epoch} (best epoch {best_epoch}).", flush=True)
                break

        done_payload = {"arm": arm, "best_epoch": best_epoch, "best_score": list(best_score), "best_ref": best_ref, "history": history}
        self.save_json(f"{arm}_complete", done_payload)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return done_payload

    def evaluate(self):
        torch = self.torch
        # Score validation (calibration + untouched 937 eval) and mimics (dev + test)
        val_df = self.roles.copy()
        old = self.source_predictions.query("cohort != 'mimics'").copy()
        old["label"] = old["label"].astype(int)
        val_df = val_df.merge(old[["image_id", "baseline"]], on="image_id", validate="one_to_one")
        val_df = val_df.rename(columns={"baseline": "retained"})

        mimic_df = self.mimic_eval_table[["image_id", "source", "lesion_id", "label", "split"]].copy()
        mimic_df = mimic_df.rename(columns={"label": "subtype"})

        cutoffs = {"retained": source_cutoff(val_df.loc[self.cal_mask, "label"], val_df.loc[self.cal_mask, "retained"])}
        for arm in ARMS:
            done = self.restore_json(f"{arm}_complete")
            if done is None:
                raise RuntimeError(f"Train {arm} before evaluate().")
            model = ConvNeXtDualHead.build(torch, self.timm, seed=SEED, pretrained=False).to(self.device)
            ckpt = torch.load(self.restore_file(done["best_ref"]), map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=True)
            val_df[arm] = score_dual_4tta(torch, model, np.asarray(self.images["val"]), self.device)
            mimic_df[arm] = score_dual_4tta(torch, model, np.asarray(self.images["mimic_eval"]), self.device)
            cutoffs[arm] = source_cutoff(val_df.loc[self.cal_mask, "label"], val_df.loc[self.cal_mask, arm])
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Also pull retained mimic scores if present in source_predictions
        old_mim = self.source_predictions.query("cohort == 'mimics'")[["image_id", "baseline"]].rename(columns={"baseline": "retained"})
        mimic_df = mimic_df.merge(old_mim, on="image_id", how="left")

        ev = val_df[self.eval_mask].copy()
        summary = []
        rows = []
        for arm in ("retained", *ARMS):
            cut = float(cutoffs[arm]["threshold"])
            for site, sub in [("pooled", ev), *ev.groupby("source", sort=True)]:
                m = metrics(sub["label"].to_numpy(), sub[arm].to_numpy(), cut)
                rows.append({"cohort": "mel_nv", "group": site, "arm": arm, "cutoff": cut, **m})
            for sp, msub in mimic_df.groupby("split", sort=True):
                if msub[arm].isna().any():
                    continue
                fa_rate = 100.0 * float((msub[arm] >= cut).mean())
                rows.append({"cohort": f"mimics_{sp}", "group": "pooled", "arm": arm, "cutoff": cut,
                             "Images": len(msub), "Benign false alarms (%)": fa_rate,
                             "Melanomas caught (%)": float("nan"), "AUROC": float("nan")})
                for st_name, st_sub in msub.groupby("subtype", sort=True):
                    rows.append({"cohort": f"mimics_{sp}", "group": st_name, "arm": arm, "cutoff": cut,
                                 "Images": len(st_sub), "Benign false alarms (%)": 100.0 * float((st_sub[arm] >= cut).mean()),
                                 "Melanomas caught (%)": float("nan"), "AUROC": float("nan")})

        summary = pd.DataFrame(rows)
        self.refs["val_predictions"] = self.save_csv("val_predictions", val_df)
        self.refs["mimic_predictions"] = self.save_csv("mimic_predictions", mimic_df)
        self.refs["cutoffs"] = self.save_json("locked_cutoffs", cutoffs)
        self.refs["summary"] = self.save_csv("summary", summary)
        self.refs["durable_save_receipt"] = self.save_json("durable_save_receipt", {
            "run_id": self.run_id, "protocol_sha256": self.protocol_sha, "files": self.refs,
        })
        display(summary[summary["group"].isin(["pooled", "HAM", "BCN", "BKL", "DF", "VASC"])].round(3))
        print(f"Verified Notebook 21 outputs saved on Drive: {self.drive.link}")
        return summary
