"""Matched ConvNeXt-224 resolution augmentation pilot with epoch-level Drive recovery."""
import os

import contextlib
import gc
import hashlib
import io
import json
import math
import platform
import shutil
import time
import uuid
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import PIL
import sklearn
from IPython.display import display
from PIL import Image, ImageFilter, ImageOps

from acquisition_stress import canonical_hash, correct_rgb, metrics, pixel_hash
from backbone_resolution_study import (
    BACKBONE_NAME, BATCH_SIZE, TRAIN_FOLDER, TRAIN_RECEIPT_SHA256, TRAIN_RUN,
    WEIGHT_DECAY, _to_imagenet_tensor, apply_epoch_augmentation_slice,
    build_convnext_model, download_file_curl, make_epoch_augmentation_plan,
    score_convnext_4tta, validate_split,
)
from inference_store import CheckpointDriveStore, FolderReader, digest
from step0_nasnet_features import degrade_native_image, locked_cutoff, resize_and_correct_shared


PARENT_RUN = "e8fb5a3e3a72d6fb723a"
PARENT_FOLDER = os.getenv("CONV_FUSION_PARENT_FOLDER", "")
PARENT_RECEIPT_SHA256 = "1e4df4b5f858b3938aabc6a8c29e32f73635f962c605a1dba52cc991d6c35038"
PARENT_MODEL_SHA256 = "d83ca73c0f9cea0cb1a868aa3c82759c6f63ea813c290acb37fd95fb4ec58b53"
PRIVATE_OUTPUT = "Melanoma Step 2 Resolution Augmentation"
TIMM_VERSION = "1.0.22"
SEED = 17
SIZE = 224
MICRO_BATCH = 16
RATES = (1e-5, 1e-5, 5e-6)
ARMS = ("baseline", "resolution")
STYLES = ("original", "lower_resolution", "held_out_box_blur", "contrast_low", "combined_low_contrast_low_res")
TRAIN_SIDES = (152, 168, 192)
TRAIN_BLUR = (0.55, 0.8, 1.10)
FPR_CAP = 0.10
BOOTSTRAP_DRAWS = 1000
METRIC_KEYS = (
    "Melanomas caught (%)", "Benign false alarms (%)",
    "Caught at up to 10% false alarms (%)", "AUROC",
)


def calibration_roles(validation):
    """Freeze a score-blind 40/60 partition of the already examined development set."""
    if validation["group"].duplicated().any():
        raise ValueError("The frozen validation set must have one photo per lesion group.")
    roles = validation[["image_id", "source", "lesion_id", "group", "label"]].copy()
    roles["role"] = "evaluation"
    for _, group in roles.groupby(["source", "label"], sort=True):
        if len(group) < 3:
            raise ValueError("Each source and diagnosis needs both calibration and evaluation rows.")
        ordered = sorted(group.index, key=lambda i: canonical_hash(["step2-calibration-v1", roles.at[i, "group"]]))
        roles.loc[ordered[:round(len(group) * 0.40)], "role"] = "calibration"
    return roles


def training_pair(native, image_id):
    """Deterministically choose moderate downsampling or Gaussian softening for one training view."""
    raw = native.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    selector = int(canonical_hash(["step2-resolution-v1", image_id])[:12], 16)
    if selector % 2 == 0:
        side = TRAIN_SIDES[(selector // 2) % len(TRAIN_SIDES)]
        modified = native.resize((side, side), Image.Resampling.BILINEAR)
        modified = modified.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        description = f"Downsampled to {side} pixels"
    else:
        radius = TRAIN_BLUR[(selector // 2) % len(TRAIN_BLUR)]
        modified = raw.filter(ImageFilter.GaussianBlur(radius))
        description = f"Gaussian blur, radius {radius:g}"
    return correct_rgb(np.asarray(raw)), correct_rgb(np.asarray(modified)), description


def evaluation_view(native, style, policy):
    if style == "held_out_box_blur":
        image = native.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return correct_rgb(np.asarray(image.filter(ImageFilter.BoxBlur(1.75))))
    return resize_and_correct_shared(degrade_native_image(native, style, policy), SIZE)


def paired_intervals(frame, baseline, candidate, baseline_cut, candidate_cut):
    """Paired lesion bootstrap, conditional on already fitted models and cutoffs."""
    labels = frame["label"].to_numpy(dtype=int)
    groups, group_index = np.unique(frame["group"], return_inverse=True)
    rng = np.random.default_rng(41)
    deltas = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))[group_index]
        if not weights[labels == 0].sum() or not weights[labels == 1].sum():
            continue
        a = metrics(labels, baseline, baseline_cut, weights=weights)
        b = metrics(labels, candidate, candidate_cut, weights=weights)
        deltas.append([b[name] - a[name] for name in METRIC_KEYS])
    if len(deltas) < 0.9 * BOOTSTRAP_DRAWS:
        raise RuntimeError("Too few paired bootstrap draws contained both diagnoses.")
    limits = np.quantile(deltas, [0.025, 0.975], axis=0)
    return {"draws": len(deltas), "groups": len(groups),
            "difference_95_percent": {key: limits[:, i].tolist() for i, key in enumerate(METRIC_KEYS)}}


def pilot_decision(summary):
    pooled = summary[summary["source"] == "pooled"].set_index(["style", "arm"])[list(METRIC_KEYS)]
    pairs = {style: pooled.loc[(style, "resolution")] - pooled.loc[(style, "baseline")]
             for style in ("original", "lower_resolution")}
    clean, stress = pairs["original"], pairs["lower_resolution"]
    checks = {
        "matched_cap_gain_at_least_3_points": bool(stress[METRIC_KEYS[2]] >= 3),
        "locked_sensitivity_improved_on_112px": bool(stress[METRIC_KEYS[0]] > 0),
        "locked_false_alarms_reduced_on_112px": bool(stress[METRIC_KEYS[1]] < 0),
        "clear_sensitivity_loss_at_most_2_points": bool(clean[METRIC_KEYS[0]] >= -2),
        "clear_false_alarm_increase_at_most_1_point": bool(clean[METRIC_KEYS[1]] <= 1),
    }
    return {"screen_passed": all(checks.values()), "checks": checks,
            "candidate_112px_meets_10_percent_false_alarm_cap": bool(pooled.loc[
                ("lower_resolution", "resolution"), METRIC_KEYS[1]] <= 10),
            "candidate_clear_meets_10_percent_false_alarm_cap": bool(pooled.loc[
                ("original", "resolution"), METRIC_KEYS[1]] <= 10),
            "scope": "Single-seed development screen on simulated degradation. Review all sources and conditions before advancing."}


class ResolutionPilot:
    """Train and evaluate just two matched arms. Each Drive state commits one epoch."""

    trained_arms = ARMS
    second_arm = "resolution"
    output_folder = PRIVATE_OUTPUT
    pair = staticmethod(training_pair)
    preview_title = "Simulated lower detail"

    @property
    def second_cache(self):
        return "train_" + self.second_arm

    def configure_comparison(self, service):
        """An extension can bind a previously verified comparison before hashing."""

    def __init__(self, source_bundle, work="/content/melanoma-step2", *, service=None):
        import timm
        import torch

        if timm.__version__ != TIMM_VERSION:
            raise RuntimeError(f"This bundle requires timm {TIMM_VERSION}. Run the setup cell first.")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Select a BF16-capable GPU, such as the Colab L4, before connecting.")
        if service is None:
            from google.colab import auth
            from googleapiclient.discovery import build

            auth.authenticate_user()
            service = build("drive", "v3", cache_discovery=False)

        self.torch, self.timm = torch, timm
        self.work = Path(work)
        self.parent_reader = FolderReader(service, PARENT_FOLDER, self.work / "notebook10")
        reader = FolderReader(service, TRAIN_FOLDER, self.work / "notebook06")
        parent = self._source_receipt(self.parent_reader, PARENT_RUN, "durable_save_receipt", PARENT_RECEIPT_SHA256)
        training = self._source_receipt(reader, TRAIN_RUN, "save_receipt", TRAIN_RECEIPT_SHA256)
        self.parent_model = parent["files"]["convnext_224_weights"]
        if self.parent_model["sha256"] != PARENT_MODEL_SHA256:
            raise RuntimeError("The parent receipt does not name the locked ConvNeXt-224 checkpoint.")
        parent_protocol = json.loads(self.parent_reader.fetch_artifact(parent["files"]["protocol"]).read_bytes())["protocol"]
        if canonical_hash(parent_protocol) != parent["protocol_sha256"]:
            raise RuntimeError("Notebook 10 protocol hash differs from its final receipt.")
        validation_report = json.loads(reader.fetch_artifact(training["validation_report"]).read_bytes())
        self.sources = validation_report["protocol"]["image_sources"]
        self.class_weights = np.asarray(parent_protocol["training"]["class_weights"], dtype=np.float32)
        self.policy = parent_protocol["stress_policy"]
        self.parent_cutoff = json.loads(self.parent_reader.fetch_artifact(parent["files"]["convnext_224_selection"]).read_bytes())["locked_cutoff"]
        split_ref = next(ref for ref in training["files"] if ref["name"].endswith("__pooled_ham_bcn_split.csv"))
        split = validate_split(pd.read_csv(reader.fetch_artifact(split_ref), dtype=str, keep_default_na=False))
        self.train_manifest, self.validation = (
            split.loc[split["split"] == name].sort_values(["source", "image_id"]).reset_index(drop=True)
            for name in ("train", "validation")
        )
        self.roles = calibration_roles(self.validation)
        saved = pd.read_csv(self.parent_reader.fetch_artifact(parent["files"]["convnext_224_val_scores"]),
                            converters={"raw_score": float}, dtype={"image_id": str, "group": str})
        aligned = saved.set_index("image_id", verify_integrity=True).loc[self.roles["image_id"]]
        if len(saved) != len(self.roles) or aligned["label"].tolist() != self.roles["label"].tolist():
            raise RuntimeError("Saved Notebook 10 predictions differ from the frozen validation identities.")
        self.parent_scores = aligned["raw_score"].to_numpy()
        self.bundle = Path(source_bundle)
        self.protocol = {
            "task": "ConvNeXt-224 matched seed-17 acquisition augmentation pilot",
            "source_bundle_sha256": digest(self.bundle), "parent_protocol_sha256": parent["protocol_sha256"],
            "parent_checkpoint_sha256": PARENT_MODEL_SHA256, "split_sha256": split_ref["sha256"],
            "validation_roles_sha256": canonical_hash(self.roles.to_dict("records")),
            "seed": SEED, "backbone": BACKBONE_NAME, "batch": BATCH_SIZE, "micro_batch": MICRO_BATCH,
            "lr_per_epoch": list(RATES), "selection": "final epoch, no early stopping",
            "optimizer": "fresh AdamW in both arms", "weight_decay": WEIGHT_DECAY,
            "class_weights": self.class_weights.tolist(), "fpr_cap": FPR_CAP,
            "views_per_training_image": 2, "train_sides": list(TRAIN_SIDES),
            "train_gaussian_radii": list(TRAIN_BLUR), "fixed_second_view_per_image": True,
            "held_out_box_radius": 1.75, "evaluation": list(STYLES), "stress_policy": self.policy,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                         "pillow": PIL.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)},
        }
        self.configure_comparison(service)
        self.protocol_sha = canonical_hash(self.protocol)
        self.run_id = self.protocol_sha[:20]
        self.root = self.work / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0")
        self.drive = CheckpointDriveStore(service, self.output_folder)
        completed = sum(self.latest_epoch(arm)["epoch"] for arm in self.trained_arms)
        remaining = len(self.trained_arms) * len(RATES) - completed
        self.drive.preflight(required_bytes=remaining * 4 * self.parent_model["bytes"] + 256 * 1024**2)
        self.save_json("protocol", {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "protocol": self.protocol})
        self.publish(self.bundle)
        self.save_csv("validation_roles", self.roles)
        print(f"Run {self.run_id}. Private Drive folder {self.drive.link}.")
        display(self.roles.groupby(["source", "role", "label"]).size().unstack("label"))

    @staticmethod
    def _source_receipt(reader, run, stem, expected_sha):
        path = reader.fetch(f"{run}__{stem}.json")
        value = json.loads(path.read_bytes())
        if digest(path) != expected_sha or value["run_id"] != run:
            raise RuntimeError(f"Source run {run} no longer matches its frozen receipt.")
        return value

    def publish(self, path):
        ref = self.drive.publish(path, f"{self.run_id}__{path.name}")
        self.drive.verify_receipt(ref)
        return ref

    def save_json(self, name, value):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return self.publish(path)

    def save_csv(self, name, frame):
        path = self.root / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        return self.publish(path)

    def restore_json(self, name):
        path = self.root / f"{name}.json"
        if not self.drive.restore(f"{self.run_id}__{path.name}", path):
            return None
        return json.loads(path.read_bytes())

    def latest_epoch(self, arm):
        if arm not in self.trained_arms:
            raise ValueError(f"Unknown pilot arm {arm}.")
        for epoch in range(len(RATES), 0, -1):
            state = self.restore_json(f"{arm}_epoch{epoch:02d}")
            if state is None:
                continue
            if (state["run_id"], state["protocol_sha256"], state["arm"], state["epoch"]) != (
                    self.run_id, self.protocol_sha, arm, epoch):
                raise RuntimeError("The Drive checkpoint boundary belongs to another run.")
            for role in ("selected", "recovery", "history"):
                self.drive.verify_receipt(state["files"][role])
            return state
        return {"epoch": 0}

    def restore_file(self, ref):
        prefix = self.run_id + "__"
        if not ref["name"].startswith(prefix):
            raise RuntimeError("Output reference belongs to another run.")
        path = self.root / ref["name"].removeprefix(prefix)
        self.drive.verify_receipt(ref)
        if not self.drive.restore(ref["name"], path) or digest(path) != ref["sha256"]:
            raise RuntimeError("Drive recovery bytes do not match their receipt.")
        return path

    def prepare_images(self):
        paths = {key: self.root / f"{key}.npy" for key in ("train_clear", self.second_cache, *STYLES)}
        receipt_path = self.root / "image_cache.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if all(path.is_file() and digest(path) == receipt.get(key) for key, path in paths.items()):
                print("Verified cached images for this run.")
                self.images = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
                return
        archives = self.work / "source_archives"
        archives.mkdir(parents=True, exist_ok=True)
        remaining = sum(spec["bytes"] for source, spec in self.sources.items()
                        if source in ("HAM", "BCN") and not (archives / f"{source}.zip").is_file())
        cache_bytes = (2 * len(self.train_manifest) + len(STYLES) * len(self.validation)) * SIZE * SIZE * 3
        if shutil.disk_usage(self.work).free < remaining + cache_bytes + 6 * 1024**3:
            raise RuntimeError("Insufficient Colab disk space for verified archives, caches and checkpoints.")
        with contextlib.ExitStack() as stack:
            zips = {}
            for key in ("HAM", "BCN"):
                spec = self.sources[key]
                path = download_file_curl(archives / f"{key}.zip", spec["url"], expected_bytes=spec["bytes"],
                                          expected_md5=spec.get("md5"))
                zips[key] = stack.enter_context(ZipFile(path))
            self._build_cache(zips, self.train_manifest, paths, ("train_clear", self.second_cache))
            self._build_cache(zips, self.validation, paths, STYLES)
        receipt_path.write_text(json.dumps({key: digest(path) for key, path in paths.items()}), encoding="utf-8")
        self.images = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}

    def _build_cache(self, zips, manifest, paths, keys):
        partial = {key: paths[key].with_suffix(".part.npy") for key in keys}
        matrices = {key: np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                    shape=(len(manifest), SIZE, SIZE, 3)) for key, path in partial.items()}
        is_training = keys[0] == "train_clear"
        for i, row in enumerate(manifest.itertuples(index=False)):
            with Image.open(io.BytesIO(zips[row.source].read(row.member))) as photo:
                native = ImageOps.exif_transpose(photo).convert("RGB")
            if pixel_hash(native) != row.pixel_sha256:
                raise RuntimeError(f"Frozen pixels changed for {row.image_id}.")
            if is_training:
                matrices[keys[0]][i], matrices[keys[1]][i], _ = self.pair(native, row.image_id)
            else:
                for key in keys:
                    matrices[key][i] = evaluation_view(native, key, self.policy)
            if (i + 1) % 1000 == 0 or i + 1 == len(manifest):
                print(f"Verified {i + 1:,}/{len(manifest):,} {'training' if is_training else 'validation'} photos.", flush=True)
        for key in keys:
            matrices[key].flush()
            del matrices[key]
            partial[key].replace(paths[key])

    def preview(self):
        from matplotlib import pyplot as plt

        chosen = []
        for _, group in self.train_manifest.groupby(["source", "label"], sort=True):
            chosen.extend(sorted(group.index, key=lambda i: canonical_hash(["step2-preview", group.at[i, "image_id"]]))[:2])
        fig, axes = plt.subplots(len(chosen), 2, figsize=(5, 2 * len(chosen)), squeeze=False, layout="constrained")
        for row, i in enumerate(chosen):
            for j, key in enumerate(("train_clear", self.second_cache)):
                axes[row, j].imshow(self.images[key][i])
                axes[row, j].axis("off")
            axes[row, 0].set_title(self.train_manifest.at[i, "image_id"])
        axes[0, 1].set_title(self.preview_title)
        plt.show()
        plt.close(fig)

    def new_model(self):
        return build_convnext_model(self.torch, self.timm, seed=SEED, pretrained=False).to(self.device)

    def load_parent(self, model):
        source = self.parent_reader.fetch_artifact(self.parent_model)
        payload = self.torch.load(source, map_location="cpu", weights_only=True)
        if payload["image_size"] != SIZE or payload["backbone"] != BACKBONE_NAME:
            raise RuntimeError("Starting checkpoint has a different resolution or backbone.")
        model.load_state_dict(payload["model"], strict=True)

    def optimizer(self, model):
        return self.torch.optim.AdamW(model.parameters(), lr=RATES[0], weight_decay=WEIGHT_DECAY)

    def _score(self, model, pixels):
        score = score_convnext_4tta(self.torch, model, pixels, self.device, batch_size=BATCH_SIZE)
        if len(score) != len(pixels) or not np.isfinite(score).all():
            raise RuntimeError("Inference returned an invalid score vector.")
        return score

    def training_caches(self, arm):
        return "train_clear", self.second_cache if arm == self.second_arm else "train_clear"

    def _update(self, model, optimizer, arm, plan, second_plan, start, end):
        torch = self.torch
        optimizer.zero_grad(set_to_none=True)
        contribution = 0.
        labels = self.train_manifest["label"].to_numpy()
        caches = self.training_caches(arm)
        for pos in range(start, end, MICRO_BATCH):
            stop = min(pos + MICRO_BATCH, end)
            ids = plan["order"][pos:stop]
            y = torch.as_tensor(labels[ids], dtype=torch.float32, device=self.device)
            weights = torch.as_tensor(self.class_weights[labels[ids]], device=self.device)
            for cache, draw in zip(caches, (plan, second_plan), strict=True):
                pixels = apply_epoch_augmentation_slice(self.images[cache][ids], draw, pos, stop)
                x = _to_imagenet_tensor(torch, pixels).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    scores = model(x).reshape(-1)
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                        scores, y, reduction="none") * weights).sum() / (2 * (end - start))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss. The current epoch will be repeated from the last verified boundary.")
                loss.backward()
                contribution += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return contribution

    def _save_checkpoint(self, prefix, value):
        staging = self.root / f"{prefix}-{uuid.uuid4().hex}.part"
        with staging.open("xb") as stream:
            self.torch.save(value, stream)
        path = self.root / f"{prefix}-{digest(staging)[:20]}.pt"
        staging.replace(path)
        return self.publish(path)

    def preflight(self):
        """Verify the parent replay and run one disposable GPU update plus Drive roundtrip."""
        existing = self.restore_json("preflight")
        if existing:
            if existing["protocol_sha256"] != self.protocol_sha:
                raise RuntimeError("Preflight belongs to another protocol.")
            self.drive.verify_receipt(existing["probe"])
            print("Current-protocol preflight already verified.")
            return
        model = self.new_model()
        self.load_parent(model)
        actual = self._score(model, self.images["original"])
        delta = np.max(np.abs(actual - self.parent_scores))
        changed = (actual >= self.parent_cutoff) != (self.parent_scores >= self.parent_cutoff)
        if delta > 0.01 or np.any(np.abs(self.parent_scores[changed] - self.parent_cutoff) > 0.001):
            raise RuntimeError(f"Starting checkpoint replay drifted. Maximum raw score difference {delta:.4g}.")
        plan = make_epoch_augmentation_plan(len(self.train_manifest), SEED, 1)
        opt = self.optimizer(model)
        self.torch.cuda.synchronize()
        start = time.monotonic()
        model.train()
        loss = self._update(model, opt, self.second_arm, plan, plan, 0, BATCH_SIZE)
        self.torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        payload = {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "format": 1}
        probe = self._save_checkpoint("restore_probe", payload)
        fresh = self.root / "restore_probe_fresh.pt"
        if not self.drive.restore(probe["name"], fresh) or digest(fresh) != probe["sha256"]:
            raise RuntimeError("Private Drive checkpoint restore readback failed.")
        if self.torch.load(fresh, map_location="cpu", weights_only=True) != payload:
            raise RuntimeError("Restored checkpoint probe could not be read.")
        self.save_json("preflight", {**payload, "max_parent_raw_score_delta": float(delta),
                      "near_cutoff_decisions_changed": int(changed.sum()), "probe": probe,
                      "one_update_seconds": elapsed, "one_update_loss": loss})
        del model, opt
        gc.collect()
        self.torch.cuda.empty_cache()
        print(f"Parent replay and private Drive restore passed. One paired update took {elapsed:.1f}s.")

    def train(self, arm):
        preflight = self.restore_json("preflight")
        if preflight is None or preflight["protocol_sha256"] != self.protocol_sha:
            raise RuntimeError("Run the GPU and Drive preflight before training.")
        self.drive.verify_receipt(preflight["probe"])
        state = self.latest_epoch(arm)
        if state["epoch"] == len(RATES):
            print(f"{arm}: all epochs already saved and verified.")
            return
        torch = self.torch
        model = self.new_model()
        optimizer = self.optimizer(model)
        history = []
        if state["epoch"]:
            recovery = torch.load(self.restore_file(state["files"]["recovery"]), map_location="cpu", weights_only=True)
            if (recovery["run_id"], recovery["protocol_sha256"], recovery["arm"], recovery["epoch"]) != (
                    self.run_id, self.protocol_sha, arm, state["epoch"]):
                raise RuntimeError("The recovery payload belongs to another epoch or protocol.")
            model.load_state_dict(recovery["model"], strict=True)
            optimizer.load_state_dict(recovery["optimizer"])
            history = list(recovery["history"])
            if len(history) != state["epoch"]:
                raise RuntimeError("Recovery history is incomplete.")
        else:
            self.load_parent(model)
        for epoch in range(state["epoch"] + 1, len(RATES) + 1):
            self.drive.quota(4 * self.parent_model["bytes"] + 8 * 1024**2)
            torch.manual_seed(SEED + epoch)
            torch.cuda.manual_seed_all(SEED + epoch)
            for group in optimizer.param_groups:
                group["lr"] = RATES[epoch - 1]
            plan = make_epoch_augmentation_plan(len(self.train_manifest), SEED, epoch)
            second_plan = make_epoch_augmentation_plan(len(self.train_manifest), SEED + 1000, epoch)
            model.train()
            total_loss = 0.
            steps = 0
            for start in range(0, len(self.train_manifest), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(self.train_manifest))
                total_loss += (end - start) * self._update(model, optimizer, arm, plan, second_plan, start, end)
                steps += 1
                if steps % 40 == 0:
                    print(f"{arm}, epoch {epoch}: {steps} updates", flush=True)
            assert steps == math.ceil(len(self.train_manifest) / BATCH_SIZE)
            history.append({"epoch": epoch, "steps": steps, "lr": RATES[epoch - 1],
                            "weighted_loss": total_loss / len(self.train_manifest)})
            boundary = {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "arm": arm, "epoch": epoch}
            prefix = f"{arm}_epoch{epoch:02d}"
            files = {
                "selected": self._save_checkpoint(prefix + "_model", {**boundary, "model": model.state_dict()}),
                "recovery": self._save_checkpoint(prefix + "_recovery", {**boundary, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "history": history, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state()}),
                "history": self.save_csv(prefix + "_history", pd.DataFrame(history)),
            }
            record = {**boundary, "files": files}
            ref = self.save_json(prefix, record)
            self.drive.verify_receipt(ref)
            if self.restore_json(prefix) != record:
                raise RuntimeError("Epoch record failed Drive readback.")
            print(f"{arm} epoch {epoch}/{len(RATES)}. {steps} updates. Drive recovery verified.", flush=True)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self):
        existing = self.restore_json("evaluation_receipt")
        if existing:
            if existing["protocol_sha256"] != self.protocol_sha:
                raise RuntimeError("Evaluation receipt belongs to another protocol.")
            for role in ("predictions", "summary", "intervals", "decision"):
                self.drive.verify_receipt(existing["files"][role])
            summary = pd.read_csv(self.restore_file(existing["files"]["summary"]))
            display(summary[summary["source"] == "pooled"].round(3))
            print(pilot_decision(summary))
            return
        all_predictions = []
        results = []
        cutoffs = {}
        calib = self.roles["role"].eq("calibration").to_numpy()
        evaluation = ~calib
        labels = self.roles["label"].to_numpy(dtype=int)
        for arm in ARMS:
            state = self.latest_epoch(arm)
            if state["epoch"] != len(RATES):
                raise RuntimeError("Both arms must reach the same final epoch before evaluation.")
            path = self.restore_file(state["files"]["selected"])
            payload = self.torch.load(path, map_location="cpu", weights_only=True)
            if (payload["protocol_sha256"], payload["arm"], payload["epoch"]) != (
                    self.protocol_sha, arm, len(RATES)):
                raise RuntimeError("Final model provenance does not match the pilot.")
            model = self.new_model()
            model.load_state_dict(payload["model"], strict=True)
            clear = self._score(model, self.images["original"])
            cutoff = locked_cutoff(labels[calib], clear[calib], FPR_CAP)
            cutoffs[arm] = {"threshold": cutoff, "calibration_cases": int(calib.sum()),
                            "benign_false_alarms": int(((labels[calib] == 0) & (clear[calib] >= cutoff)).sum())}
            self.save_json(arm + "_locked_cutoff", cutoffs[arm])
            for style in STYLES:
                score = clear if style == "original" else self._score(model, self.images[style][evaluation])
                frame = self.roles.copy() if style == "original" else self.roles[evaluation].copy()
                frame["score"] = score
                frame["arm"], frame["style"] = arm, style
                all_predictions.append(frame)
                test = frame[frame["role"] == "evaluation"]
                groups = [("pooled", test), *test.groupby("source", sort=True)]
                for source, rows in groups:
                    results.append({"arm": arm, "style": style, "source": source,
                                    **metrics(rows["label"], rows["score"], cutoff)})
            del model, payload
            gc.collect()
            self.torch.cuda.empty_cache()
        predictions = pd.concat(all_predictions, ignore_index=True)
        summary = pd.DataFrame(results)
        intervals = {}
        for style in STYLES:
            eligible = predictions[(predictions["style"] == style) & (predictions["role"] == "evaluation")]
            a, b = (eligible[eligible["arm"] == arm] for arm in ARMS)
            if a["image_id"].tolist() != b["image_id"].tolist():
                raise RuntimeError("Pilot predictions cannot be paired by image.")
            intervals[style] = paired_intervals(a, a["score"].to_numpy(), b["score"].to_numpy(),
                                                cutoffs["baseline"]["threshold"], cutoffs["resolution"]["threshold"])
        reports = {"predictions": self.save_csv("predictions", predictions),
                   "summary": self.save_csv("summary", summary),
                   "intervals": self.save_json("paired_intervals", intervals),
                   "decision": self.save_json("decision", pilot_decision(summary))}
        self.save_json("evaluation_receipt", {"protocol_sha256": self.protocol_sha, "files": reports})
        display(summary[summary["source"] == "pooled"].round(3))
        print(pilot_decision(summary))

    def verify_saved(self):
        """Verify the current run's complete set of durable outputs and read back its receipt."""
        refs = {}
        for name in ("protocol.json", self.bundle.name, "validation_roles.csv", "preflight.json",
                     "baseline_locked_cutoff.json", self.second_arm + "_locked_cutoff.json", "evaluation_receipt.json"):
            path = self.root / name
            if not self.drive.restore(f"{self.run_id}__{name}", path):
                raise RuntimeError(f"Required Drive artifact is missing: {name}.")
            refs[name] = self.publish(path)
        preflight = self.restore_json("preflight")
        evaluation = self.restore_json("evaluation_receipt")
        if preflight["protocol_sha256"] != self.protocol_sha or evaluation["protocol_sha256"] != self.protocol_sha:
            raise RuntimeError("One stage receipt belongs to another protocol.")
        refs["restore_probe"] = preflight["probe"]
        refs.update(evaluation["files"])
        for arm in self.trained_arms:
            for epoch in range(1, len(RATES) + 1):
                name = f"{arm}_epoch{epoch:02d}"
                state = self.restore_json(name)
                if state is None or (state["run_id"], state["protocol_sha256"], state["arm"], state["epoch"]) != (
                        self.run_id, self.protocol_sha, arm, epoch):
                    raise RuntimeError(f"Missing or mismatched epoch record: {name}.")
                refs[name] = self.publish(self.root / f"{name}.json")
                for role in ("selected", "recovery", "history"):
                    refs[f"{name}_{role}"] = state["files"][role]
        refs.update(self.additional_receipts())
        for ref in refs.values():
            self.drive.verify_receipt(ref)
        self.parent_reader.verify_artifact(self.parent_model)
        value = {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "parent_model": self.parent_model,
                 "files": refs, "scope": "Completed single-seed development pilot. Independent confirmation is pending."}
        receipt = self.save_json("durable_save_receipt", value)
        fresh = self.root / "durable_save_receipt_readback.json"
        if not self.drive.restore(receipt["name"], fresh) or digest(fresh) != receipt["sha256"] or json.loads(fresh.read_bytes()) != value:
            raise RuntimeError("Current final receipt failed readback.")
        print(f"Verified {len(refs)} artifacts and final receipt on private Drive. {self.drive.link}")
        return value

    def additional_receipts(self):
        return {}
