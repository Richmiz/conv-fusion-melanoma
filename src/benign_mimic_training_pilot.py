"""Matched ConvNeXt-224 fine-tuning with audited benign mimics."""
import os

import contextlib
import json
import math
import platform
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import PIL
from IPython.display import display
from PIL import Image, ImageOps

from acquisition_stress import canonical_hash, correct_rgb, pixel_hash
from backbone_resolution_study import BACKBONE_NAME, BATCH_SIZE, WEIGHT_DECAY, download_file_curl, validate_split
from inference_store import CheckpointDriveStore, FolderReader, digest
from isic_checkpoint_comparison import RUNS, frozen_receipt
from resolution_augmentation_pilot import MICRO_BATCH, RATES, SEED, SIZE, ResolutionPilot, calibration_roles


ARMS = ("baseline", "benign_mimics")
PREFLIGHT = ("2bc7b2afbb107d30e099", os.getenv("CONV_FUSION_PREFLIGHT_FOLDER", ""),
             "75275541442b5164f90c78fee3ac8116517bfbe60dd7959c6fe36756a9f948f1")
SPLIT_SHA = "fc546904c9b77e1d8cebc8d7b2e87a315494eb12d7b47388ca909c775bab5168"
MIMIC_SHA = "9ceb931010cb9ab6b1c719b59c0362b8ca6770c8580a9081ad87c669f9f2f701"
MIMIC_FRACTION = 0.25
NEGATIVE_CALIBRATION_ALLOWANCE = 45
RULE = {"require_fewer_false_alarms_on_held_out_mimics": True,
        "max_melanoma_sensitivity_loss_pp": 2, "max_nevus_false_alarm_increase_pp": 1}
REPORTS = {"predictions", "summary", "paired_intervals", "decision"}


def replacement_schedule(source_training, mimics):
    """Assign fixed source-matched benign slots; cycle through every eligible lesion."""
    eligible = mimics[mimics.split == "train"]
    mapping = []
    for source, group in source_training[source_training.label == 0].groupby("source", sort=True):
        lesions = eligible[eligible.source == source].copy()
        if lesions.empty:
            raise RuntimeError(f"No clean training mimics for {source}.")
        pool = sorted(lesions.to_dict("records"),
                      key=lambda r: canonical_hash(["mimic-exemplar-v1", source, r["lesion_id"]]))
        count = round(len(group) * MIMIC_FRACTION)
        slots = sorted(group.index, key=lambda i: canonical_hash(
            ["mimic-slot-v1", source, source_training.at[i, "image_id"]]))[:count]
        for n, slot in enumerate(slots):
            mimic = pool[n % len(pool)]
            mapping.append({"slot": int(slot), "source": source, "original_image_id": source_training.at[slot, "image_id"],
                            "mimic_image_id": mimic["image_id"], "mimic_lesion_id": mimic["lesion_id"],
                            "subtype": mimic["label"]})
    return pd.DataFrame(mapping).sort_values("slot").reset_index(drop=True)


def source_cutoff(labels, scores):
    benign = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if len(benign) != 534 or not np.isfinite(benign).all():
        raise RuntimeError("Expected the same 534 benign calibration photos per arm.")
    # The next representable score also handles ties at the allowable boundary.
    cutoff = float(np.nextafter(benign[NEGATIVE_CALIBRATION_ALLOWANCE], np.inf))
    return {"threshold": cutoff, "calibration_cases": len(labels),
            "benign_false_alarms": int(np.sum(benign >= cutoff))}


def summarize(predictions, cutoffs):
    results, intervals = [], {}
    eligible = predictions[predictions.role == "evaluation"]
    for cohort, data in eligible.groupby("cohort", sort=True):
        groups = [("pooled", data), *[(f"source/{name}", part) for name, part in data.groupby("source")]]
        if cohort == "mimics":
            groups += [(f"subtype/{name}", part) for name, part in data.groupby("label")]
        for group, part in groups:
            positive = []
            for arm in ARMS:
                hits = part[arm].to_numpy() >= cutoffs[arm]["threshold"]
                positive.append(hits)
                results.append({"cohort": cohort, "group": group, "arm": arm, "cases": len(part),
                                "flagged": int(hits.sum()), "flagged_pct": 100 * float(hits.mean()),
                                "measure": "melanoma sensitivity" if cohort == "melanomas" else "benign false alarms"})
            diff = positive[1].astype(int) - positive[0].astype(int)
            rng = np.random.default_rng(41)
            draws = rng.choice(diff, size=(1000, len(diff)), replace=True).mean(axis=1) * 100
            intervals[f"{cohort}/{group}"] = {"lesions": len(part), "new_flags": int(np.sum(diff == 1)),
                                               "flags_removed": int(np.sum(diff == -1)),
                                               "change_pp": float(100 * diff.mean()),
                                               "paired_95_interval_pp": np.quantile(draws, [0.025, 0.975]).tolist()}
    return pd.DataFrame(results), intervals


def decision(summary):
    pooled = summary[summary.group == "pooled"].pivot(index="cohort", columns="arm", values="flagged_pct")
    change = pooled["benign_mimics"] - pooled["baseline"]
    checks = {"mimic_false_alarms_reduced": bool(change["mimics"] < 0),
              "melanoma_sensitivity_preserved": bool(change["melanomas"] >= -RULE["max_melanoma_sensitivity_loss_pp"]),
              "nevus_false_alarms_preserved": bool(change["nevi"] <= RULE["max_nevus_false_alarm_increase_pp"])}
    return {"screen_passed": all(checks.values()), "checks": checks, "rule": RULE,
            "scope": "Single-seed lesion-disjoint development pilot. Confirm on an independent cohort before adoption."}


class BenignMimicTrainingPilot(ResolutionPilot):
    trained_arms = ARMS
    second_arm = "benign_mimics"
    output_folder = "Melanoma Matched Benign Mimic Training"

    def __init__(self, bundle, work="/content/benign-mimic-training", *, service=None):
        import timm
        import torch
        if timm.__version__ != "1.0.22" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the setup cell and an L4 or other BF16-capable GPU.")
        if service is None:
            from google.colab import auth
            from googleapiclient.discovery import build

            auth.authenticate_user()
            service = build("drive", "v3", cache_discovery=False)
        self.torch, self.timm = torch, timm
        self.device = torch.device("cuda:0")
        self.work, self.bundle = Path(work), Path(bundle)
        base_run, folder, sha = RUNS["baseline"]
        self.parent_reader = FolderReader(service, folder, self.work / "notebook11")
        base = frozen_receipt(self.parent_reader, base_run, sha)
        self.parent_model = base["files"]["baseline_epoch03_selected"]
        self.parent_protocol_sha = base["protocol_sha256"]
        self.parent_reader.verify_artifact(self.parent_model)
        base_protocol = json.loads(self.parent_reader.fetch_artifact(base["files"]["protocol.json"]).read_bytes())["protocol"]
        if canonical_hash(base_protocol) != self.parent_protocol_sha:
            raise RuntimeError("The Notebook 11 protocol differs from its verified receipt.")
        self.class_weights = np.asarray(base_protocol["class_weights"], dtype=np.float32)
        self.parent_cutoff = json.loads(self.parent_reader.fetch_artifact(base["files"]["baseline_locked_cutoff.json"]).read_bytes())["threshold"]
        source = Path(__file__).resolve().parents[1] / "data/results/3c9cf6562d30585daf4b/pooled_ham_bcn_split.csv"
        if digest(source) != SPLIT_SHA or base_protocol["split_sha256"] != SPLIT_SHA:
            raise RuntimeError("The original melanoma and nevus manifest differs from the frozen source.")
        split = validate_split(pd.read_csv(source, dtype=str, keep_default_na=False))
        self.train_manifest, self.validation = (
            split.loc[split.split == name].sort_values(["source", "image_id"]).reset_index(drop=True)
            for name in ("train", "validation")
        )
        self.roles = calibration_roles(self.validation)
        archived_roles = pd.read_csv(self.parent_reader.fetch_artifact(base["files"]["validation_roles.csv"]), keep_default_na=False)
        pd.testing.assert_frame_equal(self.roles, archived_roles)
        previous = pd.read_csv(self.parent_reader.fetch_artifact(base["files"]["predictions"]), converters={"score": float})
        previous = previous[(previous.arm == "baseline") & (previous["style"] == "original")]
        pd.testing.assert_frame_equal(previous[self.roles.columns].reset_index(drop=True), self.roles)
        self.parent_scores = previous.score.to_numpy()

        preflight_reader = FolderReader(service, PREFLIGHT[1], self.work / "notebook14")
        audit = frozen_receipt(preflight_reader, PREFLIGHT[0], PREFLIGHT[2])
        if audit["baseline"] != self.parent_model or not audit["train_pilot"]:
            raise RuntimeError("The verified preflight must pass for this starting checkpoint.")
        audited_split = preflight_reader.fetch_artifact(audit["files"]["mimic_split.csv"])
        readiness = json.loads(preflight_reader.fetch_artifact(audit["files"]["readiness.json"]).read_bytes())
        if digest(audited_split) != MIMIC_SHA or readiness["split_sha256"] != MIMIC_SHA or not readiness["train_pilot"]:
            raise RuntimeError("The audited mimic assignments or baseline screen differ.")
        audited_protocol = json.loads(preflight_reader.fetch_artifact(audit["files"]["protocol"]).read_bytes())
        if canonical_hash(audited_protocol) != audit["protocol_sha256"]:
            raise RuntimeError("The audit protocol differs from its verified receipt.")
        self.sources = audited_protocol["archives"]
        self.mimic_split = pd.read_csv(audited_split, dtype=str, keep_default_na=False)
        self.mimic_train = self.mimic_split[self.mimic_split.split == "train"].reset_index(drop=True)
        self.mimic_test = self.mimic_split[self.mimic_split.split == "test"].reset_index(drop=True)
        if self.mimic_split.split.value_counts().to_dict() != {"train": 577, "development": 188, "test": 188}:
            raise RuntimeError("Unexpected clean mimic cohort counts.")
        if self.mimic_split.duplicated(["source", "lesion_id"]).any() or set(self.mimic_split.pixel_sha256) & set(split.pixel_sha256):
            raise RuntimeError("The mimic extension mixes lesions or pixels across the source split.")
        self.replacements = replacement_schedule(self.train_manifest, self.mimic_split)
        if math.ceil(len(self.train_manifest) / BATCH_SIZE) != 241:
            raise RuntimeError("Training must stay within the 723-update schedule.")

        self.protocol = {
            "task": "Matched benign mimic fine-tuning", "backbone": BACKBONE_NAME, "source_bundle_sha256": digest(bundle),
            "parent_model": self.parent_model, "parent_protocol_sha256": self.parent_protocol_sha, "source_split_sha256": SPLIT_SHA,
            "mimic_split_sha256": MIMIC_SHA, "audit_receipt_sha256": PREFLIGHT[2],
            "replacement_schedule_sha256": canonical_hash(self.replacements.to_dict("records")),
            "mimic_fraction": MIMIC_FRACTION, "allocation": "Fixed source-matched negative slots; round to nearest count per source",
            "seed": SEED, "batch": BATCH_SIZE, "micro_batch": MICRO_BATCH, "updates_per_arm": 723,
            "optimizer": "fresh AdamW", "rates": list(RATES), "weight_decay": WEIGHT_DECAY,
            "class_weights": self.class_weights.tolist(), "final_epoch": len(RATES),
            "validation_roles_sha256": canonical_hash(self.roles.to_dict("records")),
            "calibration": {"benign_cases": 534, "allowed_false_alarms": NEGATIVE_CALIBRATION_ALLOWANCE},
            "screen": RULE,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__), "timm": timm.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__,
                         "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        }
        self.protocol_sha = canonical_hash(self.protocol)
        self.run_id = self.protocol_sha[:20]
        self.root = self.work / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.drive = CheckpointDriveStore(service, self.output_folder)
        completed = sum(self.latest_epoch(arm)["epoch"] for arm in ARMS)
        remaining = 2 * len(RATES) - completed
        self.drive.preflight(required_bytes=remaining * 4 * self.parent_model["bytes"] + 256 * 1024**2)
        self.save_json("protocol", {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "protocol": self.protocol})
        self.drive.verify_receipt(self.drive.publish(self.root / "protocol.json", f"run_for_{self.protocol['source_bundle_sha256']}.json"))
        self.publish(self.bundle)
        self.save_csv("validation_roles", self.roles)
        self.save_csv("mimic_replacements", self.replacements)
        self.publish(audited_split)
        print(f"Run {self.run_id}. Private Drive {self.drive.link}")
        display(self.replacements.groupby(["source", "subtype"]).mimic_image_id.agg(["size", "nunique"]))

    def load_parent(self, model):
        path = self.parent_reader.fetch_artifact(self.parent_model)
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["run_id"], payload["protocol_sha256"], payload["arm"], payload["epoch"]
        if metadata != (RUNS["baseline"][0], self.parent_protocol_sha, "baseline", 3):
            raise RuntimeError("Starting checkpoint differs from Notebook 11's final baseline.")
        model.load_state_dict(payload["model"], strict=True)

    def training_caches(self, arm):
        key = self.second_cache if arm == self.second_arm else "train_clear"
        return key, key

    @staticmethod
    def image(archives, row):
        with archives[row.source].open(row.member) as stream, Image.open(stream) as photo:
            rgb = ImageOps.exif_transpose(photo).convert("RGB")
        if pixel_hash(rgb) != row.pixel_sha256:
            raise RuntimeError(f"Decoded pixels differ for {row.image_id}.")
        return correct_rgb(np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))

    def prepare_images(self, archive_folder=None):
        frames = {"train_clear": self.train_manifest, "original": self.validation, "mimic_test": self.mimic_test,
                  self.second_cache: self.train_manifest}
        paths = {name: self.root / f"{name}.npy" for name in frames}
        cache_receipt = self.root / "image_cache.json"
        if cache_receipt.is_file():
            hashes = json.loads(cache_receipt.read_bytes())
            if all(path.is_file() and digest(path) == hashes.get(k) for k, path in paths.items()):
                self.images = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
                print("Verified cached inputs.")
                return
        archive_root = Path(archive_folder) if archive_folder else self.work / "source_archives"
        if not archive_folder:
            archive_root.mkdir(parents=True, exist_ok=True)
        download_bytes = sum(x["bytes"] for k, x in self.sources.items()
                             if not (archive_root / f"{k}.zip").is_file())
        required = download_bytes + sum(len(f) for f in frames.values()) * SIZE * SIZE * 3 + 6 * 1024**3
        if shutil.disk_usage(self.work).free < required:
            raise RuntimeError("Insufficient temporary disk for archives, caches and checkpoints.")
        with contextlib.ExitStack() as stack:
            archives = {}
            for name, spec in self.sources.items():
                path = archive_root / f"{name}.zip"
                if archive_folder:
                    if path.stat().st_size != spec["bytes"] or (spec["md5"] and digest(path, "md5") != spec["md5"]):
                        raise RuntimeError(f"Incorrect {path.name} release.")
                else:
                    path = download_file_curl(path, spec["url"], expected_bytes=spec["bytes"], expected_md5=spec["md5"])
                archives[name] = stack.enter_context(ZipFile(path))
            for name, frame in frames.items():
                if name == self.second_cache:
                    continue
                with_maps = np.lib.format.open_memmap(paths[name], mode="w+", dtype=np.uint8,
                                                       shape=(len(frame), SIZE, SIZE, 3))
                for i, row in enumerate(frame.itertuples(index=False)):
                    with_maps[i] = self.image(archives, row)
                    if (i + 1) % 1000 == 0 or i + 1 == len(frame):
                        print(f"{name}: verified {i + 1:,}/{len(frame):,}", flush=True)
                with_maps.flush()
                del with_maps
            shutil.copyfile(paths["train_clear"], paths[self.second_cache])
            candidate = np.load(paths[self.second_cache], mmap_mode="r+")
            pool = {(row.source, row.image_id): row for row in self.mimic_train.itertuples(index=False)}
            for (source, image_id), group in self.replacements.groupby(["source", "mimic_image_id"]):
                candidate[group.slot.to_numpy()] = self.image(archives, pool[source, image_id])
            candidate.flush()
            del candidate
        cache_receipt.write_text(json.dumps({name: digest(p) for name, p in paths.items()}), encoding="utf-8")
        self.images = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}

    def evaluate(self):
        old = self.restore_json("evaluation_receipt")
        if old:
            if old["protocol_sha256"] != self.protocol_sha or set(old["files"]) != REPORTS:
                raise RuntimeError("Invalid saved evaluation receipt.")
            for ref in old["files"].values():
                self.drive.verify_receipt(ref)
            display(pd.read_csv(self.restore_file(old["files"]["summary"])))
            print(json.loads(self.restore_file(old["files"]["decision"]).read_bytes()))
            return
        states = {arm: self.latest_epoch(arm) for arm in ARMS}
        if any(state["epoch"] != len(RATES) for state in states.values()):
            raise RuntimeError("Finish both verified training arms before scoring the final test cases.")
        cal = self.roles.role.eq("calibration").to_numpy()
        source = self.roles.assign(cohort=np.where(self.roles.label == 1, "melanomas", "nevi"))
        source["label"] = source.label.astype(str)
        mimics = self.mimic_test[["image_id", "source", "lesion_id", "label"]].assign(
            group=self.mimic_test.source + ":" + self.mimic_test.lesion_id, role="evaluation", cohort="mimics")
        cutoffs, models = {}, {}
        for arm, state in states.items():
            payload = self.torch.load(self.restore_file(state["files"]["selected"]), map_location="cpu", weights_only=True)
            if (payload["run_id"], payload["protocol_sha256"], payload["arm"], payload["epoch"]) != (
                    self.run_id, self.protocol_sha, arm, len(RATES)):
                raise RuntimeError("Final model belongs to a different training arm.")
            model = self.new_model()
            model.load_state_dict(payload["model"], strict=True)
            models[arm] = model
            source.loc[cal, arm] = self._score(model, self.images["original"][cal])
            cutoffs[arm] = source_cutoff(self.roles.loc[cal, "label"], source.loc[cal, arm])
            self.save_json(arm + "_locked_cutoff", cutoffs[arm])
        for arm, model in models.items():
            source.loc[~cal, arm] = self._score(model, self.images["original"][~cal])
            mimics[arm] = self._score(model, self.images["mimic_test"])
        del models, model, payload
        self.torch.cuda.empty_cache()
        predictions = pd.concat([source, mimics], ignore_index=True)
        table, intervals = summarize(predictions, cutoffs)
        result = decision(table)
        files = {"predictions": self.save_csv("predictions", predictions), "summary": self.save_csv("summary", table),
                 "paired_intervals": self.save_json("paired_intervals", intervals), "decision": self.save_json("decision", result)}
        self.save_json("evaluation_receipt", {"protocol_sha256": self.protocol_sha, "files": files})
        display(table[table.group == "pooled"].round(3))
        print(result)

    def additional_receipts(self):
        names = ("mimic_replacements.csv", f"{PREFLIGHT[0]}__mimic_split.csv")
        refs = {}
        for name in names:
            path = self.root / name
            if not self.drive.restore(f"{self.run_id}__{name}", path):
                raise RuntimeError(f"Missing saved assignment: {name}")
            refs[name] = self.publish(path)
        return refs
