"""Paired inference on frozen ISIC 2020 development cohorts."""
import os

import io
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import PIL
import sklearn
from PIL import Image, ImageOps

from acquisition_stress import canonical_hash, correct_rgb, pixel_hash
from backbone_resolution_study import (
    BATCH_SIZE, ISIC2020_ZIP_BYTES, ISIC2020_ZIP_URL, NB08_FOLDER, NB08_RECEIPT_SHA256,
    NB08_RUN, TRAIN_FOLDER, TRAIN_RUN, build_convnext_model, paired_delta_ci, score_convnext_4tta,
)
from inference_store import CheckpointDriveStore, FolderReader, RemoteZip, digest, fetch_zip_member_bytes
from step0_nasnet_features import summary_metrics

ARMS = ("baseline", "contrast_gamma")
RUNS = {
    "baseline": ("657b7afd32bd50c56309", os.getenv("CONV_FUSION_BASELINE_FOLDER", ""),
                 "f42f2f43d4f9d5401f290f4a723f142163a0efe29862338864e20fd611673bf2"),
    "contrast_gamma": ("ab249456e47d281ef4b5", os.getenv("CONV_FUSION_CONTRAST_GAMMA_FOLDER", ""),
                       "2ee76064b41a405a97aca7da3594fa6522b9dc58ce8788b959eb7e1f587c0f49"),
}
SOURCE_SPLIT = f"{TRAIN_RUN}__pooled_ham_bcn_split.csv"
NEVI = "spectrum_shades_of_gray_aug_nevi_cohort"
UNSPEC = "spectrum_shades_of_gray_aug_unspecified_cohort"
CONTRAST_BOUND = 49.11399841308594
PART_SIZE = 256
BOOTSTRAP_DRAWS = 1000
RULE = {"low_contrast_min_gain_pp": 3, "max_false_alarm_pct": 10,
        "full_max_sensitivity_loss_pp": 2, "max_false_alarm_increase_pp": 1}


def frozen_receipt(reader, run, sha256):
    path = reader.fetch(f"{run}__durable_save_receipt.json")
    result = json.loads(path.read_bytes())
    if digest(path) != sha256 or result["run_id"] != run:
        raise RuntimeError(f"Source receipt differs from the frozen run {run}.")
    return result


def align_cohorts(nevi, unspecified, extra_images, properties):
    identity = ["image_id", "patient_id", "label"]
    nevi = nevi[identity].set_index("image_id", verify_integrity=True)
    unspecified = unspecified[identity].set_index("image_id", verify_integrity=True)
    common = nevi.index.intersection(unspecified.index)
    if set(common) != set(nevi[nevi.label == 1].index) or set(common) != set(unspecified[unspecified.label == 1].index):
        raise RuntimeError("The ISIC cohorts must share exactly the same melanomas.")
    pd.testing.assert_frame_equal(nevi.loc[common], unspecified.loc[common])
    if nevi.label.value_counts().to_dict() != {0: 5191, 1: 581} or unspecified.label.value_counts().to_dict() != {0: 1000, 1: 581}:
        raise RuntimeError("The historical ISIC cohort counts changed.")
    patients = pd.concat([nevi, unspecified.loc[~unspecified.index.isin(common)]])
    manifest = patients.copy()
    manifest["in_labelled_nevi"] = manifest.index.isin(nevi.index)
    manifest["in_unspecified_benign"] = manifest.index.isin(unspecified.index)
    prop = properties.set_index("image_id", verify_integrity=True)
    if not set(prop.index) <= set(nevi.index):
        raise RuntimeError("Saved image properties include photos outside the labelled-nevi cohort.")
    pd.testing.assert_frame_equal(prop[["patient_id", "label"]], patients.loc[prop.index])
    extra = extra_images.set_index("image_id", verify_integrity=True)
    if set(extra.index) != set(unspecified[unspecified.label == 0].index):
        raise RuntimeError("The unspecified-benign decoded-pixel manifest changed.")
    pd.testing.assert_frame_equal(extra[["patient_id", "label"]], patients.loc[extra.index])
    archived_hashes = pd.concat([prop["pixel_sha256"], extra["pixel_sha256"]], verify_integrity=True)
    if not archived_hashes.str.fullmatch(r"[0-9a-f]{64}").all() or archived_hashes.duplicated().any():
        raise RuntimeError("Invalid or duplicated archived pixel fingerprints.")
    manifest["archived_pixel_sha256"] = archived_hashes.reindex(manifest.index, fill_value="")
    manifest["historical_contrast"] = prop["contrast_p90_p10"]
    manifest["in_low_contrast"] = manifest.historical_contrast.le(CONTRAST_BOUND)
    if manifest.loc[manifest.in_low_contrast, "label"].value_counts().to_dict() != {0: 514, 1: 146}:
        raise RuntimeError("The frozen low-contrast subset changed.")
    if manifest.patient_id.eq("").any():
        raise RuntimeError("Unresolved ISIC patient identities.")
    return manifest.sort_index().reset_index()


def equal_allowance(predictions, arm):
    cal = predictions.loc[(predictions.arm == arm) & (predictions["style"] == "original") &
                          (predictions.role == "calibration") & (predictions.label == 0), "score"]
    scores = np.sort(cal.to_numpy(dtype=float))[::-1]
    if len(scores) != 534 or not np.isfinite(scores).all():
        raise RuntimeError("Expected 534 archived benign calibration scores.")
    cutoff = float(np.nextafter(scores[45], np.inf))
    if int(np.sum(scores >= cutoff)) != 45:
        raise RuntimeError("Cannot recover the exact 45-case allowance.")
    return cutoff


def analyze(predictions, cutoffs):
    cohorts = {
        "low_contrast": predictions.in_low_contrast,
        "labelled_nevi": predictions.in_labelled_nevi,
        "unspecified_benign": predictions.in_unspecified_benign,
        "remainder_of_labelled_nevi": predictions.in_labelled_nevi & ~predictions.in_low_contrast,
    }
    records, comparisons = [], {}
    checked = ("low_contrast", "labelled_nevi", "unspecified_benign")
    for name, mask in cohorts.items():
        data = predictions[mask]
        for policy, cuts in cutoffs.items():
            for arm in ARMS:
                records.append({"cohort": name, "cutoff_policy": policy, "arm": arm, "cutoff": cuts[arm],
                                **summary_metrics(data.label, data[arm], cuts[arm])})
            if name in checked:
                comparisons[f"{name}/{policy}"] = paired_delta_ci(
                    data.label, data.patient_id, data.baseline, cuts["baseline"],
                    data.contrast_gamma, cuts["contrast_gamma"], draws=BOOTSTRAP_DRAWS,
                )
    return pd.DataFrame(records), comparisons


def screen(summary):
    locked = summary[summary.cutoff_policy == "original_locked"].set_index(["cohort", "arm"])
    fa = "Benign false alarms (%)"
    tpr = "Melanomas caught (%)"
    scores = locked[[tpr, fa]]
    delta = scores.xs("contrast_gamma", level="arm") - scores.xs("baseline", level="arm")
    checks = {
        "low_contrast_detection_gain": bool(delta.at["low_contrast", tpr] >= RULE["low_contrast_min_gain_pp"]),
        "low_contrast_false_alarm_cap": bool(locked.at[("low_contrast", "contrast_gamma"), fa] <= RULE["max_false_alarm_pct"]),
        "full_melanoma_detection_preserved": bool(delta.at["labelled_nevi", tpr] >= -RULE["full_max_sensitivity_loss_pp"]),
        "labelled_nevi_false_alarms_preserved": bool(delta.at["labelled_nevi", fa] <= RULE["max_false_alarm_increase_pp"]),
        "unspecified_benign_false_alarms_preserved": bool(delta.at["unspecified_benign", fa] <= RULE["max_false_alarm_increase_pp"]),
    }
    return {"checks": checks, "rule": RULE, "screen_passed": all(checks.values()),
            "recommended_mimic_experiment_starting_checkpoint": "contrast_gamma" if all(checks.values()) else "baseline",
            "scope": "Previously examined ISIC development cohort. A pass supports independent confirmation."}


class ISICCheckpointComparison:
    def __init__(self, bundle, work="/content/isic-checkpoint-comparison", *, service=None):
        import timm
        import torch

        if timm.__version__ != "1.0.22" or not torch.cuda.is_available():
            raise RuntimeError("Run setup and select a GPU runtime.")
        if service is None:
            from google.colab import auth
            from googleapiclient.discovery import build

            auth.authenticate_user()
            service = build("drive", "v3", cache_discovery=False)
        self.torch, self.timm = torch, timm
        self.device = torch.device("cuda:0")
        self.bundle = Path(bundle)
        work = Path(work)
        self.readers = {arm: FolderReader(service, folder, work / arm) for arm, (_, folder, _) in RUNS.items()}
        receipts = {arm: frozen_receipt(self.readers[arm], run, sha) for arm, (run, _, sha) in RUNS.items()}
        self.models = {arm: receipts[arm]["files"][f"{arm}_epoch03_selected"] for arm in ARMS}
        self.model_protocols = {arm: receipts[arm]["protocol_sha256"] for arm in ARMS}
        if receipts["baseline"]["parent_model"] != receipts["contrast_gamma"]["parent_model"]:
            raise RuntimeError("The matched models have different starting checkpoints.")
        self.cutoffs = {"original_locked": {}, "equal_45_of_534": {}}
        for arm in ARMS:
            self.readers[arm].verify_artifact(self.models[arm])
            path = self.readers[arm].fetch_artifact(receipts[arm]["files"][f"{arm}_locked_cutoff.json"])
            self.cutoffs["original_locked"][arm] = json.loads(path.read_bytes())["threshold"]
        candidate_reader = self.readers["contrast_gamma"]
        candidate_predictions = pd.read_csv(
            candidate_reader.fetch_artifact(receipts["contrast_gamma"]["files"]["predictions"]),
            converters={"score": float},
        )
        self.cutoffs["equal_45_of_534"] = {arm: equal_allowance(candidate_predictions, arm) for arm in ARMS}
        nb08 = FolderReader(service, NB08_FOLDER, work / "cohorts")
        source = FolderReader(service, TRAIN_FOLDER, work / "source_split")
        cohort_receipt = frozen_receipt(nb08, NB08_RUN, NB08_RECEIPT_SHA256)
        split_path = source.fetch(SOURCE_SPLIT)
        if digest(split_path) != cohort_receipt["protocol"]["input_files"][SOURCE_SPLIT]["sha256"]:
            raise RuntimeError("The source image manifest differs from the frozen input.")
        self.source_pixels = set(pd.read_csv(split_path, usecols=["pixel_sha256"]).pixel_sha256)
        refs = {key: cohort_receipt["files"][key] for key in (NEVI, UNSPEC, "spectrum_unknown_manifest", "properties_isic_images")}

        def read_cohort(key):
            return pd.read_csv(nb08.fetch_artifact(refs[key]), keep_default_na=False)

        self.manifest = align_cohorts(read_cohort(NEVI), read_cohort(UNSPEC), read_cohort("spectrum_unknown_manifest"),
                                      read_cohort("properties_isic_images"))
        self.protocol = {
            "scope": "Paired inference on frozen ISIC 2020 development photos", "source_bundle_sha256": digest(bundle),
            "model_artifacts": self.models, "model_protocols": self.model_protocols, "reference_runs": RUNS,
            "cohort_receipt_sha256": NB08_RECEIPT_SHA256, "cohort_refs": refs,
            "source_image_manifest_sha256": digest(split_path), "cutoffs": self.cutoffs,
            "archive": {"url": ISIC2020_ZIP_URL, "bytes": ISIC2020_ZIP_BYTES,
                        "checks": "Member CRC, archived decoded hashes where available, no decoded-pixel overlap"},
            "contrast_bound": CONTRAST_BOUND, "images": len(self.manifest), "part_size": PART_SIZE,
            "fixed_screen": RULE, "bootstrap_draws": BOOTSTRAP_DRAWS, "batch_size": BATCH_SIZE,
            "preprocessing": "EXIF RGB, 224 square bilinear, Shades of Gray, four-view raw score average",
            "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                         "pillow": PIL.__version__, "sklearn": sklearn.__version__,
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "gpu": torch.cuda.get_device_name(0)},
        }
        self.sha = canonical_hash(self.protocol)
        self.run = self.sha[:20]
        self.root = work / self.run
        self.root.mkdir(parents=True, exist_ok=True)
        self.drive = CheckpointDriveStore(service, "Melanoma ISIC Paired Checkpoint Comparison")
        self.probe = self.drive.preflight()["probe"]
        self.binding = self.save_json("protocol", {"run_id": self.run, "protocol_sha256": self.sha, "protocol": self.protocol})
        release = f"run_for_{self.protocol['source_bundle_sha256']}.json"
        self.drive.verify_receipt(self.drive.publish(self.root / "protocol.json", release))
        self.bundle_ref = self.publish(self.bundle)
        print(f"Run {self.run}. Locked cutoffs {self.cutoffs}. Private Drive {self.drive.link}")

    def name(self, filename):
        return f"{self.run}__{filename}"

    def publish(self, path):
        ref = self.drive.publish(path, self.name(path.name))
        self.drive.verify_receipt(ref)
        return ref

    def save_json(self, stem, value):
        path = self.root / f"{stem}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return self.publish(path)

    def save_csv(self, stem, value):
        path = self.root / f"{stem}.csv"
        value.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        return self.publish(path)

    def load_model(self, arm):
        path = self.readers[arm].fetch_artifact(self.models[arm])
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        actual = payload["run_id"], payload["protocol_sha256"], payload["arm"], payload["epoch"]
        if actual != (RUNS[arm][0], self.model_protocols[arm], arm, 3):
            raise RuntimeError("Checkpoint source or epoch differs from the expected model.")
        model = build_convnext_model(self.torch, self.timm, pretrained=False)
        model.load_state_dict(payload["model"], strict=True)
        return model.to(self.device).eval()

    def prepare_case(self, row, archive_index):
        raw = fetch_zip_member_bytes(ISIC2020_ZIP_URL, archive_index[row.image_id], ISIC2020_ZIP_BYTES)
        with Image.open(io.BytesIO(raw)) as file:
            native = ImageOps.exif_transpose(file).convert("RGB")
        fingerprint = pixel_hash(native)
        if row.archived_pixel_sha256 and fingerprint != row.archived_pixel_sha256:
            raise RuntimeError(f"Decoded pixels differ for {row.image_id}.")
        if fingerprint in self.source_pixels:
            raise RuntimeError(f"{row.image_id} duplicates a source training or validation photo.")
        pixels = native.resize((224, 224), Image.Resampling.BILINEAR)
        return correct_rgb(np.asarray(pixels)), fingerprint

    def score(self):
        self.drive.verify_receipt(self.probe)
        batches, refs = [], {}
        models = None
        seen = set()
        for start in range(0, len(self.manifest), PART_SIZE):
            chunk = self.manifest.iloc[start:start + PART_SIZE]
            stem = f"scores_{start:05d}"
            path = self.root / f"{stem}.csv"
            restored = self.drive.restore(self.name(path.name), path)
            if restored:
                part = pd.read_csv(path, converters={arm: float for arm in ARMS})
                if part.image_id.tolist() != chunk.image_id.tolist() or not np.isfinite(part[list(ARMS)]).all().all():
                    raise RuntimeError("Saved score partition differs from its frozen image order.")
            else:
                if models is None:
                    models = {arm: self.load_model(arm) for arm in ARMS}
                    wanted = set(self.manifest.image_id)
                    with ZipFile(RemoteZip(ISIC2020_ZIP_URL, ISIC2020_ZIP_BYTES)) as archive:
                        archive_index = {}
                        for item in archive.infolist():
                            member = PurePosixPath(item.filename)
                            if member.suffix.lower() == ".jpg" and member.stem in wanted:
                                if member.stem in archive_index:
                                    raise RuntimeError(f"Ambiguous ISIC archive member: {member.stem}")
                                archive_index[member.stem] = item
                    if set(archive_index) != wanted:
                        raise RuntimeError("Missing photos in the frozen ISIC archive.")
                with ThreadPoolExecutor(max_workers=12) as pool:
                    photos = list(pool.map(lambda row: self.prepare_case(row, archive_index), chunk.itertuples(index=False)))
                part = chunk[["image_id"]].copy()
                pixels, part["pixel_sha256"] = zip(*photos, strict=True)
                pixels = np.stack(pixels)
                for arm in ARMS:
                    part[arm] = score_convnext_4tta(self.torch, models[arm], pixels, self.device, batch_size=BATCH_SIZE)
                if not np.isfinite(part[list(ARMS)]).all().all():
                    raise RuntimeError("Invalid checkpoint predictions.")
            hashes = set(part.pixel_sha256)
            if (not part.pixel_sha256.str.fullmatch(r"[0-9a-f]{64}").all() or len(hashes) != len(part)
                    or hashes & self.source_pixels or hashes & seen):
                raise RuntimeError("Unexpected duplicate ISIC pixels or overlap with source photos.")
            archived = chunk.archived_pixel_sha256.to_numpy()
            if np.any((archived != "") & (archived != part.pixel_sha256.to_numpy())):
                raise RuntimeError("Saved partition does not match archived pixel fingerprints.")
            seen.update(hashes)
            refs[stem] = self.publish(path) if restored else self.save_csv(stem, part)
            batches.append(part)
            print(f"{min(start + PART_SIZE, len(self.manifest)):,}/{len(self.manifest):,} paired scores verified.", flush=True)
        del models
        self.torch.cuda.empty_cache()
        self.predictions = self.manifest.merge(pd.concat(batches), on="image_id", validate="one_to_one")
        self.parts = refs

    def finish(self):
        from IPython.display import display

        table, intervals = analyze(self.predictions, self.cutoffs)
        decision = screen(table)
        files = {"protocol": self.binding, "bundle": self.bundle_ref, **self.parts,
                 "predictions": self.save_csv("predictions", self.predictions),
                 "summary": self.save_csv("summary", table),
                 "paired_intervals": self.save_json("paired_intervals", intervals),
                 "decision": self.save_json("decision", decision)}
        for ref in files.values():
            self.drive.verify_receipt(ref)
        for arm in ARMS:
            self.readers[arm].verify_artifact(self.models[arm])
        receipt = {"run_id": self.run, "protocol_sha256": self.sha, "inputs": self.models,
                   "files": files, "paired_images": len(self.predictions), "scope": self.protocol["scope"]}
        ref = self.save_json("durable_save_receipt", receipt)
        fresh = self.root / "receipt_readback.json"
        if not self.drive.restore(ref["name"], fresh) or digest(fresh) != ref["sha256"] or json.loads(fresh.read_bytes()) != receipt:
            raise RuntimeError("The final receipt failed Drive readback.")
        display(table.round(3))
        print(decision)
        print(f"Verified paired outputs on private Drive. {self.drive.link}")
        return receipt
