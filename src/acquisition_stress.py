"""Source-validation inference check for the saved melanoma classifier. No training."""
import os

import gc
import hashlib
import io
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image, ImageOps, __version__ as pillow_version
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score, roc_curve

from inference_store import CheckpointDriveStore, FolderReader, RemoteZip, digest, fetch_zip_member_bytes

TRAIN_RUN = "3c9cf6562d30585daf4b"
TRAIN_FOLDER = os.getenv("CONV_FUSION_TRAIN_FOLDER", "")
TRAIN_RECEIPT_SHA256 = "38d20d324460c26c762520efc30ed7bd41051b84ad131c3c8f1b85c86d8cdfc5"
OUTPUT_FOLDER = "Melanoma acquisition robustness check"
WORK = Path("/content/melanoma-acquisition-robustness")
ARM = "shades_of_gray_aug"
SEEDS = (17, 29, 43)
IMAGE_SIZE = 224
BATCH_SIZE = 32
CALIBRATION_PER_SOURCE_CLASS = 32
SAMPLE_SEED = 904
BOOTSTRAP_DRAWS = 500
REPLAY_PROBABILITY_ATOL = 0.001
FALSE_ALARM_LIMIT = 0.10
STYLES = ("original", "contrast_lower", "contrast_low", "gamma_bright", "warmer", "cooler", "lower_resolution")
STYLE_NAMES = {"original": "Original", "contrast_lower": "Lower contrast", "contrast_low": "Low contrast",
               "gamma_bright": "Brighter exposure (gamma 0.75)",
               "warmer": "Warmer colour cast", "cooler": "Cooler colour cast", "lower_resolution": "Lower resolution"}
SOURCE_MAX_WORKERS = {"HAM": 8, "BCN": 3}
REQUIRED_STAGES = ("data", "replay", "stress", "report")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def uint8(values):
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def correct_rgb(pixels):
    # Identical to the selected training recipe: float64, mask, p=6, rint, clip.
    rgb = pixels.astype(np.float64)
    valid = (pixels.max(axis=-1) > 15) & (pixels.min(axis=-1) < 250)
    sample = rgb[valid] if np.count_nonzero(valid) >= 256 else rgb.reshape(-1, 3)
    illuminant = np.power(np.mean(np.power(sample, 6.0), axis=0), 1 / 6.0)
    gains = np.divide(illuminant.mean(), illuminant, out=np.ones(3), where=illuminant > 0)
    return uint8(rgb * gains)


def pixel_hash(image):
    result = hashlib.sha256(f"{image.width}x{image.height}:RGB\0".encode())
    result.update(image.tobytes())
    return result.hexdigest()


def contrast_value(image):
    f = np.asarray(image, dtype=np.float64)
    lum = f[..., 0] * .299 + f[..., 1] * .587 + f[..., 2] * .114
    return float(np.percentile(lum, 90) - np.percentile(lum, 10))


def validate_split(frame):
    required = {"source", "image_id", "lesion_id", "label", "group", "pixel_sha256", "split", "member"}
    if not required.issubset(frame.columns):
        raise RuntimeError("The saved split is missing required image fields.")
    frame = frame.copy()
    if not frame["label"].astype(str).isin(("0", "1")).all():
        raise RuntimeError("The saved split contains an unexpected diagnosis label.")
    if not frame["source"].isin(("HAM", "BCN")).all() or not frame["split"].isin(("train", "validation")).all():
        raise RuntimeError("This check accepts the original HAM/BCN split only.")
    frame["label"] = frame["label"].astype(int)
    if frame["image_id"].duplicated().any():
        raise RuntimeError("The saved split repeats an image ID.")
    train, val = (frame[frame["split"] == part] for part in ("train", "validation"))
    for field in ("group", "pixel_sha256", "image_id"):
        if frame[field].eq("").any() or set(train[field]) & set(val[field]):
            raise RuntimeError(f"Training and validation are not separated by {field}.")
    if set(zip(train.source, train.lesion_id)) & set(zip(val.source, val.lesion_id)):
        raise RuntimeError("Training and validation share a source lesion.")
    if val["group"].duplicated().any() or set(val["label"]) != {0, 1}:
        raise RuntimeError("Expected one validation photo per lesion group and both diagnoses.")
    return frame


def calibration_sample(frame):
    train = frame[frame["split"] == "train"].sort_values(["source", "image_id"])
    # Select one photo per lesion group, without any model score or validation outcome.
    train = train.drop_duplicates("group")
    pieces = []
    for (source, label), group in train.groupby(["source", "label"], sort=True):
        ordered = group.assign(_order=group["image_id"].map(
            lambda iid: hashlib.sha256(f"{SAMPLE_SEED}:{source}:{label}:{iid}".encode()).hexdigest()))
        pieces.append(ordered.sort_values("_order").head(CALIBRATION_PER_SOURCE_CLASS).drop(columns="_order"))
    if len(pieces) != 4:
        raise RuntimeError("Calibration requires both diagnoses from both training sources.")
    return pd.concat(pieces).sort_values(["source", "image_id"]).reset_index(drop=True)


def derive_policy(contrast_values):
    values = np.asarray(contrast_values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("Invalid training-sample contrast measurements.")
    q10, q25, median = np.quantile(values, [.10, .25, .50])
    if median <= 1:
        raise RuntimeError("Training-sample contrast is too low to set a meaningful policy.")
    lower = float(np.clip(q25 / median, .70, .90))
    low = min(float(np.clip(q10 / median, .55, .80)), lower - .10)
    return {"training_contrast_q10": float(q10), "training_contrast_q25": float(q25),
            "training_contrast_median": float(median), "contrast_lower": lower, "contrast_low": low,
            "gamma_bright": 0.75,
            "warmer_gains": [1.10, 1.0, .90], "cooler_gains": [.90, 1.0, 1.10],
            "resolution_side": 112, "application": "on resized RGB before the original Shades of Gray step",
            "interpretation": "Synthetic appearance challenges, not measured or certified camera shifts."}


def transform(image, style, policy):
    arr = np.asarray(image, dtype=np.uint8)
    if style == "original":
        return arr.copy()
    if style in ("contrast_lower", "contrast_low"):
        f = arr.astype(np.float64)
        return uint8((f - f.mean(axis=(0, 1), keepdims=True)) * policy[style] + f.mean(axis=(0, 1), keepdims=True))
    if style == "gamma_bright":
        gamma = float(policy["gamma_bright"])
        return uint8(255.0 * np.power(arr.astype(np.float64) / 255.0, gamma))
    if style in ("warmer", "cooler"):
        return uint8(arr.astype(np.float64) * np.asarray(policy[style + "_gains"]))
    if style == "lower_resolution":
        side = int(policy["resolution_side"])
        small = Image.fromarray(arr).resize((side, side), Image.Resampling.BILINEAR)
        return np.asarray(small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    raise ValueError(f"Unknown appearance: {style}")


def probability(scores):
    return 1 / (1 + np.exp(-np.clip(np.asarray(scores, dtype=float), -700, 700)))


def score_batch(model, pixels):
    views = (pixels, np.flip(pixels, axis=2), np.flip(pixels, axis=1), np.rot90(pixels, 2, axes=(1, 2)))
    values = []
    for view in views:
        result = model({"rgb": np.ascontiguousarray(view)}, training=False)
        values.append(np.asarray(result.numpy() if hasattr(result, "numpy") else result).reshape(-1))
    scores = np.mean(values, axis=0)
    if scores.shape != (len(pixels),) or not np.isfinite(scores).all():
        raise RuntimeError("Inference returned an invalid melanoma score.")
    return scores.astype(float)


def align_scores(frame, manifest):
    if frame["image_id"].duplicated().any() or set(frame["image_id"]) != set(manifest["image_id"]):
        raise RuntimeError("Prediction image IDs differ from the validation list.")
    result = frame.set_index("image_id").loc[manifest["image_id"]].reset_index()
    if not np.array_equal(result["label"].astype(int), manifest["label"].astype(int)):
        raise RuntimeError("Prediction labels differ from the validation list.")
    result["raw_score"] = pd.to_numeric(result["raw_score"])
    if not np.isfinite(result["raw_score"]).all():
        raise RuntimeError("A prediction file contains a non-finite score.")
    return result


def metrics(labels, scores, cutoff, weights=None):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    pos, neg, pred = y == 1, y == 0, s >= cutoff
    positive, negative = w[pos].sum(), w[neg].sum()
    if not positive or not negative:
        raise ValueError("Both diagnoses are required for performance estimates.")
    fp, tp, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    return {"Images": len(y), "Melanomas": int(pos.sum()), "Benign": int(neg.sum()),
            "Melanomas caught": int((pred & pos).sum()), "Melanomas missed": int((~pred & pos).sum()),
            "Benign false alarms": int((pred & neg).sum()),
            "Melanomas caught (%)": float(100 * w[pos & pred].sum() / positive),
            "Benign false alarms (%)": float(100 * w[neg & pred].sum() / negative),
            "Accuracy (%)": float(100 * w[pred == y].sum() / w.sum()),
            "AUROC": float(roc_auc_score(y, s, sample_weight=w)),
            "Caught at up to 10% false alarms (%)": float(100 * tp[fp <= FALSE_ALARM_LIMIT].max())}


def change_counts(labels, before, after, cutoff):
    y = np.asarray(labels, dtype=int)
    a, b = np.asarray(before) >= cutoff, np.asarray(after) >= cutoff
    return {"Previously caught melanomas now missed": int(((y == 1) & a & ~b).sum()),
            "Previously missed melanomas now caught": int(((y == 1) & ~a & b).sum()),
            "New benign false alarms": int(((y == 0) & ~a & b).sum()),
            "Benign false alarms removed": int(((y == 0) & a & ~b).sum())}


def paired_intervals(manifest, before, after, cutoff, draws=BOOTSTRAP_DRAWS):
    # Patient IDs do not exist in this saved manifest. Do not call these patient intervals.
    codes, units = pd.factorize(manifest["source"] + ":" + manifest["group"])
    rng = np.random.default_rng(41)
    changes = []
    fields = ("Melanomas caught (%)", "Benign false alarms (%)", "AUROC", "Caught at up to 10% false alarms (%)")
    for _ in range(draws):
        weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)))[codes]
        try:
            a, b = (metrics(manifest["label"], s, cutoff, weights) for s in (before, after))
        except ValueError:
            continue
        changes.append([b[k] - a[k] for k in fields])
    if len(changes) < .90 * draws:
        raise RuntimeError("Too few valid bootstrap samples for the paired interval.")
    bounds = np.quantile(changes, [.025, .975], axis=0)
    return {"unit": "saved lesion group", "groups": len(units), "valid_draws": len(changes),
            "changes_95_percent": {key: list(map(float, bounds[:, i])) for i, key in enumerate(fields)},
            "scope": "Exploratory, no multiple-comparison correction; unavailable patient grouping is not accounted for."}


class AcquisitionStressTest:
    def __init__(self, implementation_files, work=WORK):
        import tensorflow as tf
        from google.colab import auth
        from googleapiclient.discovery import build

        self.tf, self.work = tf, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        auth.authenticate_user()
        service = build("drive", "v3", cache_discovery=False)
        self.reader = FolderReader(service, TRAIN_FOLDER, self.work / "source")
        receipt_path = self.reader.fetch(f"{TRAIN_RUN}__save_receipt.json")
        if digest(receipt_path) != TRAIN_RECEIPT_SHA256:
            raise RuntimeError("The original training receipt changed; this notebook will not guess another run.")
        self.input_paths = {"training_receipt": receipt_path}
        receipt = json.loads(receipt_path.read_text())
        val_path = self.reader.fetch_artifact(receipt["validation_report"])
        val = json.loads(val_path.read_text())
        if receipt["run_id"] != TRAIN_RUN or val["run_id"] != TRAIN_RUN:
            raise RuntimeError("The restored results refer to a different training run.")
        self.input_paths["validation_report"] = val_path
        split_ref = next(x for x in receipt["files"] if x["name"].endswith("__pooled_ham_bcn_split.csv"))
        split_path = self.reader.fetch_artifact(split_ref)
        pred_path = self.reader.fetch_artifact(val["predictions"])
        self.input_paths.update(split=split_path, validation_predictions=pred_path)
        self.split = validate_split(pd.read_csv(split_path, dtype=str, keep_default_na=False))
        self.manifest = self.split[self.split["split"] == "validation"].sort_values(["source", "image_id"]).reset_index(drop=True)
        self.calibration = calibration_sample(self.split)
        expected_counts = val["protocol"]["cohorts"]
        if len(self.manifest) != expected_counts["validation"] or (self.split["split"] == "train").sum() != expected_counts["train"]:
            raise RuntimeError("The saved training/validation counts changed.")
        if int(val["protocol"]["batch_size"]) != BATCH_SIZE or int(val["protocol"]["image_size"]) != IMAGE_SIZE:
            raise RuntimeError("The saved image or reference batch size changed.")
        saved = pd.read_csv(pred_path, dtype={"seed": str, "image_id": str}, keep_default_na=False,
                            converters={"raw_score": float})
        saved = saved[(saved["arm"] == ARM) & (saved["appearance"] == "original")]
        self.reference = {str(seed): align_scores(saved[saved["seed"] == str(seed)], self.manifest)
                          for seed in (*SEEDS, "ensemble")}
        average = np.mean([self.reference[str(seed)]["raw_score"].to_numpy() for seed in SEEDS], axis=0)
        np.testing.assert_allclose(average, self.reference["ensemble"]["raw_score"], atol=1e-10, rtol=1e-10)
        compare = [x for x in val["comparison"] if x["arm"] == ARM and x["appearance"] == "original"]
        self.cutoffs = {str(x["seed"]): float(x["cutoff"]) for x in compare}
        self.cutoff = self.cutoffs["ensemble"]
        self.models = {str(seed): val["model_states"][ARM][str(seed)]["selected"] for seed in SEEDS}
        self.sources = val["protocol"]["image_sources"]
        actual = metrics(self.manifest["label"], average, self.cutoff)
        expected = next(x for x in compare if str(x["seed"]) == "ensemble")
        if (actual["Melanomas caught"] != expected["fixed_cutoff"]["caught"] or
                actual["Benign false alarms"] != expected["fixed_cutoff"]["false_alarms"] or
                abs(actual["AUROC"] - expected["auroc"]) > 1e-10):
            raise RuntimeError("The restored scores and cutoff do not reproduce the saved validation result.")
        self.implementation_files = dict(implementation_files)
        self.protocol = {
            "training_run": TRAIN_RUN, "data_role": "source validation; no external images or retraining",
            "inputs": {key: {"bytes": path.stat().st_size, "sha256": digest(path)} for key, path in self.input_paths.items()},
            "implementation": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sorted(self.implementation_files.items())},
            "selected_models": self.models, "seeds": list(SEEDS), "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE, "tta_views": 4, "cutoffs": self.cutoffs, "false_alarm_limit": FALSE_ALARM_LIMIT,
            "validation_list_sha256": canonical_hash(self.manifest.to_dict("records")),
            "calibration_list_sha256": canonical_hash(self.calibration.to_dict("records")),
            "calibration": {"source": "training only", "per_source_diagnosis_lesions": CALIBRATION_PER_SOURCE_CLASS,
                            "seed": SAMPLE_SEED, "lower_contrast_ratio": "clip(q25/median, .70, .90)",
                            "low_contrast_ratio": "min(clip(q10/median, .55, .80), lower-.10)"},
            "styles": list(STYLES), "gamma_bright": 0.75, "colour_gains": {"warmer": [1.10, 1., .90], "cooler": [.90, 1., 1.10]},
            "resolution_side": 112, "bootstrap_draws": BOOTSTRAP_DRAWS, "replay_probability_atol": REPLAY_PROBABILITY_ATOL,
            "software": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "pillow": pillow_version,
                         "sklearn": sklearn_version, "matplotlib": matplotlib.__version__},
            "scope": "An exploratory response to synthetic image changes; no causal proof or new external confirmation."}
        self.run_id = canonical_hash(self.protocol)[:20]
        self.directory = self.work / self.run_id
        self.directory.mkdir(exist_ok=True)
        self.artifacts, self.catalogs, self.images = {}, {}, {}
        self.drive = CheckpointDriveStore(service, OUTPUT_FOLDER)
        self.drive.preflight(required_bytes=128 * 1024**2)
        self.save_json("protocol", {"run_id": self.run_id, "protocol": self.protocol})
        self.save_df("validation_manifest", self.manifest)
        self.save_df("calibration_manifest", self.calibration)
        self.save_json("source_models", {"training_folder": TRAIN_FOLDER, "selected": self.models})
        self.save_json("source_inputs", {key: {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                                         for key, path in self.input_paths.items()})
        code_path = self.directory / "implementation.zip"
        with ZipFile(code_path, "w") as archive:
            for name, contents in sorted(self.implementation_files.items()):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                archive.writestr(info, contents)
        self.publish("implementation", code_path)
        print(f"Run {self.run_id} | Private results: {self.drive.link}")
        print("Dataset connected: the original HAM10000 + BCN20000 split and saved model scores are verified.")
        counts = self.split.groupby(["split", "source", "label"]).size().unstack("label", fill_value=0)
        display(counts.rename(columns={0: "Benign photos", 1: "Melanoma photos"}))
        print(f"Validation: {len(self.manifest):,} photos; {int(self.manifest.label.sum())} melanomas. No external datasets are used.")
        print(f"Original saved cutoff: {self.cutoff:.4f}; caught {actual['Melanomas caught']}/{actual['Melanomas']}; "
              f"false alarms {actual['Benign false alarms']}/{actual['Benign']}.")
        print("Images will be fetched into temporary Colab storage, not downloaded to your Windows project.")

    def name(self, name):
        return f"{self.run_id}__{name}"

    def publish(self, logical, path):
        reference = self.drive.publish(path, self.name(path.name))
        self.drive.verify_receipt(reference)
        self.artifacts[logical] = reference
        return reference

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

    def restore_df(self, logical):
        path = self.directory / f"{logical}.csv"
        if not self.drive.restore(self.name(path.name), path):
            return None
        self.publish(logical, path)
        return pd.read_csv(path, dtype={"image_id": str, "seed": str}, keep_default_na=False,
                           converters={"raw_score": float})

    def restore_json(self, logical):
        path = self.directory / f"{logical}.json"
        if not self.drive.restore(self.name(path.name), path):
            return None
        self.publish(logical, path)
        return json.loads(path.read_text())

    def stage_done(self, stage):
        receipt = self.restore_json(stage + "_receipt")
        if receipt is None:
            return False
        if receipt["run_id"] != self.run_id or receipt["stage"] != stage or not receipt["files"]:
            raise RuntimeError("A stage receipt belongs to a different run or is empty.")
        for key, ref in receipt["files"].items():
            self.drive.verify_receipt(ref)
            self.artifacts[key] = ref
        return True

    def finish_stage(self, stage, required_keys):
        if not required_keys or any(key not in self.artifacts for key in required_keys):
            raise RuntimeError("A stage is missing required saved outputs.")
        self.save_json(stage + "_receipt", {"run_id": self.run_id, "stage": stage,
                        "files": {key: self.artifacts[key] for key in required_keys}})

    def save_figure(self, logical, figure):
        path = self.directory / f"{logical}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        self.publish(logical, path)
        plt.show()
        plt.close(figure)

    def catalog(self, source):
        if source not in self.catalogs:
            spec = self.sources[source]
            archive = ZipFile(RemoteZip(spec["url"], int(spec["bytes"])))
            self.catalogs[source] = {info.filename: info for info in archive.infolist()}
            archive.close()
        return self.catalogs[source]

    def fetch_image(self, row):
        cache = self.work / "images"
        cache.mkdir(exist_ok=True)
        path = cache / (row["image_id"] + "-" + row["pixel_sha256"][:16] + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as value:
                rgb = value["rgb"]
                props = json.loads(str(value["properties"]))
                if (str(value["original_sha256"]) != row["pixel_sha256"] or rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3)
                        or rgb.dtype != np.uint8 or hashlib.sha256(rgb.tobytes()).hexdigest() != props["resized_sha256"]):
                    raise RuntimeError("A local image cache changed. Remove only that cache file and refetch it.")
                return rgb, props
        info = self.catalog(row["source"]).get(row["member"])
        if info is None:
            raise RuntimeError(f"Image {row['image_id']} is absent from its original archive.")
        spec = self.sources[row["source"]]
        raw = fetch_zip_member_bytes(spec["url"], info, int(spec["bytes"]))
        with Image.open(io.BytesIO(raw)) as photo:
            original = ImageOps.exif_transpose(photo).convert("RGB")
        if pixel_hash(original) != row["pixel_sha256"]:
            raise RuntimeError(f"Decoded pixels changed for {row['image_id']}.")
        rgb = np.asarray(original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
        props = {"image_id": row["image_id"], "source": row["source"], "label": int(row["label"]),
                 "split": row["split"], "width": original.width, "height": original.height,
                 "contrast": contrast_value(rgb), "resized_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        partial = path.with_suffix(".npz.part")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, rgb=rgb, properties=json.dumps(props), original_sha256=row["pixel_sha256"])
        partial.replace(path)
        return rgb, props

    def ensure_images(self, frame=None):
        target = pd.concat([self.calibration, self.manifest], ignore_index=True) if frame is None else frame
        missing = target[~target["image_id"].isin(self.images)].drop_duplicates("image_id")
        if missing.empty:
            return
        total, done = len(missing), 0
        for source, group in missing.groupby("source", sort=True):
            self.catalog(source)
            workers = SOURCE_MAX_WORKERS.get(source, 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self.fetch_image, row) for row in group.to_dict("records")]
                for future in as_completed(futures):
                    rgb, props = future.result()
                    self.images[props["image_id"]] = (rgb, props)
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"Fetched and checked {done:,}/{total:,} needed photos.", flush=True)

    def prepare_images(self):
        if self.stage_done("data"):
            self.policy = self.restore_json("data_policy")
            print("Restored the saved image policy and image lists. Cached photos will be reused or refetched when needed.")
            return
        self.ensure_images()
        values = [self.images[i][1]["contrast"] for i in self.calibration["image_id"]]
        self.policy = derive_policy(values)
        self.save_json("data_policy", self.policy)
        properties = pd.DataFrame([props for _, props in self.images.values()]).sort_values(["split", "source", "image_id"])
        self.save_df("data_image_properties", properties)
        show = pd.concat([group.head(1) for _, group in self.calibration.groupby(["source", "label"], sort=True)])
        figure, axes = plt.subplots(len(show), len(STYLES), figsize=(13, 8), squeeze=False, layout="constrained")
        for row_axes, row in zip(axes, show.to_dict("records"), strict=True):
            rgb = self.images[row["image_id"]][0]
            for axis, style in zip(row_axes, STYLES, strict=True):
                axis.imshow(correct_rgb(transform(rgb, style, self.policy)))
                axis.set_title(STYLE_NAMES[style], fontsize=8)
                axis.set_xticks([]); axis.set_yticks([])
            row_axes[0].set_ylabel(f"{row['source']}\n{'Melanoma' if row['label'] else 'Benign mole'}", fontsize=9)
        figure.suptitle("What the classifier sees: fixed training examples, with the original colour correction")
        self.save_figure("data_preview", figure)
        self.finish_stage("data", ["data_policy", "data_image_properties", "data_preview"])
        print(f"Contrast multipliers from the training sample: {self.policy['contrast_lower']:.2f} and {self.policy['contrast_low']:.2f}.")
        print("These are synthetic challenges. A poor result will identify a candidate weakness, not prove the real ISIC cause.")

    def validate_prediction(self, frame, seed, style):
        frame = align_scores(frame, self.manifest)
        if (not frame["seed"].astype(str).eq(str(seed)).all() or not frame["style"].eq(style).all()
                or not frame["run_id"].eq(self.run_id).all()
                or not frame["policy_sha256"].eq(canonical_hash(self.policy)).all()):
            raise RuntimeError("A cached prediction belongs to a different seed, appearance, run or image policy.")
        return frame

    def score(self, model, seed, style):
        logical = f"scores_{style}_seed{seed}"
        restored = self.restore_df(logical)
        if restored is not None:
            return self.validate_prediction(restored, seed, style)
        vectors = []
        for start in range(0, len(self.manifest), BATCH_SIZE):
            rows = self.manifest.iloc[start:start+BATCH_SIZE]
            batch = np.stack([correct_rgb(transform(self.images[i][0], style, self.policy)) for i in rows["image_id"]])
            vectors.extend(score_batch(model, batch))
            done = min(start+BATCH_SIZE, len(self.manifest))
            if done % 512 == 0 or done == len(self.manifest):
                print(f"  {STYLE_NAMES[style]} | seed {seed} | {done:,}/{len(self.manifest):,}", flush=True)
        frame = self.manifest[["image_id", "source", "lesion_id", "group", "label"]].copy()
        frame["seed"], frame["style"], frame["run_id"] = str(seed), style, self.run_id
        frame["policy_sha256"], frame["raw_score"] = canonical_hash(self.policy), vectors
        self.save_df(logical, frame)
        return frame

    def load_model(self, seed):
        if not self.tf.config.list_physical_devices("GPU"):
            raise RuntimeError("Connect a Colab GPU before scoring. This notebook does not train a model.")
        self.tf.keras.backend.clear_session()
        gc.collect()
        path = self.reader.fetch_artifact(self.models[str(seed)])
        return self.tf.keras.models.load_model(path, compile=False)

    def benchmark(self, model):
        previous = self.restore_json("replay_timing")
        if previous is not None:
            print("The first-session timing estimate is already saved; replay and scoring can resume.")
            return
        rows = self.manifest.head(BATCH_SIZE)
        batch = np.stack([correct_rgb(self.images[i][0]) for i in rows["image_id"]])
        score_batch(model, batch)
        start = time.monotonic()
        score_batch(model, batch)
        seconds = time.monotonic() - start
        gpus = self.tf.config.list_physical_devices("GPU")
        name = self.tf.config.experimental.get_device_details(gpus[0]).get("device_name", gpus[0].name)
        estimate = seconds / len(batch) * len(self.manifest) * len(SEEDS) * len(STYLES)
        result = {"gpu": name, "batch_seconds_after_warmup": seconds, "photos_in_batch": len(batch),
                  "estimated_all_condition_inference_minutes": estimate / 60,
                  "scope": "Approximate forward-pass time; excludes image/model downloads, analysis and Drive saves."}
        self.save_json("replay_timing", result)
        print(f"GPU: {name}. Estimated total scoring time: about {estimate/60:.1f} minutes, plus downloads and saving.", flush=True)

    def ensemble(self, style):
        pieces = []
        for seed in SEEDS:
            frame = self.restore_df(f"scores_{style}_seed{seed}")
            if frame is None:
                raise RuntimeError("A required seed prediction is missing.")
            pieces.append(self.validate_prediction(frame, seed, style))
        result = pieces[0].copy()
        result["seed"] = "ensemble"
        result["raw_score"] = np.mean([f["raw_score"].to_numpy() for f in pieces], axis=0)
        self.save_df(f"scores_{style}_ensemble", result)
        return result

    def run_replay(self):
        if not self.stage_done("data"):
            raise RuntimeError("Prepare the images and source-derived policy first.")
        self.policy = self.restore_json("data_policy")
        if self.stage_done("replay"):
            checks = self.restore_df("replay_checks")
            if not checks["Passed"].astype(str).str.lower().eq("true").all():
                raise RuntimeError("The saved replay did not pass.")
            print("The saved reference replay passed. No new scoring is needed for this step.")
            return
        self.ensure_images(self.manifest)
        checks = []
        for seed in SEEDS:
            model = self.load_model(seed)
            if seed == SEEDS[0]:
                self.benchmark(model)
            actual = self.score(model, seed, "original")["raw_score"].to_numpy()
            expected = self.reference[str(seed)]["raw_score"].to_numpy()
            delta = float(np.max(np.abs(probability(actual) - probability(expected))))
            checks.append({"Seed": str(seed), "Max raw-score difference": float(np.max(np.abs(actual - expected))),
                           "Max probability difference": delta, "Passed": delta <= REPLAY_PROBABILITY_ATOL,
                           "Changed cutoff decisions": int(((actual >= self.cutoffs[str(seed)]) !=
                                                             (expected >= self.cutoffs[str(seed)])).sum())})
            self.save_df(f"replay_check_seed{seed}", pd.DataFrame([checks[-1]]))
            del model
            gc.collect()
            if not checks[-1]["Passed"]:
                raise RuntimeError("Original source-validation replay changed beyond tolerance. Inspect software and preprocessing before stress inference.")
        actual = self.ensemble("original")["raw_score"].to_numpy()
        expected = self.reference["ensemble"]["raw_score"].to_numpy()
        delta = float(np.max(np.abs(probability(actual) - probability(expected))))
        changed = int(((actual >= self.cutoff) != (expected >= self.cutoff)).sum())
        checks.append({"Seed": "ensemble", "Max raw-score difference": float(np.max(np.abs(actual - expected))),
                       "Max probability difference": delta, "Passed": delta <= REPLAY_PROBABILITY_ATOL and changed == 0,
                       "Changed cutoff decisions": changed})
        self.save_df("replay_checks", pd.DataFrame(checks))
        display(pd.DataFrame(checks))
        if not checks[-1]["Passed"]:
            raise RuntimeError("Ensemble replay or its original decisions changed. Stress inference is stopped.")
        self.finish_stage("replay", ["replay_checks", "replay_timing", "scores_original_ensemble"] +
                          [f"replay_check_seed{seed}" for seed in SEEDS] +
                          [f"scores_original_seed{seed}" for seed in SEEDS])
        print("Reference replay passed in the original validation order and batches. The cutoff remains unchanged.")

    def run_stress(self):
        if not self.stage_done("replay"):
            raise RuntimeError("The original prediction replay must pass before image changes are interpreted.")
        self.policy = self.restore_json("data_policy")
        if self.stage_done("stress"):
            print("All saved appearance predictions are verified; no scoring is repeated.")
            return
        self.ensure_images(self.manifest)
        for seed in SEEDS:
            missing = [style for style in STYLES[1:] if self.restore_df(f"scores_{style}_seed{seed}") is None]
            if not missing:
                continue
            model = self.load_model(seed)
            for style in missing:
                self.score(model, seed, style)
                print(f"Saved and verified {STYLE_NAMES[style]} predictions for seed {seed}.", flush=True)
            del model
            self.tf.keras.backend.clear_session()
            gc.collect()
        for style in STYLES[1:]:
            self.ensemble(style)
        required = [f"scores_{style}_seed{seed}" for style in STYLES[1:] for seed in SEEDS]
        required += [f"scores_{style}_ensemble" for style in STYLES[1:]]
        # Re-register resumed files, validating their identities before marking completion.
        for style in STYLES[1:]:
            for seed in SEEDS:
                self.validate_prediction(self.restore_df(f"scores_{style}_seed{seed}"), seed, style)
        self.finish_stage("stress", required)
        print(f"All {len(STYLES)} appearances are complete. The three saved model seeds were used without retraining.")

    def build_report(self):
        if not self.stage_done("stress"):
            raise RuntimeError("Finish the appearance predictions before producing the report.")
        self.policy = self.restore_json("data_policy")
        if self.stage_done("report"):
            self.show_report()
            return
        summaries, changes, sources, seed_rows, intervals = [], [], [], [], []
        original = self.validate_prediction(self.restore_df("scores_original_ensemble"), "ensemble", "original")
        baseline = original["raw_score"].to_numpy()
        all_scores = {}
        for style in STYLES:
            frame = self.validate_prediction(self.restore_df(f"scores_{style}_ensemble"), "ensemble", style)
            scores = frame["raw_score"].to_numpy()
            all_scores[style] = scores
            summaries.append({"Appearance": STYLE_NAMES[style], "style": style, **metrics(self.manifest.label, scores, self.cutoff)})
            changes.append({"Appearance": STYLE_NAMES[style], **change_counts(self.manifest.label, baseline, scores, self.cutoff)})
            for source, group in self.manifest.groupby("source", sort=True):
                idx = group.index.to_numpy()
                sources.append({"Source": source, "Appearance": STYLE_NAMES[style], **metrics(group.label, scores[idx], self.cutoff)})
            for seed in SEEDS:
                seed_frame = self.validate_prediction(self.restore_df(f"scores_{style}_seed{seed}"), seed, style)
                seed_rows.append({"Seed": seed, "Appearance": STYLE_NAMES[style],
                                  **metrics(self.manifest.label, seed_frame.raw_score, self.cutoffs[str(seed)])})
            if style != "original":
                intervals.append({"Appearance": STYLE_NAMES[style],
                                  **paired_intervals(self.manifest, baseline, scores, self.cutoff)})
        self.save_df("report_summary", pd.DataFrame(summaries))
        self.save_df("report_decision_changes", pd.DataFrame(changes))
        self.save_df("report_by_source", pd.DataFrame(sources))
        self.save_df("report_by_seed", pd.DataFrame(seed_rows))
        self.save_json("report_intervals", intervals)
        baseline_metrics = summaries[0]
        worst = max(summaries[1:], key=lambda row: baseline_metrics["Caught at up to 10% false alarms (%)"] - row["Caught at up to 10% false alarms (%)"])
        verdict = {"run_id": self.run_id, "training_completed": False, "external_confirmation": False,
                   "baseline": baseline_metrics, "largest_observed_ranking_drop": worst,
                   "policy_sha256": canonical_hash(self.policy),
                   "interpretation": ["Compare fixed-cutoff changes with matched-false-alarm discrimination.",
                       "A cutoff shift with preserved ranking suggests a threshold-stability issue.",
                       "A ranking loss identifies a candidate appearance sensitivity for a training experiment.",
                       "No synthetic response proves the real ISIC distribution-shift cause.",
                       "This result cannot establish novelty or predict 90% external sensitivity."],
                   "next_decision": "Review effect sizes, uncertainty, per-seed and per-source results and preview realism before choosing the A/B/C training pilot."}
        self.save_json("report_verdict", verdict)
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
        short_labels = {"original": "Original", "contrast_lower": "Lower\ncontrast", "contrast_low": "Low\ncontrast",
                        "gamma_bright": "Brighter\n(γ=0.75)", "warmer": "Warmer", "cooler": "Cooler",
                        "lower_resolution": "Lower\nresolution"}
        labels = [short_labels[s] for s in STYLES]
        axes[0].plot(labels, [r["Melanomas caught (%)"] for r in summaries], "o-", label="Saved cutoff")
        axes[0].plot(labels, [r["Caught at up to 10% false alarms (%)"] for r in summaries], "s--", label="At up to 10% false alarms")
        axes[0].set(ylabel="Melanomas caught (%)", ylim=(0, 100), title="Detection under each appearance")
        axes[0].legend(fontsize=8)
        axes[1].plot(labels, [r["Benign false alarms (%)"] for r in summaries], "o-", color="#a34738")
        axes[1].axhline(10, color="gray", ls="--", lw=1)
        axes[1].set(ylabel="Benign false alarms (%)", title="False alarms at the saved cutoff")
        self.save_figure("report_performance", figure)
        self.save_examples(baseline, all_scores)
        self.finish_stage("report", ["report_summary", "report_decision_changes", "report_by_source", "report_by_seed",
                          "report_intervals", "report_verdict", "report_performance", "report_examples", "report_example_list"])
        self.show_report()

    def save_examples(self, baseline, scores):
        cases = []
        selected_indices = []
        labels = self.manifest.label.to_numpy()
        for style in STYLES[1:]:
            lost = np.flatnonzero((labels == 1) & (baseline >= self.cutoff) & (scores[style] < self.cutoff))
            if len(lost):
                idx = min(lost, key=lambda k: scores[style][k] - baseline[k])
            else:
                melanoma = np.flatnonzero(labels == 1)
                idx = min(melanoma, key=lambda k: scores[style][k] - baseline[k])
            selected_indices.append(idx)
            row = self.manifest.iloc[idx]
            cases.append({"image_id": row.image_id, "source": row.source, "label": int(row.label), "style": style,
                          "original_score": float(baseline[idx]), "changed_score": float(scores[style][idx]),
                          "selection": "Largest melanoma score loss among newly missed cases, or among all melanomas if none were newly missed."})
        self.ensure_images(self.manifest.iloc[sorted(set(selected_indices))])
        fig, axes = plt.subplots(2, len(cases), figsize=(13, 5), squeeze=False, layout="constrained")
        for i, case in enumerate(cases):
            original = self.images[case["image_id"]][0]
            for axis, pixels, title in ((axes[0, i], correct_rgb(original), "Original"),
                                       (axes[1, i], correct_rgb(transform(original, case["style"], self.policy)), STYLE_NAMES[case["style"]])):
                axis.imshow(pixels); axis.axis("off"); axis.set_title(title, fontsize=8)
            axes[0, i].set_title(case["image_id"] + "\nOriginal melanoma photo", fontsize=8)
            axes[1, i].set_title(f"{STYLE_NAMES[case['style']]}\nScore {case['original_score']:.2f} → {case['changed_score']:.2f}\nCutoff {self.cutoff:.2f}", fontsize=8)
        fig.suptitle("Selected melanoma examples for inspection; these do not estimate performance")
        self.save_figure("report_examples", fig)
        self.save_df("report_example_list", pd.DataFrame(cases))

    def show_report(self):
        summary = self.restore_df("report_summary")
        changes = self.restore_df("report_decision_changes")
        print(f"Main result: all appearances use the same {int(self.manifest.label.sum()):,} melanomas, "
              f"{int((self.manifest.label == 0).sum()):,} benign photos and saved decision cutoff.")
        columns = ["Appearance", "Melanomas caught (%)", "Benign false alarms (%)", "Caught at up to 10% false alarms (%)", "Accuracy (%)", "AUROC"]
        display(summary[columns].round(3))
        display(changes)
        print("The cutoff was not adjusted to make a changed image appearance look better.")
        print("Matched-false-alarm results use a retrospective ROC cutoff for diagnosis only. They are not a new deployed cutoff.")
        print("This check identifies possible weaknesses. It does not train a better model or prove the cause of the ISIC failures.")

    def finalize_and_release(self):
        for stage in REQUIRED_STAGES:
            if not self.stage_done(stage):
                raise RuntimeError(f"{stage} is incomplete. Colab will stay running.")
        if not self.restore_df("replay_checks")["Passed"].astype(str).str.lower().eq("true").all():
            raise RuntimeError("Replay did not pass. Colab will stay running.")
        required = {key: ref for key, ref in self.artifacts.items() if key != "durable_save_receipt"}
        for ref in required.values():
            self.drive.verify_receipt(ref)
            path = self.directory / ref["name"].split("__", 1)[1]
            if path.exists() and (path.stat().st_size != ref["bytes"] or digest(path) != ref["sha256"]):
                raise RuntimeError("A current local output differs from its durable copy. Colab will stay running.")
        for ref in self.models.values():
            self.reader.verify_artifact(ref)
        receipt = {"run_id": self.run_id, "protocol_sha256": canonical_hash(self.protocol), "protocol": self.protocol,
                   "folder_id": self.drive.folder_id, "files": required, "selected_model_references": self.models,
                   "model_folder_id": TRAIN_FOLDER, "computation": "inference only; no new trained checkpoints",
                   "scientific_status": "diagnostic experiment completed; no new external confirmation"}
        self.save_json("durable_save_receipt", receipt)
        restored = self.restore_json("durable_save_receipt")
        if canonical_hash(restored) != canonical_hash(receipt):
            raise RuntimeError("The final receipt readback differs. Colab will stay running.")
        print(f"Predictions, tables, figures, code and hashes verified on Drive: {self.drive.link}", flush=True)
        from google.colab import runtime
        runtime.unassign()
