"""Notebook 23: Final external benchmark across ISIC 2020, DERM12345, PROVe-AI, and PH2."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gc
import importlib.util
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from IPython.display import Image as IPyImage, display
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve

from acquisition_stress import canonical_hash, correct_rgb, pixel_hash
from backbone_resolution_study import (
    BATCH_SIZE,
    ISIC2020_ZIP_BYTES,
    ISIC2020_ZIP_URL,
    build_convnext_model,
    download_file_curl,
    score_convnext_4tta,
)
from balanced_mimic_aux_pilot import ConvNeXtDualHead, score_dual_4tta
from benign_mimic_training_pilot import source_cutoff
from inference_store import FolderReader, RemoteZip, digest, fetch_zip_member_bytes
from paired_view_store import ProjectRunStore
import prove_ai_retrained_siim as retrained_siim
import prove_ai_adae as adae
from committee_ph2 import PROVE, PROVE_SHA, competition_weights, score_committee
from final_benchmark_intervals import paired_results
from resolution_augmentation_pilot import SEED, SIZE

NB21_RUN = "7aa18cb9ec6abf4021fb"
NB21_FOLDER = os.getenv("CONV_FUSION_NB21_FOLDER", "")
NB15_RUN = "878abeb20c9fc0bec172"
NB15_FOLDER = os.getenv("CONV_FUSION_NB15_FOLDER", "")
DERM_URL = "https://isic-archive.s3.amazonaws.com/dois/10.34970-705541/derm12345.zip"
DERM_BYTES = 1863258972
DERM_SHA = "432bbbcf9f3f9ff3ae81cfaf740b909a8e697aa811f68298d7946178a6f94a54"

CUTOFFS = {
    "retained": -4.641292393207549,
    "benign_mimics": -3.3330977559089656,
    "balanced_mel_nv": -2.8829455375671382,
    "contrast_gamma": -2.6876800358295436,
    "siim_retrained": 0.047476470470428474,
}
MODEL_LABELS = {
    "adae": "ADAE (image + metadata)",
    "committee29": "Committee-29",
    "retained": "Retained (NB15)",
    "contrast_gamma": "Contrast+Gamma (NB12)",
    "benign_mimics": "Benign-Mimics (NB15)",
    "siim_retrained": "SIIM-Retrained",
    "balanced_mel_nv": "Source-Balanced (NB21)",
    "blend_60_40": "Blend 60/40 (Ret+Bal)",
}
MODEL_COLORS = {
    "adae": "#806000",
    "committee29": "#8754a1",
    "retained": "#1f77b4",
    "contrast_gamma": "#8c564b",
    "benign_mimics": "#2ca02c",
    "siim_retrained": "#7f7f7f",
    "balanced_mel_nv": "#d62728",
    "blend_60_40": "#9467bd",
}
TIMM_VERSION = "1.0.22"
PART_SIZE = 64
TITLE = "23 Final External Benchmark"


def calibrated_blend_cutoff(val_df, w_ret=0.6):
    """Compute the internal calibration cutoff for w_ret * retained + (1 - w_ret) * balanced_mel_nv."""
    cal = val_df[val_df["role"] == "calibration"]
    scores = w_ret * cal["retained"].to_numpy(dtype=np.float64) + (1.0 - w_ret) * cal["balanced_mel_nv"].to_numpy(dtype=np.float64)
    cut = source_cutoff(cal["label"].to_numpy(dtype=int), scores)
    return float(cut["threshold"] if isinstance(cut, dict) else cut)


PRIMARY = ("retained", "balanced_mel_nv", "blend_60_40", "benign_mimics", "siim_retrained")
COMMITTEE = "committee29"


def shared_cutoffs(cal, models):
    if len(cal) != 71 or cal.image_id.duplicated().any() or not cal.cohort.eq("nevus").all() or not cal.split.eq("calibration").all():
        raise ValueError("Expected 71 distinct calibration nevi")
    allowed = len(cal) // 10
    cutoffs = {}
    for model in models:
        scores = cal[model].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Missing calibration scores for {model}")
        cutoffs[model] = float(np.nextafter(np.sort(scores)[::-1][allowed], np.inf))
    return cutoffs


def eval_binary(y, scores, cut):
    """Compute AUROC, locked sensitivity/FA, and retrospective sensitivity at <=10% FA."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = (y == 1), (y == 0)
    if not np.isfinite(s).all() or not pos.any() or not neg.any():
        raise ValueError("Expected finite scores and both diagnostic classes")
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "melanomas": int(pos.sum()),
        "benign": int(neg.sum()),
        "melanomas_detected": int((s[pos] >= cut).sum()),
        "nevi_flagged": int((s[neg] >= cut).sum()),
        "auroc": float(roc_auc_score(y, s)),
        "locked_sens_pct": 100.0 * float((s[pos] >= cut).mean()),
        "locked_fa_pct": 100.0 * float((s[neg] >= cut).mean()),
        "matched_le10_fa_sens_pct": 100.0 * float(np.max(tpr[fpr <= 0.10 + 1e-12])),
    }


def summarize_all_datasets(cutoffs, isic_df, derm_df, prove_df, ph2_df):
    """Build unified melanoma-vs-nevus and mimic/unspecified-benign tables across all 4 external datasets."""
    mel_nv_rows = []
    mimic_rows = []

    def add_mn(dataset, subset, sub_df, y, models):
        for m in models:
            if m not in cutoffs or m not in sub_df.columns:
                continue
            res = eval_binary(y, sub_df[m].to_numpy(dtype=np.float64), cutoffs[m])
            mel_nv_rows.append({"dataset": dataset, "subset": subset, "model": m, "cutoff": cutoffs[m], **res})

    def add_mimic(dataset, subtype, sub_df, models):
        for m in models:
            if m not in cutoffs or m not in sub_df.columns:
                continue
            s = sub_df[m].to_numpy(dtype=np.float64)
            if not np.isfinite(s).all():
                raise ValueError(f"Missing {m} scores for {dataset} {subtype}")
            flagged = (s >= cutoffs[m])
            mimic_rows.append({
                "dataset": dataset,
                "subtype": subtype,
                "model": m,
                "images": int(len(sub_df)),
                "flagged": int(flagged.sum()),
                "fa_pct": 100.0 * float(flagged.mean()),
            })

    # 1. ISIC 2020
    isic_models = ["retained", "contrast_gamma", "benign_mimics", "siim_retrained", "balanced_mel_nv", "blend_60_40"]
    isic_mn = isic_df[(isic_df["label"] == 1) | isic_df["in_labelled_nevi"]]
    add_mn("ISIC 2020", "Labelled Nevi", isic_mn, isic_mn["label"].to_numpy(dtype=int), isic_models)
    isic_lc = isic_df[isic_df["in_low_contrast"]]
    add_mn("ISIC 2020", "Low Contrast", isic_lc, isic_lc["label"].to_numpy(dtype=int), isic_models)
    add_mimic("ISIC 2020", "unspecified_benign", isic_df[isic_df["in_unspecified_benign"] & (isic_df["label"] == 0)], isic_models)

    # 2. DERM12345
    derm_models = ["retained", "benign_mimics", "siim_retrained", "balanced_mel_nv", "blend_60_40"]
    derm_mn = derm_df[derm_df["cohort"].isin(["melanoma", "ordinary_nevi"])]
    add_mn("DERM12345", "All Eligible", derm_mn, derm_mn["cohort"].eq("melanoma").to_numpy(dtype=int), derm_models)
    for st in ("keratosis", "dermatofibroma", "vascular"):
        add_mimic("DERM12345", st, derm_df[derm_df["cohort"] == st], derm_models)

    # 3. PROVe-AI (frozen test split)
    prove_models = ["adae", "retained", "benign_mimics", "siim_retrained", "balanced_mel_nv", "blend_60_40"]
    prove_t = prove_df[prove_df["split"] == "test"]
    prove_mn = prove_t[prove_t["cohort"].isin(["melanoma", "nevus"])]
    add_mn("PROVe-AI", "Frozen Test", prove_mn, prove_mn["cohort"].eq("melanoma").to_numpy(dtype=int), prove_models)
    for st in ("keratosis", "dermatofibroma", "vascular"):
        add_mimic("PROVe-AI", st, prove_t[prove_t["cohort"] == st], prove_models)

    # 4. PH2
    ph2_models = [COMMITTEE, "retained", "benign_mimics", "siim_retrained", "balanced_mel_nv", "blend_60_40"]
    ph2_y = ph2_df["diagnosis"].eq("melanoma").to_numpy(dtype=int)
    add_mn("PH2", "Full Release", ph2_df, ph2_y, ph2_models)

    return pd.DataFrame(mel_nv_rows), pd.DataFrame(mimic_rows)


def render_benchmark_plots(mn_df, mimic_df, out_dir):
    """Render 3 comparative benchmark figures (PNG + PDF) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

    mn = mn_df.copy()
    mn["cohort_label"] = mn["dataset"] + "\n(" + mn["subset"] + ")"
    cohorts = list(dict.fromkeys(mn["cohort_label"]))
    model_order = [m for m in (COMMITTEE, "adae", "retained", "contrast_gamma", "benign_mimics", "siim_retrained", "balanced_mel_nv", "blend_60_40") if m in set(mn["model"])]

    # Figure 1: AUROC and Matched <=10% FA Sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    x = np.arange(len(cohorts))
    width = 0.8 / max(1, len(model_order))
    for i, m in enumerate(model_order):
        sub = mn[mn["model"] == m].set_index("cohort_label")
        auroc_vals = [sub.loc[c, "auroc"] if c in sub.index else np.nan for c in cohorts]
        sens_vals = [sub.loc[c, "matched_le10_fa_sens_pct"] if c in sub.index else np.nan for c in cohorts]
        offset = (i - (len(model_order) - 1) / 2.0) * width
        axes[0].bar(x + offset, auroc_vals, width=width * 0.9, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.9)
        axes[1].bar(x + offset, sens_vals, width=width * 0.9, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.9)
    axes[0].set_ylim(0, 1.01)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("A. External Melanoma vs. Nevus AUROC")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Sensitivity (%) at ≤10% Nevus FA")
    axes[1].set_title("B. Sensitivity at Matched ≤10% Nevus False Alarms")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig1_png, fig1_pdf = out_dir / "external_auroc_and_matched_sens.png", out_dir / "external_auroc_and_matched_sens.pdf"
    fig.savefig(fig1_png, bbox_inches="tight")
    fig.savefig(fig1_pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    # Figure 2: Locked-Cutoff Sensitivity vs Nevus FA% (with 10% cap)
    markers = {"ISIC 2020\n(Labelled Nevi)": "o", "ISIC 2020\n(Low Contrast)": "s", "DERM12345\n(All Eligible)": "^", "PROVe-AI\n(Frozen Test)": "D", "PH2\n(Full Release)": "P"}
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.axvline(10.0, color="#c0392b", linestyle="--", linewidth=1.4, label="10% Nevus FA Cap")
    for m in model_order:
        sub = mn[mn["model"] == m]
        for _, row in sub.iterrows():
            ax.scatter(
                row["locked_fa_pct"], row["locked_sens_pct"],
                color=MODEL_COLORS[m], marker=markers.get(row["cohort_label"], "o"),
                s=75, edgecolors="white", linewidth=0.7, alpha=0.92,
            )
    # Legend entries for models + datasets
    model_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m], markersize=8, label=MODEL_LABELS[m]) for m in model_order]
    ds_handles = [plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor="#444444", markersize=7, label=cl.replace("\n", " ")) for cl, mk in markers.items() if cl in set(mn["cohort_label"])]
    cap_handle = [plt.Line2D([0], [0], color="#c0392b", linestyle="--", linewidth=1.4, label="10% FA Cap")]
    ax.legend(handles=model_handles + ds_handles + cap_handle, frameon=False, fontsize=8, loc="lower right", ncol=2)
    ax.set_xlabel("Locked-Cutoff Nevus False-Alarm Rate (%)")
    ax.set_ylabel("Locked-Cutoff Melanoma Sensitivity (%)")
    ax.set_title("Locked-Cutoff Operating Points Across External Cohorts")
    ax.set_xlim(-0.5, max(23.0, float(mn["locked_fa_pct"].max()) + 2.0))
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    fig2_png, fig2_pdf = out_dir / "locked_operating_points.png", out_dir / "locked_operating_points.pdf"
    fig.savefig(fig2_png, bbox_inches="tight")
    fig.savefig(fig2_pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    # Figure 3: Benign-Mimic & Unspecified-Benign False Alarms
    mim = mimic_df.copy()
    mim["group_label"] = mim["dataset"] + "\n" + mim["subtype"]
    groups = list(dict.fromkeys(mim["group_label"]))
    mim_models = [m for m in model_order if m in set(mim["model"])]
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    gx = np.arange(len(groups))
    gw = 0.8 / max(1, len(mim_models))
    for i, m in enumerate(mim_models):
        sub = mim[mim["model"] == m].set_index("group_label")
        vals = [sub.loc[g, "fa_pct"] if g in sub.index else np.nan for g in groups]
        offset = (i - (len(mim_models) - 1) / 2.0) * gw
        ax.bar(gx + offset, vals, width=gw * 0.9, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.9)
    ax.set_xticks(gx)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Locked-Cutoff False-Alarm Rate (%)")
    ax.set_title("Benign-Mimic & Unspecified-Benign False Alarms Across External Cohorts")
    ax.set_ylim(0, 102)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    fig3_png, fig3_pdf = out_dir / "mimic_false_alarms.png", out_dir / "mimic_false_alarms.pdf"
    fig.savefig(fig3_png, bbox_inches="tight")
    fig.savefig(fig3_pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    return [fig1_png, fig1_pdf, fig2_png, fig2_pdf, fig3_png, fig3_pdf]


def _ph2_labels(text):
    clinical = {}
    for line in text.splitlines():
        cells = line.split("||")
        if len(cells) < 5:
            continue
        image_id = cells[1].strip()
        if re.fullmatch(r"IMD\d{3}", image_id):
            clinical[image_id] = int(cells[3].strip())
    return clinical


class FinalExternalBenchmark:
    """Score missing comparative models on ISIC 2020 (and optional PH2) and unify all 4 external datasets."""

    def __init__(self, service, bundle, code, work="/content/final_external_benchmark", ph2_zip=None):
        import timm
        import torch

        if timm.__version__ != TIMM_VERSION or not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime with timm 1.0.22.")
        self.torch, self.timm = torch, timm
        self.device = torch.device("cuda:0")
        self.bundle, self.code = Path(bundle), Path(code)
        self.ph2_zip = Path(ph2_zip) if ph2_zip else None
        self.run_id = digest(self.bundle)[:20]
        self.root = Path(work) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.drive = ProjectRunStore(service, self.run_id, title=TITLE)
        self.drive.claim_writer()
        self.drive.preflight(required_bytes=128 * 1024**2)
        if not self.ph2_zip or not self.ph2_zip.is_file():
            raise FileNotFoundError("The full PH2 archive is required for the final comparison.")

        self.val_ref = pd.read_csv(self.code / "val_predictions.csv", keep_default_na=False, float_precision="round_trip")
        self.isic_ref = pd.read_csv(self.code / "isic_predictions.csv", keep_default_na=False, float_precision="round_trip").rename(columns={"baseline": "nb11_legacy"})
        self.derm_ref = pd.read_csv(self.code / "derm_predictions.csv", keep_default_na=False, float_precision="round_trip")
        self.prove_ref = pd.read_csv(self.code / "prove_predictions.csv", keep_default_na=False, float_precision="round_trip")
        self.ph2_ref = pd.read_csv(self.code / "ph2_predictions.csv", keep_default_na=False, float_precision="round_trip").rename(columns={"score": "retained"})

        self.cutoffs = {**CUTOFFS, "blend_60_40": calibrated_blend_cutoff(self.val_ref, 0.6)}

        self.nb21 = FolderReader(service, NB21_FOLDER, self.root / "nb21")
        nb21_rec = json.loads(self.nb21.fetch(f"{NB21_RUN}__durable_save_receipt.json").read_bytes())
        if nb21_rec["run_id"] != NB21_RUN:
            raise ValueError("Unexpected balanced model run identity")
        arm1_done = json.loads(self.nb21.fetch(f"{NB21_RUN}__balanced_mel_nv_complete.json").read_bytes())
        self.bal_ref = arm1_done["best_ref"]
        self.nb21.verify_artifact(self.bal_ref)

        self.nb15 = FolderReader(service, NB15_FOLDER, self.root / "nb15")
        nb15_ep3 = json.loads(self.nb15.fetch(f"{NB15_RUN}__benign_mimics_epoch03.json").read_bytes())
        self.mim_ref = nb15_ep3["files"]["selected"]
        self.nb15.verify_artifact(self.mim_ref)

        history = json.loads((self.code / "reference_protocol.json").read_text())
        self.ret_ref = history["models"]["baseline"]
        self.nb15.verify_artifact(self.ret_ref)
        if self.mim_ref["sha256"] != history["models"]["benign_mimics"]["sha256"]:
            raise RuntimeError("Mimic model differs from the archived PROVe calibration model")
        self.protocol = {
            "retained_ref": self.ret_ref,
            "siim_provenance": history["outside_models"]["siim_retrained"],
            "primary_calibration": "Same 71 PROVe calibration nevi for each model. Strict tie-safe budget of seven alerts. Apply unchanged across evaluation sets.",
            "interpretation": "DERM and PROVe informed blend selection. ISIC 2020 is a training source of the SIIM comparator. Its ISIC rows are descriptive only.",
            "adae_scope": "PROVe-AI image and metadata, 18 families x 5 folds, with shared calibration nevi",
            "committee_scope": "PH2 comparison plus the 71 shared calibration nevi",
            "task": "Final unified external benchmark across ISIC 2020, DERM12345, PROVe-AI, and PH2",
            "bundle_sha256": digest(self.bundle),
            "balanced_ref": self.bal_ref,
            "benign_mimics_ref": self.mim_ref,
            "original_cutoffs": self.cutoffs,
            "software": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "timm": timm.__version__,

            },
        }
        self.protocol_sha = canonical_hash(self.protocol)
        self.refs = {}
        self.refs["protocol"] = self.save_json("protocol", {"run_id": self.run_id, "protocol_sha256": self.protocol_sha, "protocol": self.protocol})
        self.refs["bundle"] = self.publish(self.bundle)
        print(f"Run {self.run_id} ready | Original blend cutoff: {self.cutoffs['blend_60_40']:.4f} | Drive: {self.drive.link}", flush=True)

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

    def _load_balanced_model(self):
        ckpt = self.nb21.fetch_artifact(self.bal_ref)
        state = self.torch.load(ckpt, map_location="cpu", weights_only=True)
        model = ConvNeXtDualHead.build(self.torch, self.timm, seed=SEED, pretrained=False)
        model.load_state_dict(state["model"], strict=True)
        return model.to(self.device).eval()

    def _load_mimic_model(self, ref):
        ckpt = self.nb15.fetch_artifact(ref)
        state = self.torch.load(ckpt, map_location="cpu", weights_only=True)
        model = build_convnext_model(self.torch, self.timm, pretrained=False)
        model.load_state_dict(state["model"], strict=True)
        return model.to(self.device).eval()

    def _load_siim_model(self):
        from torchvision import transforms
        if not retrained_siim._installed("geffnet==1.0.2"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "geffnet==1.0.2"], check=True)
        source, provenance = retrained_siim.prepare(self.root / "siim")
        if provenance["checkpoint_sha256"] != self.protocol["siim_provenance"]["checkpoint_sha256"]:
            raise RuntimeError("SIIM weights differ from the shared calibration run")
        spec = importlib.util.spec_from_file_location("prove_ai_siim_isic_release", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prev = Path.cwd()
        force_safe = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        try:
            os.chdir(source.parent)
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
            model = module.SIIMISICClassifier().to(self.device).eval()
        finally:
            os.chdir(prev)
            if force_safe is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = force_safe
        model.enable_augment()
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        return model, tf

    def _score_siim_pils(self, siim_model, siim_tf, pil_images, batch_size=32):
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(pil_images), batch_size):
                x = self.torch.stack([siim_tf(im) for im in pil_images[start:start + batch_size]]).to(self.device)
                batches.append(siim_model(x)[:, siim_model.positive_index].cpu().numpy())
        return np.concatenate(batches)

    def _score_batch(self, bal_model, mim_model, pixels):
        bal_s = score_dual_4tta(self.torch, bal_model, pixels, self.device, BATCH_SIZE)
        mim_s = score_convnext_4tta(self.torch, mim_model, pixels, self.device, BATCH_SIZE)
        return bal_s, mim_s

    def _score_isic(self, bal_model, mim_model, ret_model, siim_model, siim_tf):
        df = self.isic_ref.copy()
        wanted = set(df["image_id"])
        archive_index = None
        parts = []
        for start in range(0, len(df), PART_SIZE):
            chunk = df.iloc[start:start + PART_SIZE]
            stem = f"isic_scores_{start:05d}"
            path = self.root / f"{stem}.csv"
            restored = self.drive.restore(f"{self.run_id}__{path.name}", path)
            if restored:
                part = pd.read_csv(path, float_precision="round_trip")
                if part["image_id"].tolist() == chunk["image_id"].tolist() and {"retained", "siim_retrained"}.issubset(part.columns):
                    parts.append(part)
                    continue
            if archive_index is None:
                print("Indexing remote ISIC 2020 archive...", flush=True)
                with ZipFile(RemoteZip(ISIC2020_ZIP_URL, ISIC2020_ZIP_BYTES)) as archive:
                    archive_index = {
                        PurePosixPath(it.filename).stem: it
                        for it in archive.infolist()
                        if PurePosixPath(it.filename).suffix.lower() == ".jpg" and PurePosixPath(it.filename).stem in wanted
                    }
                if set(archive_index) != wanted:
                    raise RuntimeError("Missing ISIC 2020 images in remote archive.")

            def prep(row):
                raw = fetch_zip_member_bytes(ISIC2020_ZIP_URL, archive_index[row.image_id], ISIC2020_ZIP_BYTES)
                with Image.open(io.BytesIO(raw)) as f:
                    native = ImageOps.exif_transpose(f).convert("RGB")
                if pixel_hash(native) != row.pixel_sha256:
                    raise RuntimeError(f"Decoded pixels differ for {row.image_id}.")
                arr = correct_rgb(np.asarray(native.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))
                return arr, siim_tf(native)

            with ThreadPoolExecutor(max_workers=12) as pool:
                prepped = list(pool.map(prep, chunk.itertuples(index=False)))
            pixels = np.stack([p[0] for p in prepped])
            pils = [p[1] for p in prepped]
            bal_s, mim_s = self._score_batch(bal_model, mim_model, pixels)
            siim_s = self._score_siim_pils(siim_model, lambda tensor: tensor, pils)
            part = pd.DataFrame({
                "image_id": chunk["image_id"].values,
                "retained": score_convnext_4tta(self.torch, ret_model, pixels, self.device, BATCH_SIZE),
                "balanced_mel_nv": bal_s,
                "benign_mimics": mim_s,
                "siim_retrained": siim_s,
            })
            part.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
            self.refs[stem] = self.publish(path)
            parts.append(part)
            print(f"  ISIC 2020 scored {min(start + PART_SIZE, len(df)):,}/{len(df):,}", flush=True)

        scored = pd.concat(parts, ignore_index=True)
        out = df.merge(scored, on="image_id", how="left", validate="one_to_one")
        out["blend_60_40"] = 0.6 * out["retained"] + 0.4 * out["balanced_mel_nv"]
        return out

    def _score_derm_siim(self, siim_model, siim_tf):
        df = self.derm_ref.copy()
        df["blend_60_40"] = 0.6 * df["retained"] + 0.4 * df["balanced_mel_nv"]
        derm_zip = download_file_curl(self.root / "archives" / "derm12345.zip", DERM_URL, expected_bytes=DERM_BYTES, expected_sha256=DERM_SHA)
        parts = []
        with ZipFile(derm_zip) as zf:
            for start in range(0, len(df), PART_SIZE):
                chunk = df.iloc[start:start + PART_SIZE]
                stem = f"derm_siim_{start:05d}"
                path = self.root / f"{stem}.csv"
                restored = self.drive.restore(f"{self.run_id}__{path.name}", path)
                if restored:
                    part = pd.read_csv(path, float_precision="round_trip")
                    if part["image_id"].tolist() == chunk["image_id"].tolist():
                        parts.append(part)
                        continue
                pils = []
                for row in chunk.itertuples(index=False):
                    with Image.open(io.BytesIO(zf.read(row.member))) as img:
                        rgb = ImageOps.exif_transpose(img).convert("RGB")
                    if pixel_hash(rgb) != row.pixel_sha256:
                        raise RuntimeError(f"DERM12345 pixel hash mismatch for {row.image_id}.")
                    pils.append(siim_tf(rgb))
                siim_s = self._score_siim_pils(siim_model, lambda t: t, pils)
                part = pd.DataFrame({"image_id": chunk["image_id"].values, "siim_retrained": siim_s})
                part.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
                self.refs[stem] = self.publish(path)
                parts.append(part)
                print(f"  DERM12345 SIIM scored {min(start + PART_SIZE, len(df)):,}/{len(df):,}", flush=True)
        scored = pd.concat(parts, ignore_index=True)
        return df.merge(scored, on="image_id", how="left", validate="one_to_one")

    def _score_ph2(self, bal_model, mim_model, siim_model, siim_tf):
        df = self.ph2_ref.copy()
        if self.ph2_zip is None or not self.ph2_zip.is_file():
            raise FileNotFoundError("PH2 archive is required")
        print(f"Scoring PH2 from {self.ph2_zip} ({len(df)} images)...", flush=True)
        with ZipFile(self.ph2_zip) as archive:
            members = [n for n in archive.namelist() if PurePosixPath(n).name.lower() == "ph2_dataset.txt"]
            if len(members) != 1 or _ph2_labels(archive.read(members[0]).decode("utf-8-sig", errors="replace")) != _ph2_labels((self.code / "ph2_reference.txt").read_text(encoding="utf-8-sig", errors="replace")):
                raise RuntimeError("PH2 metadata check failed.")
            originals = {
                PurePosixPath(n).stem.upper(): n
                for n in archive.namelist()
                if re.fullmatch(r"IMD\d{3}\.bmp", PurePosixPath(n).name, flags=re.IGNORECASE)
                and PurePosixPath(n).parent.name.lower() == f"{PurePosixPath(n).stem.lower()}_dermoscopic_image"
            }
            pixels, pils = [], []
            for row in df.itertuples(index=False):
                with Image.open(io.BytesIO(archive.read(originals[row.image_id]))) as img:
                    rgb = ImageOps.exif_transpose(img).convert("RGB")
                if pixel_hash(rgb) != row.pixel_sha256:
                    raise RuntimeError(f"PH2 pixel hash mismatch for {row.image_id}.")
                pixels.append(correct_rgb(np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR))))
                pils.append(siim_tf(rgb))
        bal_s, mim_s = self._score_batch(bal_model, mim_model, np.stack(pixels))
        df["balanced_mel_nv"] = bal_s
        df["benign_mimics"] = mim_s
        df["siim_retrained"] = self._score_siim_pils(siim_model, lambda t: t, pils)
        df["blend_60_40"] = 0.6 * df["retained"] + 0.4 * df["balanced_mel_nv"]
        return df

    def _score_adae(self, frame):
        work = self.root / "adae"
        work.mkdir(exist_ok=True)
        receipt = work / "adae_complete.json"
        restore = lambda path: self.drive.restore(f"{self.run_id}__{path.name}", path)
        if restore(receipt):
            saved = json.loads(receipt.read_text())
            if saved["protocol"] != self.protocol_sha:
                raise ValueError("ADAE restore protocol mismatch")
            for name, ref in saved["files"].items():
                self.drive.verify_receipt(ref)
                self.refs[name] = ref
            path = work / "adae_predictions.csv"
            if not restore(path):
                raise RuntimeError("Missing verified ADAE predictions")
            scores = pd.read_csv(path, float_precision="round_trip")
        else:
            cache = self.root / "committee"
            weights = competition_weights(cache)
            families, provenance = adae.prepare(work, weights)
            images = work / "images"
            images.mkdir(exist_ok=True)
            archive = download_file_curl(cache / "prove.zip", PROVE, expected_sha256=PROVE_SHA)
            with ZipFile(archive) as zf:
                for row in frame.itertuples():
                    raw = zf.read(f"images/{row.image_id}.jpg")
                    if hashlib.sha256(raw).hexdigest() != row.jpeg_sha256:
                        raise ValueError(f"ADAE input differs for {row.image_id}")
                    (images / f"{row.image_id}.jpg").write_bytes(raw)
            scores, refs = adae.score(work, weights, images, frame, families, provenance, restore, self.publish)
            refs["adae_provenance"] = self.save_json("adae_provenance", provenance)
            refs["adae_predictions"] = self.save_csv("adae_predictions", scores)
            self.refs.update(refs)
            self.save_json("adae_complete", {"protocol": self.protocol_sha, "files": refs})
        series = scores.set_index("image_id", verify_integrity=True).adae
        if set(series.index) != set(frame.image_id) or not series.between(0, 1).all():
            raise ValueError("Incomplete ADAE probabilities")
        return frame.image_id.map(series)

    def run(self):
        saved = self.root / "durable_save_receipt.json"
        if self.drive.restore(f"{self.run_id}__{saved.name}", saved):
            receipt = json.loads(saved.read_text())
            if receipt["protocol_sha256"] != self.protocol_sha or receipt["run_id"] != self.run_id:
                raise ValueError("Saved run identity differs")
            for ref in receipt["files"].values():
                self.drive.verify_receipt(ref)
            tables = []
            for name in ("unified_external_summary", "unified_mimic_summary"):
                path = self.root / (name + ".csv")
                self.drive.restore(receipt["files"][name]["name"], path)
                tables.append(pd.read_csv(path))
            print(f"Verified completed comparison: {self.drive.link}")
            return tuple(tables)
        # Reuse whole verified cohorts after a runtime interruption.
        frames = {}
        for name in ("isic", "derm", "prove", "ph2"):
            stem = name + "_base"
            path = self.root / (stem + ".csv")
            if self.drive.restore(f"{self.run_id}__{path.name}", path):
                frames[name] = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
                self.refs[stem] = self.publish(path)
        if len(frames) != 4:
            bal = self._load_balanced_model()
            mim = self._load_mimic_model(self.mim_ref)
            ret = self._load_mimic_model(self.ret_ref)
            siim, transform = self._load_siim_model()
            for name in ("prove", "ph2", "derm", "isic"):
                if name in frames:
                    continue
                if name == "prove":
                    frame = self.prove_ref.copy()
                    frame["blend_60_40"] = .6 * frame.retained + .4 * frame.balanced_mel_nv
                elif name == "ph2":
                    frame = self._score_ph2(bal, mim, siim, transform)
                elif name == "derm":
                    frame = self._score_derm_siim(siim, transform)
                else:
                    frame = self._score_isic(bal, mim, ret, siim, transform)
                if frame.image_id.duplicated().any() or not np.isfinite(frame[list(PRIMARY)].to_numpy()).all():
                    raise ValueError(f"Incomplete primary predictions for {name}")
                self.refs[name + "_base"] = self.save_csv(name + "_base", frame)
                frames[name] = frame
            del bal, mim, ret, siim
            gc.collect()
            self.torch.cuda.empty_cache()
        isic_df, derm_df, prove_df, ph2_df = (frames[k] for k in ("isic", "derm", "prove", "ph2"))
        frames = (isic_df, derm_df, prove_df, ph2_df)
        cal = prove_df.loc[prove_df.split.eq("calibration") & prove_df.cohort.eq("nevus")].copy()
        if set(cal.patient_id) & set(prove_df.loc[prove_df.split.eq("test"), "patient_id"]):
            raise ValueError("Calibration and test patients overlap")
        historical, historical_mimics = summarize_all_datasets(self.cutoffs, *frames)
        self.refs["historical"] = self.save_csv("original_cutoff_summary", historical)
        self.refs["historical_mimics"] = self.save_csv("original_cutoff_mimics", historical_mimics)
        base_cuts = shared_cutoffs(cal, PRIMARY)
        base_summary, base_mimics = summarize_all_datasets(base_cuts, *frames)
        self.refs["base_locked_cutoffs"] = self.save_json("base_locked_cutoffs", base_cuts)
        self.refs["base_locked_summary"] = self.save_csv("base_locked_summary", base_summary)
        self.refs["base_locked_mimics"] = self.save_csv("base_locked_mimics", base_mimics)
        prove_df["adae"] = self._score_adae(prove_df)
        cal["adae"] = prove_df.set_index("image_id").adae.reindex(cal.image_id).to_numpy()
        scores = score_committee(self, cal, ph2_df)
        cal[COMMITTEE] = cal.image_id.map(scores)
        ph2_df[COMMITTEE] = ph2_df.image_id.map(scores)
        cuts = shared_cutoffs(cal, (*PRIMARY, COMMITTEE, "adae"))
        self.refs["shared_calibration"] = self.save_csv("shared_calibration", cal)
        self.refs["shared_cutoffs"] = self.save_json("shared_cutoffs", cuts)
        mn_df, mimic_df = summarize_all_datasets(cuts, *frames)
        mn_df["training_exposure"] = np.where(mn_df.dataset.eq("ISIC 2020") & mn_df.model.eq("siim_retrained"), "competition training source", "see cohort provenance")
        mimic_df["training_exposure"] = np.where(mimic_df.dataset.eq("ISIC 2020") & mimic_df.model.eq("siim_retrained"),
                                               "competition training source", "see cohort provenance")
        test = prove_df.loc[prove_df.split.eq("test") & prove_df.cohort.isin(["melanoma", "nevus"])]
        derm = derm_df.loc[derm_df.cohort.isin(["melanoma", "ordinary_nevi"])]
        audit = json.loads((self.code / "overlap_review.json").read_text())
        omitted = prove_df.loc[prove_df.image_id.isin([r["prove_id"] for r in audit["matches"]]), "patient_id"]
        inputs = {
            "PROVe-AI": (test, test.cohort.eq("melanoma")),
            "DERM12345": (derm, derm.cohort.eq("melanoma")),
            "PH2": (ph2_df, ph2_df.diagnosis.eq("melanoma")),
        }
        for name, sub in (("ISIC 2020", isic_df.loc[isic_df.in_labelled_nevi]),
                          ("ISIC low contrast", isic_df.loc[isic_df.in_low_contrast])):
            inputs[name] = sub, sub.label.eq(1)
        intervals = pd.concat([paired_results(rows, y, cuts, name) for name, (rows, y) in inputs.items()])
        intervals["eligibility"] = np.where(intervals.cohort.str.startswith("ISIC") & intervals.comparator.eq("siim_retrained"),
                                            "training-source descriptive only", "see cohort provenance")
        self.refs["paired"] = self.save_csv("paired_locked_differences", intervals)
        checks = []
        for name, sub in (("PROVe possible-overlap patients omitted", test.loc[~test.patient_id.isin(omitted)]),
                          ("DERM published test subset", derm.loc[derm.published_split.eq("test")]),
                          ("DERM full", derm)):
            y = sub.cohort.eq("melanoma")
            for m in (m for m in cuts if m in sub):
                rates = eval_binary(y, sub[m], cuts[m])
                detected = sub.loc[y, m].ge(cuts[m])
                rates["patient_equal_melanoma_image_sensitivity_pct"] = 100*detected.groupby(sub.loc[y, "patient_id"]).mean().mean()
                checks.append(dict(dataset=name, model=m, **rates))
        self.refs["checks"] = self.save_csv("sensitivity_analyses", pd.DataFrame(checks))
        # The comparison plots describe external transfer; retain exposed rows in the CSV only.
        published = pd.DataFrame([dict(model="Lituma Villamar EfficientNet-V2-S", dataset="DERM12345",
            comparison="Published aggregate at authors' BCN cutoff", melanomas=400, nonmelanomas=10830,
            cutoff=0.91015625, auroc=0.923, sensitivity_pct=62.5, specificity_pct=97.8,
            source="https://doi.org/10.5826/dpc.160S1a7666")])
        self.refs["published_comparison"] = self.save_csv("published_comparison", published)
        display(published)
        plot_paths = render_benchmark_plots(mn_df[mn_df.training_exposure != "competition training source"],
                                            mimic_df[mimic_df.training_exposure != "competition training source"],
                                            self.root / "plots")
        for p in plot_paths:
            self.refs[p.stem + "_" + p.suffix.lstrip(".")] = self.publish(p)

        self.refs["isic_predictions"] = self.save_csv("isic_predictions", isic_df)
        self.refs["ph2_predictions"] = self.save_csv("ph2_predictions", ph2_df)
        self.refs["derm_predictions"] = self.save_csv("derm_predictions", derm_df)
        self.refs["prove_predictions"] = self.save_csv("prove_predictions", prove_df)
        self.refs["unified_external_summary"] = self.save_csv("unified_external_summary", mn_df)
        self.refs["unified_mimic_summary"] = self.save_csv("unified_mimic_summary", mimic_df)
        for ref in self.refs.values():
            self.drive.verify_receipt(ref)
        self.refs["durable_save_receipt"] = self.save_json("durable_save_receipt", {
            "run_id": self.run_id, "protocol_sha256": self.protocol_sha, "files": self.refs,
        })

        print("\n=== Unified External Melanoma vs. Nevus Summary ===")
        display(mn_df.round(4))
        print("\n=== Unified Benign-Mimic & Unspecified-Benign False Alarms ===")
        display(mimic_df.round(3))
        for p in plot_paths:
            if p.suffix == ".png":
                display(IPyImage(filename=str(p)))
        print(f"Verified outputs saved on Drive: {self.drive.link}")
        return mn_df, mimic_df
