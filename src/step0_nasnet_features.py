"""Step 0: Measure penultimate feature-norm vs. direction shifts on saved NASNetMobile models.

Reuses the exact verified Notebook 09 colour correction (`acquisition_stress.correct_rgb`)
and transform logic (`acquisition_stress.transform`) so Step 0 explains the exact saved
NASNetMobile predictions and verifies zero replay drift before analyzing feature geometry.

Can be imported for pure-NumPy feature decomposition math (shared with ConvNeXt),
or executed as an isolated subprocess so TensorFlow releases 100% of GPU VRAM
before PyTorch initializes CUDA in Step 1.
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from acquisition_stress import (
    IMAGE_SIZE,
    BATCH_SIZE,
    FALSE_ALARM_LIMIT,
    REPLAY_PROBABILITY_ATOL,
    correct_rgb,
    probability,
    transform as nb09_transform,
    uint8 as clamp_u8,
)

STEP0_STYLES = (
    "original",
    "contrast_lower",
    "contrast_low",
    "gamma_bright",
    "lower_resolution",
    "combined_low_contrast_low_res",
)
STEP0_STYLE_NAMES = {
    "original": "Original",
    "contrast_lower": "Lower contrast (0.70x)",
    "contrast_low": "Low contrast (0.55x)",
    "gamma_bright": "Brighter exposure (gamma 0.75)",
    "lower_resolution": "Lower resolution (112px)",
    "combined_low_contrast_low_res": "Combined (0.55x + 112px)",
}


def apply_nb09_stress_224(rgb_224, style, policy):
    """Exact Notebook 09 stress transform on 224x224 RGB (for Step 0 NASNetMobile replay)."""
    if style == "combined_low_contrast_low_res":
        low_c = nb09_transform(rgb_224, "contrast_low", policy)
        return nb09_transform(low_c, "lower_resolution", policy)
    return nb09_transform(rgb_224, style, policy)


def degrade_native_image(native_rgb, style, policy):
    """Create each degraded validation image ONCE from the native decoded photo.

    For `lower_resolution` and `combined_low_contrast_low_res`, returns the shared
    112x112 degraded image so both 224px and 384px models are resized from the exact
    same degraded pixels (0 differing pixels at the 112px stage).
    """
    arr = np.asarray(native_rgb, dtype=np.uint8)
    if style == "original":
        return arr.copy()
    if style in ("contrast_lower", "contrast_low"):
        f = arr.astype(np.float64)
        mean = f.mean(axis=(0, 1), keepdims=True)
        return clamp_u8((f - mean) * float(policy[style]) + mean)
    if style == "gamma_bright":
        gamma = float(policy["gamma_bright"])
        return clamp_u8(255.0 * np.power(arr.astype(np.float64) / 255.0, gamma))
    if style in ("warmer", "cooler"):
        gains = np.asarray(policy[f"{style}_gains"], dtype=np.float64)
        return clamp_u8(arr.astype(np.float64) * gains)
    if style == "lower_resolution":
        side = int(policy["resolution_side"])
        return np.asarray(Image.fromarray(arr).resize((side, side), Image.Resampling.BILINEAR))
    if style == "combined_low_contrast_low_res":
        low_c = degrade_native_image(arr, "contrast_low", policy)
        side = int(policy["resolution_side"])
        return np.asarray(Image.fromarray(low_c).resize((side, side), Image.Resampling.BILINEAR))
    raise ValueError(f"Unknown style: {style}")


def resize_and_correct_shared(degraded_rgb, target_size):
    """Resize a shared degraded image to target_size and apply exact NB09 Shades of Gray."""
    resized = np.asarray(Image.fromarray(degraded_rgb).resize((int(target_size), int(target_size)), Image.Resampling.BILINEAR))
    return correct_rgb(resized)


def locked_cutoff(labels, scores, fpr_limit=FALSE_ALARM_LIMIT):
    """Select the lowest score threshold keeping benign false-alarm rate <= fpr_limit."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        raise ValueError("Cannot select cutoff without benign examples.")
    allowed_fp = int(np.floor(fpr_limit * len(neg)))
    if allowed_fp <= 0:
        return float(np.nextafter(neg[-1], np.inf))
    if allowed_fp >= len(neg):
        return float(np.nextafter(neg[0], -np.inf))
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    valid = np.where(fpr <= fpr_limit + 1e-12)[0]
    best = valid[np.argmax(tpr[valid])]
    t = float(thresholds[best])
    if not np.isfinite(t):
        t = float(np.nextafter(np.max(s), np.inf))
    return t


def summary_metrics(labels, scores, cutoff, fpr_limit=FALSE_ALARM_LIMIT):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    caught = int(np.sum(s[pos] >= cutoff))
    fa = int(np.sum(s[neg] >= cutoff))
    fpr, tpr, _ = roc_curve(y, s)
    matched = float(np.max(tpr[fpr <= fpr_limit + 1e-12])) * 100.0
    return {
        "Melanomas": int(pos.sum()),
        "Benign": int(neg.sum()),
        "Melanomas caught": caught,
        "Benign false alarms": fa,
        "Melanomas caught (%)": 100.0 * caught / max(int(pos.sum()), 1),
        "Benign false alarms (%)": 100.0 * fa / max(int(neg.sum()), 1),
        "Matched <=10% FA sens (%)": matched,
        "AUROC": float(roc_auc_score(y, s)),
    }


def decompose_score_shift(h_orig, h_stress, w, b):
    """Exact algebraic decomposition of linear-head score shifts into norm vs. direction.

    For z = w^T h + b:
        z_stress - z_orig = delta_z_norm + delta_z_dir
    where:
        delta_z_norm = (||h_stress|| / ||h_orig|| - 1) * (w^T h_orig)
        delta_z_dir  = ||h_stress|| * (w^T (h_hat_stress - h_hat_orig))
    """
    h0 = np.asarray(h_orig, dtype=np.float64)
    h1 = np.asarray(h_stress, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    bv = float(b)

    norm0 = np.linalg.norm(h0, axis=1)
    norm1 = np.linalg.norm(h1, axis=1)
    safe_norm0 = np.maximum(norm0, 1e-12)
    safe_norm1 = np.maximum(norm1, 1e-12)
    hat0 = h0 / safe_norm0[:, None]
    hat1 = h1 / safe_norm1[:, None]

    dot0 = h0 @ wv
    dot1 = h1 @ wv
    z0 = dot0 + bv
    z1 = dot1 + bv
    delta_z = z1 - z0

    norm_ratio = norm1 / safe_norm0
    cos_sim = np.sum(hat0 * hat1, axis=1)
    delta_z_norm = (norm_ratio - 1.0) * dot0
    delta_z_dir = norm1 * ((hat1 - hat0) @ wv)

    max_err = float(np.max(np.abs((delta_z_norm + delta_z_dir) - delta_z)))
    if max_err > 1e-7:
        raise RuntimeError(f"Feature decomposition identity violated: max error {max_err:.3e}")

    return {
        "z_orig": z0,
        "z_stress": z1,
        "delta_z": delta_z,
        "norm_orig": norm0,
        "norm_stress": norm1,
        "norm_ratio": norm_ratio,
        "cos_sim": cos_sim,
        "delta_z_norm": delta_z_norm,
        "delta_z_dir": delta_z_dir,
        "max_identity_error": max_err,
    }


def posthoc_cosine_preview(h_orig, h_stress, w, labels, fpr_limit=FALSE_ALARM_LIMIT):
    """Diagnostic-only preview: replace w^T h + b with cos(h, w) locked on original val."""
    h0 = np.asarray(h_orig, dtype=np.float64)
    h1 = np.asarray(h_stress, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    w_hat = wv / max(float(np.linalg.norm(wv)), 1e-12)
    c0 = (h0 / np.maximum(np.linalg.norm(h0, axis=1, keepdims=True), 1e-12)) @ w_hat
    c1 = (h1 / np.maximum(np.linalg.norm(h1, axis=1, keepdims=True), 1e-12)) @ w_hat
    cut = locked_cutoff(labels, c0, fpr_limit=fpr_limit)
    m0 = summary_metrics(labels, c0, cut, fpr_limit=fpr_limit)
    m1 = summary_metrics(labels, c1, cut, fpr_limit=fpr_limit)
    return {
        "cosine_cutoff": cut,
        "orig_sens_pct": m0["Melanomas caught (%)"],
        "orig_fa_pct": m0["Benign false alarms (%)"],
        "orig_auroc": m0["AUROC"],
        "stress_sens_pct": m1["Melanomas caught (%)"],
        "stress_fa_pct": m1["Benign false alarms (%)"],
        "stress_matched_sens_pct": m1["Matched <=10% FA sens (%)"],
        "stress_auroc": m1["AUROC"],
    }


def summarize_feature_geometry(model_label, style, labels, cutoff, h_orig, h_stress, w, b):
    """Build a compact diagnostic row for one model and one stress appearance."""
    y = np.asarray(labels, dtype=int)
    pos = y == 1
    neg = y == 0
    dec = decompose_score_shift(h_orig, h_stress, w, b)
    linear_m = summary_metrics(y, dec["z_stress"], cutoff)
    cos_m = posthoc_cosine_preview(h_orig, h_stress, w, y)

    abs_norm = np.abs(dec["delta_z_norm"])
    abs_dir = np.abs(dec["delta_z_dir"])
    denom = np.maximum(abs_norm + abs_dir, 1e-12)
    norm_share_all = float(np.median(abs_norm / denom))
    norm_share_mel = float(np.median(abs_norm[pos] / denom[pos]))

    return {
        "Model": model_label,
        "Style": style,
        "Appearance": STEP0_STYLE_NAMES.get(style, style),
        "Head bias b": float(b),
        "Head ||w||": float(np.linalg.norm(w)),
        "Median ||h|| ratio (all)": float(np.median(dec["norm_ratio"])),
        "Median ||h|| ratio (mel)": float(np.median(dec["norm_ratio"][pos])),
        "Median ||h|| ratio (benign)": float(np.median(dec["norm_ratio"][neg])),
        "Median cos(h0, h1) (all)": float(np.median(dec["cos_sim"])),
        "Median cos(h0, h1) (mel)": float(np.median(dec["cos_sim"][pos])),
        "Median dz total (mel)": float(np.median(dec["delta_z"][pos])),
        "Median dz norm-term (mel)": float(np.median(dec["delta_z_norm"][pos])),
        "Median dz dir-term (mel)": float(np.median(dec["delta_z_dir"][pos])),
        "Median dz total (benign)": float(np.median(dec["delta_z"][neg])),
        "Median dz norm-term (benign)": float(np.median(dec["delta_z_norm"][neg])),
        "Median dz dir-term (benign)": float(np.median(dec["delta_z_dir"][neg])),
        "Median norm share of |dz| (all)": norm_share_all,
        "Median norm share of |dz| (mel)": norm_share_mel,
        "Linear fixed sens (%)": linear_m["Melanomas caught (%)"],
        "Linear fixed FA (%)": linear_m["Benign false alarms (%)"],
        "Linear matched <=10% FA (%)": linear_m["Matched <=10% FA sens (%)"],
        "Linear AUROC": linear_m["AUROC"],
        "Posthoc cosine fixed sens (%)": cos_m["stress_sens_pct"],
        "Posthoc cosine fixed FA (%)": cos_m["stress_fa_pct"],
        "Posthoc cosine AUROC": cos_m["stress_auroc"],
        "Max identity error": dec["max_identity_error"],
    }


def _extract_tf_features_4tta(feat_model, images_u8, batch_size=BATCH_SIZE):
    """Return 4x-TTA averaged penultimate features and raw scores."""
    n = len(images_u8)
    feats_list, scores_list = [], []
    for start in range(0, n, batch_size):
        batch = images_u8[start:start + batch_size]
        views = (
            batch,
            np.flip(batch, axis=2),
            np.flip(batch, axis=1),
            np.rot90(batch, k=2, axes=(1, 2)),
        )
        f_views, s_views = [], []
        for v in views:
            f_out, s_out = feat_model(np.ascontiguousarray(v), training=False)
            f_views.append(np.asarray(f_out.numpy() if hasattr(f_out, "numpy") else f_out, dtype=np.float32))
            s_views.append(np.asarray(s_out.numpy() if hasattr(s_out, "numpy") else s_out, dtype=np.float32).reshape(-1))
        feats_list.append(np.mean(f_views, axis=0).astype(np.float64))
        scores_list.append(np.mean(s_views, axis=0).astype(np.float64))
    return np.concatenate(feats_list, axis=0), np.concatenate(scores_list, axis=0)


def check_replay_agreement(replayed_scores, reference_scores, cutoff, label):
    """Verify exact NB09 replay criteria: max probability diff <= 1e-4 and 0 changed decisions."""
    s_new = np.asarray(replayed_scores, dtype=np.float64)
    s_ref = np.asarray(reference_scores, dtype=np.float64)
    max_raw_diff = float(np.max(np.abs(s_new - s_ref)))
    max_prob_diff = float(np.max(np.abs(probability(s_new) - probability(s_ref))))
    changed_decisions = int(np.sum((s_new >= cutoff) != (s_ref >= cutoff)))
    passed = (max_prob_diff <= REPLAY_PROBABILITY_ATOL) and (changed_decisions == 0)
    if not passed:
        raise RuntimeError(
            f"Original replay failed for {label}: max_prob_diff={max_prob_diff:.4e} "
            f"(limit {REPLAY_PROBABILITY_ATOL:.1e}), changed_cutoff_decisions={changed_decisions}."
        )
    return {
        "max_raw_score_diff": max_raw_diff,
        "max_prob_diff": max_prob_diff,
        "changed_cutoff_decisions": changed_decisions,
        "passed": passed,
    }


def run_nasnet_subprocess(config_path):
    """Isolated TensorFlow entrypoint so GPU VRAM is released completely on exit."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("float32")
    cfg = json.loads(Path(config_path).read_text(encoding="utf8"))
    val_raw = np.load(cfg["val_raw_224_path"], mmap_mode="r")
    labels = np.asarray(cfg["labels"], dtype=int)
    policy = cfg["policy"]
    styles = list(cfg["styles"])
    cutoffs = cfg["cutoffs"]
    ref_scores = {k: np.asarray(v, dtype=np.float64) for k, v in cfg["reference_scores"].items()}

    # Precompute exact NB09 Shades-of-Gray corrected uint8 arrays per style
    corrected_by_style = {}
    for style in styles:
        arr = np.empty((len(val_raw), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for i in range(len(val_raw)):
            arr[i] = correct_rgb(apply_nb09_stress_224(val_raw[i], style, policy))
        corrected_by_style[style] = arr

    head_params = {}
    rows = []
    seed_scores_by_style = {style: {} for style in styles}

    for seed_str, model_path in sorted(cfg["model_paths"].items(), key=lambda kv: int(kv[0])):
        tf.keras.backend.clear_session()
        gc.collect()
        model = tf.keras.models.load_model(model_path, compile=False)
        head = model.get_layer("raw_melanoma_score")
        w, b = head.get_weights()
        w = w.reshape(-1).astype(np.float64)
        b_scalar = float(b.reshape(-1)[0])
        in_tensor = model.inputs[0] if isinstance(model.inputs, (list, tuple)) else model.inputs
        penult_tensor = head.input[0] if isinstance(head.input, (list, tuple)) else head.input
        out_tensor = model.output[0] if isinstance(model.output, (list, tuple)) else model.output
        feat_model = tf.keras.Model(inputs=in_tensor, outputs=[penult_tensor, out_tensor])

        h_orig, s_orig = _extract_tf_features_4tta(feat_model, corrected_by_style["original"])
        reconstructed = h_orig @ w + b_scalar
        max_lin_err = float(np.max(np.abs(reconstructed - s_orig)))
        if max_lin_err > 1e-4:
            raise RuntimeError(f"Seed {seed_str} linear head reconstruction mismatch: {max_lin_err:.3e}")

        # Strict NB09 replay verification (probability tolerance <= 1e-4 AND 0 changed cutoff decisions)
        replay = check_replay_agreement(s_orig, ref_scores[seed_str], float(cutoffs[seed_str]), f"seed {seed_str}")

        head_params[seed_str] = {
            "bias": b_scalar,
            "weight_l2_norm": float(np.linalg.norm(w)),
            "feature_dim": int(w.shape[0]),
            "median_dot_orig_all": float(np.median(h_orig @ w)),
            "median_dot_orig_mel": float(np.median((h_orig @ w)[labels == 1])),
            "median_dot_orig_benign": float(np.median((h_orig @ w)[labels == 0])),
            "max_linear_reconstruction_err": max_lin_err,
            **replay,
        }
        seed_scores_by_style["original"][seed_str] = s_orig.tolist()
        rows.append(summarize_feature_geometry(
            f"NASNetMobile seed {seed_str}", "original", labels, float(cutoffs[seed_str]),
            h_orig, h_orig, w, b_scalar,
        ))

        for style in styles:
            if style == "original":
                continue
            h_st, s_st = _extract_tf_features_4tta(feat_model, corrected_by_style[style])
            seed_scores_by_style[style][seed_str] = s_st.tolist()
            rows.append(summarize_feature_geometry(
                f"NASNetMobile seed {seed_str}", style, labels, float(cutoffs[seed_str]),
                h_orig, h_st, w, b_scalar,
            ))
            print(f"  [Step 0] NASNetMobile seed {seed_str} | {STEP0_STYLE_NAMES[style]} done.", flush=True)

        del feat_model, model
        tf.keras.backend.clear_session()
        gc.collect()

    # Verify ensemble replay on original before saving any Step 0 results
    ens_orig = np.mean([np.asarray(v, dtype=np.float64) for v in seed_scores_by_style["original"].values()], axis=0)
    if "ensemble" in ref_scores and "ensemble" in cutoffs:
        head_params["ensemble_replay"] = check_replay_agreement(
            ens_orig, ref_scores["ensemble"], float(cutoffs["ensemble"]), "ensemble",
        )

    ens_out = {
        style: {
            **by_seed,
            "ensemble": np.mean([np.asarray(v, dtype=np.float64) for v in by_seed.values()], axis=0).tolist(),
        }
        for style, by_seed in seed_scores_by_style.items()
    }
    Path(cfg["out_head_params"]).write_text(json.dumps(head_params, indent=2, sort_keys=True) + "\n", encoding="utf8")
    pd.DataFrame(rows).to_csv(cfg["out_summary_csv"], index=False, float_format="%.10g", lineterminator="\n")
    Path(cfg["out_ensemble_scores"]).write_text(json.dumps(ens_out), encoding="utf8")
    print("[Step 0] Replay passed (0 changed cutoff decisions) and NASNetMobile feature geometry saved.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python step0_nasnet_features.py <config.json>")
    run_nasnet_subprocess(sys.argv[1])
