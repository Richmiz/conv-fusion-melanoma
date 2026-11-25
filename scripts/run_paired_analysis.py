"""CLI entrypoint for shared threshold calibration, cross-cohort evaluation, and paired bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_benchmark_intervals import paired_results
from final_external_benchmark import PRIMARY, shared_cutoffs, summarize_all_datasets


def _add_blend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "blend_60_40" not in out.columns and {"retained", "balanced_mel_nv"}.issubset(out.columns):
        out["blend_60_40"] = 0.60 * out["retained"] + 0.40 * out["balanced_mel_nv"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared 90%%-specificity calibration, external cohort evaluation, and paired bootstrap CIs."
    )
    parser.add_argument("--isic", type=Path, required=True, help="Path to ISIC 2020 prediction CSV.")
    parser.add_argument("--derm", type=Path, required=True, help="Path to DERM12345 prediction CSV.")
    parser.add_argument("--prove", type=Path, required=True, help="Path to PROVe-AI prediction CSV.")
    parser.add_argument("--ph2", type=Path, required=True, help="Path to PH2 prediction CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for output tables.")
    parser.add_argument("--draws", type=int, default=2000, help="Number of paired cluster bootstrap draws.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    isic_df = _add_blend(pd.read_csv(args.isic))
    derm_df = _add_blend(pd.read_csv(args.derm))
    prove_df = _add_blend(pd.read_csv(args.prove))
    ph2_df = _add_blend(pd.read_csv(args.ph2))

    cal = prove_df.loc[prove_df.split.eq("calibration") & prove_df.cohort.eq("nevus")].copy()
    if set(cal.patient_id) & set(prove_df.loc[prove_df.split.eq("test"), "patient_id"]):
        raise ValueError("Calibration and test patients overlap in PROVe-AI predictions.")

    models = [m for m in (*PRIMARY, "committee29", "adae") if m in cal.columns]
    cutoffs = shared_cutoffs(cal, models)
    (args.out_dir / "shared_cutoffs.json").write_text(json.dumps(cutoffs, indent=2), encoding="utf-8")

    mn_df, mimic_df = summarize_all_datasets(cutoffs, isic_df, derm_df, prove_df, ph2_df)
    mn_df.to_csv(args.out_dir / "external_melanoma_vs_nevus.csv", index=False)
    mimic_df.to_csv(args.out_dir / "external_mimic_false_alarms.csv", index=False)

    test_prove = prove_df.loc[prove_df.split.eq("test") & prove_df.cohort.isin(["melanoma", "nevus"])]
    derm_mn = derm_df.loc[derm_df.cohort.isin(["melanoma", "ordinary_nevi"])]
    inputs = {
        "PROVe-AI": (test_prove, test_prove.cohort.eq("melanoma")),
        "DERM12345": (derm_mn, derm_mn.cohort.eq("melanoma")),
        "PH2": (ph2_df, ph2_df.diagnosis.eq("melanoma")),
        "ISIC 2020": (isic_df.loc[isic_df.in_labelled_nevi], isic_df.loc[isic_df.in_labelled_nevi, "label"].eq(1)),
    }
    if "in_low_contrast" in isic_df.columns:
        low_c = isic_df.loc[isic_df.in_low_contrast]
        inputs["ISIC low contrast"] = (low_c, low_c.label.eq(1))

    paired = pd.concat(
        [paired_results(rows, y, cutoffs, name, draws=args.draws) for name, (rows, y) in inputs.items()],
        ignore_index=True,
    )
    paired.to_csv(args.out_dir / "paired_bootstrap_differences.csv", index=False)
    print(f"Saved calibrated cutoffs, cohort summaries, and paired bootstrap CIs to {args.out_dir}")


if __name__ == "__main__":
    main()
