"""Paired locked-cutoff differences between the blend and its comparators."""
import numpy as np
import pandas as pd


def paired_results(frame, label, cutoffs, cohort, draws=2000):
    model = "blend_60_40"
    units = "patient_id" if "patient_id" in frame else "image_id"
    if frame[units].isna().any() or frame[units].eq("").any() or frame.image_id.duplicated().any():
        raise ValueError(f"Invalid resampling identifiers for {cohort}")
    cluster, names = pd.factorize(frame[units])
    y = np.asarray(label, dtype=bool)
    rng = np.random.default_rng(23)
    counts = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=draws)
    counts = np.vstack([np.ones(len(names), dtype=int), counts])
    metrics, alerts = {}, {}
    for m in cutoffs:
        if m not in frame:
            continue
        scores = frame[m].to_numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f"Invalid {m} scores for {cohort}")
        pred = scores >= cutoffs[m]
        alerts[m] = pred
        order = np.argsort(scores, kind="stable")
        boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(scores[order]))]
        blocks = []
        for start in range(0, len(counts), 32):
            w = counts[start:start+32, cluster]
            wp, wn = w * y, w * ~y
            p, n = wp.sum(axis=1), wn.sum(axis=1)
            pos = np.add.reduceat(wp[:, order], boundaries, axis=1)
            neg = np.add.reduceat(wn[:, order], boundaries, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                auc = (pos * (np.cumsum(neg, axis=1) - neg / 2)).sum(axis=1) / (p*n)
                blocks.append(np.column_stack(((wp*pred).sum(axis=1)/p, (wn*pred).sum(axis=1)/n, auc)))
        metrics[m] = np.vstack(blocks)
    out = []
    for other in metrics:
        if other == model:
            continue
        diff = metrics[model] - metrics[other]
        for idx, metric in enumerate(("sensitivity", "nevus_false_alarm_rate", "auroc")):
            low, high = np.nanpercentile(diff[1:, idx], [2.5, 97.5])
            row = dict(cohort=cohort, comparator=other, metric=metric, units="fraction",
                       blend_minus_comparator=diff[0, idx], low95=low, high95=high,
                       cluster=units, clusters=len(names), draws=draws)
            if idx < 2:
                group = y if idx == 0 else ~y
                row.update(new_alerts=int((alerts[model] & ~alerts[other] & group).sum()),
                           cleared_alerts=int((~alerts[model] & alerts[other] & group).sum()))
            out.append(row)
    return pd.DataFrame(out)
