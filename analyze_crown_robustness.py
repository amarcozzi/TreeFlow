"""
analyze_crown_robustness.py  (reviewer #4, options 1 & 2)

From the on-disk evaluation CSVs, produce:

  Table 1 (robust statistics): crown-radius central tendency, real vs generated,
           per genus + global — mean (outlier-sensitive) vs median / trimmed mean.
  Table 2 (sensitivity): crown-radius W1 raw vs. after excluding physically
           impossible measurements, per genus + global, with the exclusion rate.
  Table 3 (failure-rate result): rate of implausible crown-radius measurements
           per genus and per height bin — the tall-conifer degenerate-generation
           rate, a reportable result in its own right.

"Implausible / failed measurement" is defined a priori and physically: a
generated crown radius exceeding the largest crown radius observed anywhere in
the real dataset (T_real). We also cross-check against the per-sample tree
height (a crown radius exceeding the tree's own height is geometrically
impossible). Neither threshold is tuned to the result.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, trim_mean

TF = "evaluation_results/finetune-8-512-16384"
OUT = "evaluation_results/crown_robustness"
TALL_CONIFERS = ["Picea", "Pseudotsuga", "Abies", "Larix"]


def load():
    rf = pd.read_csv(f"{TF}/df_real_features.csv")
    gf = pd.read_csv(f"{TF}/df_gen_features.csv", dtype={"gen_id": str, "real_id": str})
    pp = pd.read_csv(f"{TF}/df_pairs.csv", dtype={"gen_id": str})[["gen_id", "height_m"]]
    gf = gf.merge(pp, on="gen_id", how="left")
    return rf, gf


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rf, gf = load()

    # ---- a priori physical threshold: larger than any real tree's crown radius ----
    T_real = float(rf["max_crown_r"].max())
    gf["impossible"] = (gf["max_crown_r"] > T_real) | (gf["max_crown_r"] > gf["height_m"])

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit(f"Physical plausibility bound T_real = max real crown radius across all "
         f"genera = {T_real:.2f} m")
    emit(f"A generated sample is flagged 'failed/implausible' if crown radius "
         f"> {T_real:.2f} m OR > its own tree height.")
    emit("")

    # genera to show: the tall conifers + a few well-sampled controls + others
    genera = (TALL_CONIFERS
              + [g for g in ["Pinus", "Fagus", "Quercus", "Betula"] if g in set(gf.genus)])

    # ===================== TABLE 1 — robust central tendency =====================
    emit("=" * 78)
    emit("TABLE 1 (Option 1): Crown radius central tendency — real vs generated (m)")
    emit("Mean is outlier-sensitive; median / 10%-trimmed mean are robust.")
    emit("-" * 78)
    emit(f"  {'genus':<12s}{'n_gen':>7s}{'real_med':>9s}{'gen_med':>9s}"
         f"{'real_mean':>10s}{'gen_mean':>10s}{'gen_trim10':>11s}")
    emit("-" * 78)
    def t1_row(label, rsub, gsub):
        return (f"  {label:<12s}{len(gsub):>7d}{np.median(rsub):>9.2f}{np.median(gsub):>9.2f}"
                f"{np.mean(rsub):>10.2f}{np.mean(gsub):>10.2f}"
                f"{trim_mean(gsub, 0.10):>11.2f}")
    for g in genera:
        rsub = rf[rf.genus == g]["max_crown_r"].dropna().values
        gsub = gf[gf.genus == g]["max_crown_r"].dropna().values
        if len(rsub) >= 3 and len(gsub) >= 5:
            emit(t1_row(g, rsub, gsub))
    emit(t1_row("GLOBAL", rf["max_crown_r"].dropna().values,
                gf["max_crown_r"].dropna().values))
    emit("-" * 78)
    emit("")

    # =========== TABLE 2 — sensitivity: W1 raw vs. valid subset + rate ===========
    emit("=" * 78)
    emit("TABLE 2 (Option 2): Crown-radius W1 before/after removing failed fits (m)")
    emit("-" * 78)
    emit(f"  {'genus':<12s}{'n_gen':>7s}{'n_failed':>9s}{'fail_%':>8s}"
         f"{'W1_raw':>9s}{'W1_valid':>10s}")
    emit("-" * 78)
    rows2 = []
    def t2_row(label, rsub, gsub_all, gsub_valid):
        nflag = len(gsub_all) - len(gsub_valid)
        rate = 100.0 * nflag / max(len(gsub_all), 1)
        w_raw = wasserstein_distance(rsub, gsub_all) if len(gsub_all) >= 5 else float("nan")
        w_val = wasserstein_distance(rsub, gsub_valid) if len(gsub_valid) >= 5 else float("nan")
        rows2.append({"genus": label, "n_gen": len(gsub_all), "n_failed": nflag,
                      "fail_pct": round(rate, 2), "w1_raw": round(w_raw, 3),
                      "w1_valid": round(w_val, 3)})
        return (f"  {label:<12s}{len(gsub_all):>7d}{nflag:>9d}{rate:>7.1f}%"
                f"{w_raw:>9.2f}{w_val:>10.3f}")
    for g in genera:
        rsub = rf[rf.genus == g]["max_crown_r"].dropna().values
        gall = gf[gf.genus == g]
        if len(rsub) >= 3 and len(gall) >= 5:
            emit(t2_row(g, rsub, gall["max_crown_r"].dropna().values,
                        gall[~gall.impossible]["max_crown_r"].dropna().values))
    emit(t2_row("GLOBAL", rf["max_crown_r"].dropna().values,
                gf["max_crown_r"].dropna().values,
                gf[~gf.impossible]["max_crown_r"].dropna().values))
    emit("-" * 78)
    emit("")

    # ============ TABLE 3 — failed/degenerate rate by genus & height bin =========
    emit("=" * 78)
    emit("TABLE 3 (Result): Implausible-crown-radius (degenerate generation) rate")
    emit("-" * 78)
    by_g = (gf.groupby("genus")["impossible"].agg(["sum", "count"])
            .assign(rate=lambda d: 100 * d["sum"] / d["count"])
            .sort_values("rate", ascending=False))
    emit("  By genus (rate %, n_failed / n_gen):")
    for g, r in by_g.iterrows():
        if r["count"] >= 5 and r["sum"] > 0:
            emit(f"    {g:<12s}{r['rate']:>6.1f}%   ({int(r['sum'])}/{int(r['count'])})")
    emit("")
    HB_ORDER = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40+"]
    by_h = (gf.groupby("height_bin")["impossible"].agg(["sum", "count"])
            .assign(rate=lambda d: 100 * d["sum"] / d["count"]))
    emit("  By height bin (rate %, n_failed / n_gen):")
    for hb in HB_ORDER:
        if hb in by_h.index:
            r = by_h.loc[hb]
            emit(f"    {hb:<12s}{r['rate']:>6.1f}%   ({int(r['sum'])}/{int(r['count'])})")
    emit("")
    tc = gf[gf.genus.isin(TALL_CONIFERS)]
    emit(f"  Tall conifers ({'+'.join(TALL_CONIFERS)}): "
         f"{100*tc['impossible'].mean():.1f}%  ({int(tc['impossible'].sum())}/{len(tc)})")
    emit(f"  All other genera: "
         f"{100*gf[~gf.genus.isin(TALL_CONIFERS)]['impossible'].mean():.2f}%  "
         f"({int(gf[~gf.genus.isin(TALL_CONIFERS)]['impossible'].sum())}/"
         f"{len(gf[~gf.genus.isin(TALL_CONIFERS)])})")
    emit(f"  Global: {100*gf['impossible'].mean():.2f}%  "
         f"({int(gf['impossible'].sum())}/{len(gf)})")
    emit("=" * 78)

    # save
    by_g.reset_index().to_csv(f"{OUT}/failure_rate_by_genus.csv", index=False)
    by_h.reset_index().to_csv(f"{OUT}/failure_rate_by_height.csv", index=False)
    pd.DataFrame(rows2).to_csv(f"{OUT}/crown_w1_sensitivity.csv", index=False)
    open(f"{OUT}/crown_robustness_tables.txt", "w").write("\n".join(lines))
    print(f"\nSaved tables + CSVs to {OUT}/")


if __name__ == "__main__":
    main()
