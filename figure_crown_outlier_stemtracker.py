"""
figure_crown_outlier_stemtracker.py

Reviewer #4 evidence: two square single-panel subfigures (one Picea, one
Pseudotsuga) showing a degenerate generated tall conifer with the stem-tracking
polynomial spine (red) overlaid. They are designed for a 1x2 LaTeX subfigure
grid; captions are written to a JSON sidecar, not drawn on the figure.

Visual style matches Appendix B / Figure A5 of the TreeFlow paper:
  - point cloud colored by height (viridis)
  - fitted polynomial spine in red (#d62728)
  - gridded 3D box, unfilled lightgray panes, no tick labels
  - view_init(elev=15, azim=135), serif font
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from stem_tracker import compute_rs_spine
from evaluate import compute_mean_r_per_slice

CLOUD_DIR = "evaluation_results/outlier_clouds"
OUTDIR = "evaluation_results/outlier_figs"
TF = "evaluation_results/finetune-8-512-16384"

# Recognizable-but-diverging generated sample per genus (moderate divergence,
# full vertical extent — chosen over the absolute-worst "pancake" clouds).
CASES = {"picea": "00537_3", "pseudotsuga": "13771_1"}
REAL_MAX_CROWN_R = {"Picea": 3.42, "Pseudotsuga": 6.70}

PAD = 1.08          # cube side = max(cloud+spine extent) * PAD (small margin)


def render(gid, meta, out_pdf):
    cloud_norm = zarr.load(f"{CLOUD_DIR}/{gid}.zarr").astype(np.float64)
    H = float(meta["height_m"])
    cloud = cloud_norm * H

    r, s, spine_raw, px, py = compute_rs_spine(cloud_norm)
    zn = cloud_norm[:, 2]
    zg = np.linspace(zn.min(), zn.max(), 300)
    spine = np.column_stack([px(zg) * H, py(zg) * H, zg * H])

    s_max = s.max() + 1e-6
    sc, meanr = compute_mean_r_per_slice(s, r, s_max)
    max_cr_m = float(meanr.max() * H / 2.0)
    spine_div_m = float(np.max(np.sqrt(px(zg) ** 2 + py(zg) ** 2)) * H / 2.0)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(cloud), min(8000, len(cloud)), replace=False)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cloud[idx, 0], cloud[idx, 1], cloud[idx, 2], c=cloud[idx, 2],
               cmap="viridis", s=1.2, alpha=0.5, rasterized=True)
    ax.plot(spine[:, 0], spine[:, 1], spine[:, 2], color="#d62728", lw=3.0, zorder=10)

    # Cubic box bounding cloud + spine; centered in x/y but with the cloud
    # anchored near the bottom (z grounded), so the diverging spine has the
    # full height of the box to sweep through.
    allpts = np.vstack([cloud, spine])
    mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
    side = float((maxs - mins).max()) * PAD
    xc = (mins[0] + maxs[0]) / 2.0
    yc = (mins[1] + maxs[1]) / 2.0
    z_lo = mins[2] - 0.04 * side
    ax.set_xlim(xc - side / 2, xc + side / 2)
    ax.set_ylim(yc - side / 2, yc + side / 2)
    ax.set_zlim(z_lo, z_lo + side)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=135)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.2)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"max_crown_r_m": round(max_cr_m, 1),
            "spine_divergence_r_m": round(spine_div_m, 1)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    info = pd.read_csv(f"{TF}/df_pairs.csv",
                       dtype={"gen_id": str, "real_id": str}).set_index("gen_id")

    out = {}
    for key, gid in CASES.items():
        row = info.loc[gid]
        meta = {"gen_id": gid, "species": row["species"], "genus": row["genus"],
                "height_m": float(row["height_m"]), "scan_type": row["scan_type"],
                "source_real_tree": str(row["real_id"]).zfill(5)}
        out_pdf = f"{OUTDIR}/figure_appendix_crown_outlier_{key}.pdf"
        stats = render(gid, meta, out_pdf)
        ref = REAL_MAX_CROWN_R.get(meta["genus"])
        meta.update(stats)
        meta["real_genus_max_crown_r_m"] = ref
        meta["figure_file"] = os.path.basename(out_pdf)
        meta["suggested_caption"] = (
            f"Stem-tracking spine fit (red) on a degenerate generated "
            f"{meta['species']} (H = {meta['height_m']:.1f} m, {meta['scan_type']}). "
            f"The generated cloud is physically bounded, but the cubic spine "
            f"diverges from the ill-defined trunk, placing the fitted stem "
            f"~{stats['spine_divergence_r_m']:,.0f} m from the points and inflating "
            f"the measured maximum crown radius to {stats['max_crown_r_m']:,.0f} m "
            f"(real {meta['genus']} maximum: {ref:.1f} m). This is the failure mode "
            f"behind the extreme crown-radius W1 values for tall conifers."
        )
        out[key] = meta
        print(f"{key}: {gid} {meta['species']} H={meta['height_m']:.1f} "
              f"max_cr={stats['max_crown_r_m']:.0f} m spine_div={stats['spine_divergence_r_m']:.0f} m")

    with open(f"{OUTDIR}/figure_appendix_crown_outliers_captions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved 2 subfigures + captions JSON to {OUTDIR}/")


if __name__ == "__main__":
    main()
