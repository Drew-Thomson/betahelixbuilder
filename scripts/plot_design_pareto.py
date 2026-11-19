#!/usr/bin/env python3
"""Plot 3D stability vs. handedness-specificity Pareto frontier and surrogate calibration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_design_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load Stage-1 surrogate and Stage-2 3D metrics from ``design-seq`` CSV output.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV produced by ``betahelixbuilder design-seq``.

    Returns
    -------
    e_tgt : np.ndarray
        Stage-2 3D target energy vector, shape ``(M,)``.
    hand_gap : np.ndarray
        Stage-2 3D handedness specificity gap ``E_decoy - E_target``, shape ``(M,)``.
    surr_tgt : np.ndarray
        Stage-1 surrogate target delta energy vs. poly-Ala host, shape ``(M,)``.
    de_3d_host : np.ndarray
        Stage-2 3D delta energy relative to poly-Ala 30-mer host, shape ``(M,)``.
    net_charge : np.ndarray
        Formal integer net charge, shape ``(M,)``.
    pareto : np.ndarray
        Boolean Pareto-optimal indicator vector, shape ``(M,)``.
    ranks : list[int]
        Integer candidate ranks, length ``M``.
    """
    e_tgt_l: list[float] = []
    gap_l: list[float] = []
    surr_l: list[float] = []
    de3d_l: list[float] = []
    q_l: list[int] = []
    pareto_l: list[bool] = []
    ranks: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ranks.append(int(row["rank"]))
            e_tgt_l.append(float(row["e_3d_target"]))
            gap_l.append(float(row["e_3d_hand_gap"]))
            surr_l.append(float(row["surr_target_dE"]))
            de3d_l.append(float(row["dE_3d_vs_ala_host"]))
            q_l.append(int(row["net_charge"]))
            pareto_l.append(row["pareto_optimal"].strip().lower() == "true")

    if not ranks:
        raise ValueError(f"No design rows found in {csv_path}")

    return (
        np.asarray(e_tgt_l, dtype=np.float64),
        np.asarray(gap_l, dtype=np.float64),
        np.asarray(surr_l, dtype=np.float64),
        np.asarray(de3d_l, dtype=np.float64),
        np.asarray(q_l, dtype=np.float64),
        np.asarray(pareto_l, dtype=bool),
        ranks,
    )


def plot_pareto_and_calibration(
    e_tgt: np.ndarray,
    hand_gap: np.ndarray,
    surr_tgt: np.ndarray,
    de_3d_host: np.ndarray,
    net_charge: np.ndarray,
    pareto: np.ndarray,
    ranks: list[int],
    output_path: Path,
    host_energy: float | None,
) -> None:
    """Render two-panel Pareto frontier and surrogate calibration plot.

    Parameters
    ----------
    e_tgt : np.ndarray
        Target 3D energy vector, shape ``(M,)``.
    hand_gap : np.ndarray
        Handedness specificity gap ``E_decoy - E_target``, shape ``(M,)``.
    surr_tgt : np.ndarray
        Stage-1 surrogate target delta energy, shape ``(M,)``.
    de_3d_host : np.ndarray
        Stage-2 3D delta energy relative to host poly-Ala, shape ``(M,)``.
    net_charge : np.ndarray
        Net formal charge per candidate, shape ``(M,)``.
    pareto : np.ndarray
        Boolean mask of Pareto-optimal candidates, shape ``(M,)``.
    ranks : list[int]
        Design rank labels of length ``M``.
    output_path : Path
        Destination PNG file path.
    host_energy : float | None
        Optional poly-(L/D-Ala) 30-mer baseline energy reference line.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=180)
    cmap = plt.get_cmap("viridis")

    q_min = float(np.min(net_charge))
    q_max = float(np.max(net_charge))
    if q_max <= q_min:
        q_max = q_min + 1.0

    sc1 = ax1.scatter(
        e_tgt[~pareto],
        hand_gap[~pareto],
        c=net_charge[~pareto],
        cmap=cmap,
        vmin=q_min,
        vmax=q_max,
        s=50,
        alpha=0.65,
        edgecolors="#333333",
        linewidths=0.5,
        label="Evaluated candidates",
    )
    ax1.scatter(
        e_tgt[pareto],
        hand_gap[pareto],
        c=net_charge[pareto],
        cmap=cmap,
        vmin=q_min,
        vmax=q_max,
        s=95,
        marker="D",
        edgecolors="black",
        linewidths=1.1,
        zorder=4,
        label="Pareto frontier",
    )

    pareto_x = e_tgt[pareto]
    pareto_y = hand_gap[pareto]
    if pareto_x.size > 1:
        order = np.argsort(pareto_x)
        ax1.plot(
            pareto_x[order],
            pareto_y[order],
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    for idx in range(min(5, len(ranks))):
        ax1.annotate(
            f"#{ranks[idx]}",
            (e_tgt[idx], hand_gap[idx]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            fontweight="bold",
        )

    ax1.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, label="No chiral bias (gap = 0)")
    if host_energy is not None:
        ax1.axvline(
            host_energy,
            color="#994f00",
            linestyle="-.",
            linewidth=1.0,
            label=f"Poly-Ala host ({host_energy:.1f})",
        )
    ax1.set_xlabel(r"3D Target Energy $E_{\mathrm{target}}$ (BUDEFF)", fontsize=10)
    ax1.set_ylabel(r"Handedness Gap $E_{\mathrm{decoy}} - E_{\mathrm{target}}$ (BUDEFF)", fontsize=10)
    ax1.set_title("Stability vs. Handedness Specificity (3D)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(loc="best", fontsize=8)
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cbar1.set_label("Net Charge (+e)", fontsize=9)

    sc2 = ax2.scatter(
        surr_tgt,
        de_3d_host,
        c=hand_gap,
        cmap="cividis",
        s=55,
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        label="Stage-2 candidates",
    )
    corr = (
        float(np.corrcoef(surr_tgt, de_3d_host)[0, 1])
        if surr_tgt.size >= 2 and np.std(surr_tgt) > 1e-12 and np.std(de_3d_host) > 1e-12
        else 1.0
    )
    lo = float(min(np.min(surr_tgt), np.min(de_3d_host)))
    hi = float(max(np.max(surr_tgt), np.max(de_3d_host)))
    pad = 0.05 * max(hi - lo, 1.0)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, label=r"$y = x$")
    ax2.set_xlabel(r"Stage-1 Pairwise Surrogate $\Delta E_{\mathrm{target}}$", fontsize=10)
    ax2.set_ylabel(r"Stage-2 3D $\Delta E_{\mathrm{target}}$ vs. Poly-Ala Host", fontsize=10)
    ax2.set_title(f"Surrogate vs. 3D Validation ($r = {corr:.3f}$)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.35)
    ax2.legend(loc="best", fontsize=8)
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label(r"3D Handedness Gap (BUDEFF)", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot stability vs. handedness-specificity Pareto frontier."
    )
    parser.add_argument("csv_path", type=Path, help="Design CSV produced by `design-seq`.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("design_pareto.png"),
        help="Destination PNG image path.",
    )
    parser.add_argument(
        "--host-energy",
        type=float,
        default=None,
        help="Optional poly-Ala 30-mer baseline energy for reference line.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    data = load_design_csv(args.csv_path)
    plot_pareto_and_calibration(*data, output_path=args.output, host_energy=args.host_energy)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
