#!/usr/bin/env python3
"""Plot candidate x oligomer-state association heatmap and right-handed/parallel selectivity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_oligomer_tables(
    mono_csv: Path,
    oligo_csv: Path,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load monomer screening metrics and candidate x oligomer-state score matrices.

    Parameters
    ----------
    mono_csv : Path
        Path to ``monomer_screen.csv``.
    oligo_csv : Path
        Path to ``oligomer_screen.csv``.

    Returns
    -------
    cand_labels : list[str]
        Candidate row labels ``[C01 .. CN]``, length ``N``.
    states : list[str]
        Ordered oligomer state labels ``[C2, D1, ...]``, length ``S``.
    assoc_mat : np.ndarray
        Per-chain association energy matrix ``de_assoc_per_chain``, shape ``(N, S)``.
    still_min_mat : np.ndarray
        Boolean matrix flagging DE runs still decreasing at termination, shape ``(N, S)``.
    mono_gap : np.ndarray
        Monomer 3D right-handed specificity ``E_left - E_right``, shape ``(N,)``.
    par_pref_gap : np.ndarray
        Parallel vs. antiparallel oligomer preference gap, shape ``(N,)``.
    """
    cand_ids: list[int] = []
    cand_labels: list[str] = []
    mono_gap_list: list[float] = []
    par_gap_list: list[float] = []

    with mono_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = int(row["candidate_id"])
            cand_ids.append(cid)
            cand_labels.append(
                f"#{cid:02d} {row['pattern']}/{row['width']}/{row['composition']}"
            )
            mono_gap_list.append(float(row["e_3d_hand_gap"]))
            par_gap_list.append(float(row["parallel_preference_gap"]))

    id_to_row: dict[int, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    states: list[str] = []
    raw_entries: list[tuple[int, int, float, bool]] = []

    with oligo_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = int(row["candidate_id"])
            st = row["state"].strip()
            if st not in states:
                states.append(st)
            if cid not in id_to_row:
                continue
            r_idx = id_to_row[cid]
            s_idx = states.index(st)
            val = float(row["de_assoc_per_chain"])
            still = row["still_minimizing"].strip().lower() == "true"
            raw_entries.append((r_idx, s_idx, val, still))

    n_cand = len(cand_ids)
    n_states = len(states)
    assoc_mat = np.full((n_cand, n_states), np.nan, dtype=np.float64)
    still_min_mat = np.zeros((n_cand, n_states), dtype=bool)
    if raw_entries:
        arr = np.asarray([(r, c, v) for r, c, v, _ in raw_entries], dtype=np.float64)
        rows = arr[:, 0].astype(np.intp)
        cols = arr[:, 1].astype(np.intp)
        assoc_mat[rows, cols] = arr[:, 2]
        flags = np.asarray([flag for _, _, _, flag in raw_entries], dtype=bool)
        still_min_mat[rows, cols] = flags

    return (
        cand_labels,
        states,
        assoc_mat,
        still_min_mat,
        np.asarray(mono_gap_list, dtype=np.float64),
        np.asarray(par_gap_list, dtype=np.float64),
    )


def plot_landscape(
    cand_labels: list[str],
    states: list[str],
    assoc_mat: np.ndarray,
    still_min_mat: np.ndarray,
    mono_gap: np.ndarray,
    par_pref_gap: np.ndarray,
    output_path: Path,
) -> None:
    """Render two-panel oligomer association heatmap and specificity summary plot.

    Parameters
    ----------
    cand_labels : list[str]
        Candidate row labels of length ``N``.
    states : list[str]
        Oligomer state column labels of length ``S``.
    assoc_mat : np.ndarray
        Per-chain association energy matrix, shape ``(N, S)``.
    still_min_mat : np.ndarray
        Boolean non-convergence flag mask, shape ``(N, S)``.
    mono_gap : np.ndarray
        Right-handed monomer energy gap, shape ``(N,)``.
    par_pref_gap : np.ndarray
        Parallel vs. antiparallel association preference gap, shape ``(N,)``.
    output_path : Path
        Output PNG image path.
    """
    n_cand, n_states = assoc_mat.shape
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(max(11.5, 5.5 + 1.0 * n_states), max(4.8, 0.7 * n_cand + 2.0)),
        dpi=180,
    )

    finite = assoc_mat[np.isfinite(assoc_mat)]
    vmin = float(np.min(finite)) if finite.size > 0 else -10.0
    vmax = float(np.percentile(finite, 95.0)) if finite.size > 0 else 5.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#dddddd")
    img = ax1.imshow(assoc_mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax1.set_xticks(np.arange(n_states))
    ax1.set_xticklabels(states, fontsize=9, fontweight="bold")
    ax1.set_yticks(np.arange(n_cand))
    ax1.set_yticklabels(cand_labels, fontsize=8)
    ax1.set_xlabel("Oligomer Symmetry State", fontsize=10)
    ax1.set_ylabel("Shortlisted Design Candidate", fontsize=10)
    ax1.set_title(r"Per-Chain Association Energy $\Delta E_{\mathrm{assoc/chain}}$ (* = still minimizing)", fontsize=10)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(np.nan_to_num(assoc_mat, nan=0.0)))
    lum = 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]

    for i in range(n_cand):
        for j in range(n_states):
            val = assoc_mat[i, j]
            if not np.isfinite(val):
                continue
            txt_color = "#111111" if lum[i, j] > 0.55 else "#f8f8f8"
            suffix = "*" if still_min_mat[i, j] else ""
            ax1.text(
                j,
                i,
                f"{val:+.1f}{suffix}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold" if still_min_mat[i, j] else "normal",
            )

    cbar1 = fig.colorbar(img, ax=ax1, pad=0.02)
    cbar1.set_label(r"$\Delta E_{\mathrm{assoc/chain}}$ (BUDEFF)", fontsize=9)

    best_assoc = np.nanmin(assoc_mat, axis=1) if n_states > 0 else np.zeros(n_cand)
    sc = ax2.scatter(
        mono_gap,
        par_pref_gap,
        c=best_assoc,
        cmap="viridis",
        s=95,
        edgecolors="black",
        linewidths=0.9,
        label="Design candidates",
    )
    for i in range(n_cand):
        ax2.annotate(
            f"#{i + 1}",
            (mono_gap[i], par_pref_gap[i]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            fontweight="bold",
        )
    ax2.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0, label="Parallel = Antiparallel")
    ax2.axvline(0.0, color="#994f00", linestyle="--", linewidth=1.0, label="Right = Left monomer")
    ax2.set_xlabel(r"Monomer Right-Handed Specificity $E_{\mathrm{L}} - E_{\mathrm{R}}$ (BUDEFF)", fontsize=10)
    ax2.set_ylabel(r"Parallel Preference $E_{\mathrm{best,Dn}} - E_{\mathrm{best,Cn}}$ (/chain)", fontsize=10)
    ax2.set_title("Monomer Chiral Specificity vs. Parallel Oligomer Selectivity", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="best", fontsize=8)
    cbar2 = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar2.set_label(r"Best $\Delta E_{\mathrm{assoc/chain}}$", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot beta-helix oligomer workflow landscape.")
    parser.add_argument("mono_csv", type=Path, help="Path to monomer_screen.csv.")
    parser.add_argument("oligo_csv", type=Path, help="Path to oligomer_screen.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oligomer_landscape.png"),
        help="Destination PNG image path.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    data = load_oligomer_tables(args.mono_csv, args.oligo_csv)
    plot_landscape(*data, output_path=args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
