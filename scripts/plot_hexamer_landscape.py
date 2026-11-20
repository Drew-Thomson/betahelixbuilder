#!/usr/bin/env python3
"""Plot multi-state association energies and AB-vs-AA/BB hexamer selectivity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_hexamer_tables(
    monomer_csv: Path, hexamer_csv: Path
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load monomer summary and full DE hexamer screening CSV tables.

    Parameters
    ----------
    monomer_csv : Path
        Path to `monomer_screen.csv`.
    hexamer_csv : Path
        Path to `hexamer_screen.csv`.

    Returns
    -------
    Tuple[List[Dict[str, str]], List[Dict[str, str]]]
        Tuple of `(promoted_monomers, hexamer_rows)`.
    """
    with monomer_csv.open(newline="", encoding="utf-8") as fh:
        all_mono = list(csv.DictReader(fh))
    promoted_mono = [
        row
        for row in all_mono
        if row.get("promoted_to_de", "true").strip().lower() == "true"
    ]
    with hexamer_csv.open(newline="", encoding="utf-8") as fh:
        hex_rows = list(csv.DictReader(fh))
    return promoted_mono, hex_rows


def build_state_matrix(
    mono_rows: List[Dict[str, str]], hex_rows: List[Dict[str, str]]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build candidate-by-state energy and minimization-flag matrices.

    Parameters
    ----------
    mono_rows : List[Dict[str, str]]
        Promoted monomer metadata rows (`N` candidates).
    hex_rows : List[Dict[str, str]]
        Multi-state DE rows (`N x S` evaluations).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str], List[str]]
        `(energy_mat, minimizing_mat, y_labels, state_order)` where matrices
        have shape `(N, S)`.
    """
    cand_ids = [int(r["candidate_id"]) for r in mono_rows]
    preferred_order = [
        "C2_AA",
        "C2_BB",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "D1",
        "D2",
        "D3",
    ]
    present_states = list(dict.fromkeys(str(r["state"]) for r in hex_rows))
    states = [s for s in preferred_order if s in present_states] + [
        s for s in present_states if s not in preferred_order
    ]

    lookup: Dict[Tuple[int, str], Tuple[float, bool]] = {}
    for row in hex_rows:
        cid = int(row["candidate_id"])
        st = str(row["state"])
        de = float(row["de_assoc_per_chain"])
        still = row["still_minimizing"].strip().lower() == "true"
        lookup[(cid, st)] = (de, still)

    mat = np.full((len(cand_ids), len(states)), np.nan, dtype=float)
    flags = np.zeros((len(cand_ids), len(states)), dtype=bool)
    for i, cid in enumerate(cand_ids):
        for j, st in enumerate(states):
            if (cid, st) in lookup:
                mat[i, j], flags[i, j] = lookup[(cid, st)]

    y_labels = [
        f"#{int(r['candidate_id']):02d} {r['pattern']}/{r['lumen_mode']}"
        for r in mono_rows
    ]
    return mat, flags, y_labels, states


def plot_hexamer_landscape(
    mono_rows: List[Dict[str, str]],
    hex_rows: List[Dict[str, str]],
    target_state: str,
    output_path: Path,
) -> None:
    """Render two-panel heatmap and AB-vs-AA/BB pore selectivity landscape plot.

    Parameters
    ----------
    mono_rows : List[Dict[str, str]]
        Promoted monomer rows.
    hex_rows : List[Dict[str, str]]
        Multi-state DE rows.
    target_state : str
        Target parallel oligomer label (e.g., ``C6``).
    output_path : Path
        Output PNG image path.
    """
    mat, flags, y_labels, states = build_state_matrix(mono_rows, hex_rows)
    if mat.size == 0:
        raise ValueError("No promoted DE candidates to plot.")

    fig, (ax_heat, ax_scat) = plt.subplots(
        1,
        2,
        figsize=(16.5, max(5.4, 0.85 * len(y_labels) + 2.5)),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    vmax = float(np.nanpercentile(mat, 95))
    vmin = float(np.nanmin(mat))
    im = ax_heat.imshow(mat, cmap="cividis", aspect="auto", vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(im, ax=ax_heat, pad=0.02)
    cbar1.set_label(r"$\Delta E_{\mathrm{assoc/chain}}$ (BUDEFF)", fontsize=10)

    ax_heat.set_xticks(np.arange(len(states)))
    ax_heat.set_xticklabels(states, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_heat.set_yticks(np.arange(len(y_labels)))
    ax_heat.set_yticklabels(y_labels, fontsize=9)
    ax_heat.set_xlabel("Oligomer Symmetry & Homotypic Controls", fontsize=10)
    ax_heat.set_ylabel("Promoted Two-Interface Candidate", fontsize=10)
    ax_heat.set_title(
        r"State Energies $\Delta E_{\mathrm{assoc/chain}}$ (* = still minimizing)",
        fontsize=11,
    )

    mid = 0.5 * (vmin + vmax)
    for i in range(mat.shape[0]):
        best_j = int(np.nanargmin(mat[i]))
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            star = "*" if flags[i, j] else ""
            txt_color = "white" if val < mid else "black"
            weight = "bold" if j == best_j or flags[i, j] else "normal"
            ax_heat.text(
                j,
                i,
                f"{val:+.0f}{star}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
                fontweight=weight,
            )

    x_vals = np.array([float(r["ab_vs_aa_bb_gap"]) for r in mono_rows], dtype=float)
    y_vals = np.array([float(r["target_vs_competitors_gap"]) for r in mono_rows], dtype=float)
    pore_vals = np.array([float(r["target_pore_diameter_a"]) for r in mono_rows], dtype=float)
    valid_pore = np.where(np.isfinite(pore_vals), pore_vals, 8.0)

    sc = ax_scat.scatter(
        x_vals,
        y_vals,
        c=valid_pore,
        cmap="viridis",
        s=155,
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    cbar2 = fig.colorbar(sc, ax=ax_scat, pad=0.02)
    cbar2.set_label(rf"{target_state} Min Pore Diameter ($\mathrm{{\AA}}$)", fontsize=10)

    ax_scat.axvline(0.0, color="#555555", linestyle=":", linewidth=1.1, label="AB = AA/BB")
    ax_scat.axhline(
        0.0,
        color="#8c510a",
        linestyle="--",
        linewidth=1.1,
        label=f"{target_state} = Best Competitor",
    )
    x_max = float(np.nanmax(x_vals))
    x_min = float(np.nanmin(x_vals))
    x_mid = 0.5 * (x_min + x_max)
    for r in mono_rows:
        xv = float(r["ab_vs_aa_bb_gap"])
        yv = float(r["target_vs_competitors_gap"])
        on_right = xv > x_min + 0.72 * (x_max - x_min) if x_max > x_min else xv > x_mid
        ax_scat.annotate(
            f"#{int(r['candidate_id'])} ({r['lumen_mode'][:4]})",
            (xv, yv),
            xytext=(-7 if on_right else 6, -12 if on_right else 5),
            textcoords="offset points",
            ha="right" if on_right else "left",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_scat.set_xlabel(
        rf"$AB$ vs $AA/BB$ Gap ($\min(\Delta E_{{C2\_AA}},\Delta E_{{C2\_BB}}) - \Delta E_{{{target_state}}}$)",
        fontsize=10,
    )
    ax_scat.set_ylabel(
        rf"{target_state} Specificity ($\Delta E_{{\mathrm{{best\,other}}}} - \Delta E_{{{target_state}}}$)",
        fontsize=10,
    )
    ax_scat.set_title(rf"{target_state} Selectivity vs Homotypic Controls & Channel Size", fontsize=11)
    ax_scat.grid(True, linestyle=":", alpha=0.5)
    ax_scat.legend(loc="best", fontsize=8.5, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point for plotting hexamer design landscape."""
    parser = argparse.ArgumentParser(
        description="Plot two-interface hexamer association energy and pore selectivity landscape."
    )
    parser.add_argument("mono_csv", type=Path, help="Path to monomer_screen.csv")
    parser.add_argument("hex_csv", type=Path, help="Path to hexamer_screen.csv")
    parser.add_argument(
        "--target-state", type=str, default="C6", help="Target state label (default C6)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hexamer_landscape.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    mono_rows, hex_rows = load_hexamer_tables(args.mono_csv, args.hex_csv)
    plot_hexamer_landscape(mono_rows, hex_rows, args.target_state, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
