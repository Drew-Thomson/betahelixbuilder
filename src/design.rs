//! Sequence design optimizing beta-helix stability and handedness specificity.
//!
//! Uses a two-stage pipeline:
//! 1. **Stage 1 (Pairwise Surrogate):** Multi-start Simulated Annealing over 6-residue or
//!    charge-alternating 12-residue repeating templates scored with the `probe-pairs` table
//!    across both the target helix and its parity-inverted mirror enantiomer.
//! 2. **Stage 2 (3D Verification & Terminal Polish):** Parallel `dASPR` packing + native
//!    BUDEFF scoring of the top Pareto-diverse candidates in both target and mirror-decoy
//!    3D helices, with optional 3D coordinate-descent polishing of N/C terminal residues.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;

use rand::Rng;
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::energy::{BudeForceField, EnergyOptions, EnergyScore, ScoredStructure, internal_energy};
use crate::geometry::build_beta_helix;
use crate::input::{
    CCap, CanonicalResidue, Chirality, Handedness, HelixParams, InputError, NCap, Sequence,
};
use crate::probe::{
    AnchorParity, CHEM_GROUP_ORDER, ProbeConfig, ProbeError, ProbeReport, probe_pair_interactions,
};
use crate::sidechain::{DasprConfig, SidechainError, pack_chain, resolve_daspr_binary};

const FWD_OFFSETS: [isize; 4] = [1, 5, 6, 7];

/// Canonical residue index `[0..20)` in [`CHEM_GROUP_ORDER`].
fn residue_index(aa: CanonicalResidue) -> usize {
    CHEM_GROUP_ORDER
        .iter()
        .position(|&r| r == aa)
        .expect("all CanonicalResidue variants exist in CHEM_GROUP_ORDER")
}

fn residue_from_upper(ch: char) -> Result<CanonicalResidue, DesignError> {
    let (aa, _, _) = CanonicalResidue::from_case_char(ch.to_ascii_uppercase())?;
    Ok(aa)
}

/// Formal side-chain net charge at neutral pH (`+1` for Lys/Arg, `-1` for Asp/Glu).
pub fn residue_formal_charge(aa: CanonicalResidue) -> i32 {
    match aa {
        CanonicalResidue::Lys | CanonicalResidue::Arg => 1,
        CanonicalResidue::Asp | CanonicalResidue::Glu => -1,
        _ => 0,
    }
}

/// True for `Trp` or `Tyr` residues required for UV absorbance / anchor stability.
fn is_trp_or_tyr(aa: CanonicalResidue) -> bool {
    matches!(aa, CanonicalResidue::Trp | CanonicalResidue::Tyr)
}

/// True for aromatic residues (`Phe`, `Tyr`, `Trp`).
fn is_aromatic_fyw(aa: CanonicalResidue) -> bool {
    matches!(
        aa,
        CanonicalResidue::Phe | CanonicalResidue::Tyr | CanonicalResidue::Trp
    )
}

/// True for bulky hydrophobic aliphatic residues (`Val`, `Leu`, `Ile`).
fn is_aliphatic_vli(aa: CanonicalResidue) -> bool {
    matches!(
        aa,
        CanonicalResidue::Val | CanonicalResidue::Leu | CanonicalResidue::Ile
    )
}

/// True for hydrophilic/polar or charged water-solubilizing residues.
fn is_polar_soluble(aa: CanonicalResidue) -> bool {
    matches!(
        aa,
        CanonicalResidue::Ser
            | CanonicalResidue::Thr
            | CanonicalResidue::Asn
            | CanonicalResidue::Gln
            | CanonicalResidue::His
            | CanonicalResidue::Asp
            | CanonicalResidue::Glu
            | CanonicalResidue::Lys
            | CanonicalResidue::Arg
    )
}

/// Repetitive frame parameterization over alternating L/D 6-residue turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameMode {
    /// Strict 6-residue repeat (`A-A-A-...`, all columns satisfy `b_k == a_k`).
    Strict,
    /// Paired 12-mer super-repeat (`A-B-A-B-...`) allowing up to `max_alternating_cols`
    /// columns (`b_k != a_k`), enabling alternating salt bridges across `i -> i+6`.
    Alternating { max_alternating_cols: usize },
}

impl Default for FrameMode {
    fn default() -> Self {
        Self::Alternating {
            max_alternating_cols: 2,
        }
    }
}

/// Compositional and chemical constraints enforced on designed sequences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignConstraints {
    /// Allow Cysteine (`C`/`c`), excluded by default (`false`).
    pub allow_cys: bool,
    /// Allow Methionine (`M`/`m`), excluded by default (`false`).
    pub allow_met: bool,
    /// Allow Arginine (`R`/`r`) (`true` default; set `false` via `--exclude-arg`).
    pub allow_arg: bool,
    /// Allow Proline (`P`/`p`) inside the repeating core frame (default `false`).
    pub allow_pro: bool,
    /// Allow Proline (`P`/`p`) inside the first/last `terminal_window` residues during refinement (default `false`).
    pub allow_terminal_pro: bool,
    /// Number of N-terminal and C-terminal residues eligible for `--refine` polish (default `4`).
    pub terminal_window: usize,
    /// Minimum formal net charge (`+1` default).
    pub min_net_charge: i32,
    /// Maximum formal net charge (`+4` default).
    pub max_net_charge: i32,
    /// Minimum total count of `Trp` + `Tyr` residues across the peptide (`1` default).
    pub min_trp_tyr: usize,
    /// Optional cap on total aromatic (`Phe` + `Tyr` + `Trp`) residues across the peptide.
    pub max_aromatic: Option<usize>,
    /// Optional cap on Histidine (`His`) residues across the peptide.
    pub max_his: Option<usize>,
    /// Optional cap on bulky hydrophobic aliphatic (`Val` + `Leu` + `Ile`) residues across the peptide.
    pub max_aliphatic: Option<usize>,
    /// Disallow continuous vertical `Val`/`Leu`/`Ile` ladders (`col_a[k]` and `col_b[k]` both in `{V,L,I}`).
    pub no_aliphatic_ladders: bool,
    /// Minimum total count of polar/charged water-solubilizing residues (`S,T,N,Q,H,D,E,K,R`).
    pub min_polar: usize,
}

impl Default for DesignConstraints {
    fn default() -> Self {
        Self {
            allow_cys: false,
            allow_met: false,
            allow_arg: true,
            allow_pro: false,
            allow_terminal_pro: false,
            terminal_window: 4,
            min_net_charge: 1,
            max_net_charge: 4,
            min_trp_tyr: 1,
            max_aromatic: None,
            max_his: None,
            max_aliphatic: None,
            no_aliphatic_ladders: false,
            min_polar: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SeqStats {
    net_charge: i32,
    trp_tyr: usize,
    aromatic_fyw: usize,
    aliphatic_vli: usize,
    his_count: usize,
    polar_count: usize,
}

fn compute_seq_stats_iter<I: Iterator<Item = CanonicalResidue>>(residues: I) -> SeqStats {
    let mut s = SeqStats::default();
    for aa in residues {
        s.net_charge += residue_formal_charge(aa);
        if is_trp_or_tyr(aa) {
            s.trp_tyr += 1;
        }
        if is_aromatic_fyw(aa) {
            s.aromatic_fyw += 1;
        }
        if is_aliphatic_vli(aa) {
            s.aliphatic_vli += 1;
        }
        if aa == CanonicalResidue::His {
            s.his_count += 1;
        }
        if is_polar_soluble(aa) {
            s.polar_count += 1;
        }
    }
    s
}

impl DesignConstraints {
    /// Ordered repeating-core residue alphabet permitted by these constraints.
    pub fn core_alphabet(&self) -> Vec<CanonicalResidue> {
        CHEM_GROUP_ORDER
            .iter()
            .copied()
            .filter(|&aa| self.is_allowed_at(aa, false))
            .collect()
    }

    /// Ordered terminal-window residue alphabet permitted during 3D refinement.
    pub fn terminal_alphabet(&self) -> Vec<CanonicalResidue> {
        CHEM_GROUP_ORDER
            .iter()
            .copied()
            .filter(|&aa| self.is_allowed_at(aa, true))
            .collect()
    }

    /// Returns `true` if `aa` is permitted in core (`is_terminal == false`) or terminal positions.
    pub fn is_allowed_at(&self, aa: CanonicalResidue, is_terminal: bool) -> bool {
        match aa {
            CanonicalResidue::Cys => self.allow_cys,
            CanonicalResidue::Met => self.allow_met,
            CanonicalResidue::Arg => self.allow_arg,
            CanonicalResidue::Pro => self.allow_pro || (is_terminal && self.allow_terminal_pro),
            _ => true,
        }
    }

    fn check_stats(&self, s: SeqStats) -> bool {
        s.net_charge >= self.min_net_charge
            && s.net_charge <= self.max_net_charge
            && s.trp_tyr >= self.min_trp_tyr
            && self.max_aromatic.is_none_or(|cap| s.aromatic_fyw <= cap)
            && self.max_his.is_none_or(|cap| s.his_count <= cap)
            && self.max_aliphatic.is_none_or(|cap| s.aliphatic_vli <= cap)
            && s.polar_count >= self.min_polar
    }

    /// Validates whether an indexed [`CHEM_GROUP_ORDER`] sequence obeys all constraints.
    fn satisfies_indices(&self, indices: &[usize]) -> bool {
        let n = indices.len();
        for (idx, &res_idx) in indices.iter().enumerate() {
            let is_term = idx < self.terminal_window || idx + self.terminal_window >= n;
            if !self.is_allowed_at(CHEM_GROUP_ORDER[res_idx], is_term) {
                return false;
            }
        }
        self.check_stats(compute_seq_stats_iter(
            indices.iter().map(|&ix| CHEM_GROUP_ORDER[ix]),
        ))
    }

    /// Validates whether a parsed [`Sequence`] satisfies all sequence-level constraints.
    pub fn satisfies_sequence(&self, seq: &Sequence) -> bool {
        let n = seq.len();
        for (idx, res) in seq.residues.iter().enumerate() {
            let is_term = idx < self.terminal_window || idx + self.terminal_window >= n;
            if !self.is_allowed_at(res.residue, is_term) {
                return false;
            }
        }
        self.check_stats(compute_seq_stats_iter(
            seq.residues.iter().map(|r| r.residue),
        ))
    }
}

/// Fast Stage-1 pairwise energetic surrogate built from `probe-pairs` measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairSurrogateModel {
    /// Host poly-(L/D-Ala) internal energy recorded during the reference pair scan.
    pub reference_e_aa: f64,
    /// Helical handedness under which the `probe-pairs` table was recorded.
    pub data_handedness: Handedness,
    /// Single-site mutation energies `dE_vs_ala(aa)` indexed by `[parity_idx][res_idx]`,
    /// where `parity_idx = 0` is `L` and `1` is `D`.
    single_de: [[f64; 20]; 2],
    /// Pairwise double-mutant couplings `ddE_dmc` indexed by
    /// `[anchor_parity][fwd_offset_idx in {+1,+5,+6,+7}][res_i][res_j]`.
    pair_dde: [[[[f64; 20]; 20]; 4]; 2],
}

impl PairSurrogateModel {
    fn new_empty(reference_e_aa: f64, data_handedness: Handedness) -> Self {
        let mut model = Self {
            reference_e_aa,
            data_handedness,
            single_de: [[0.0; 20]; 2],
            pair_dde: [[[[5.0; 20]; 20]; 4]; 2],
        };
        let ala_idx = residue_index(CanonicalResidue::Ala);
        for parity_slice in &mut model.pair_dde {
            for mat in parity_slice {
                for r in 0..20 {
                    // Single-Ala reference is identically 0.0 by construction.
                    mat[ala_idx][r] = 0.0;
                    mat[r][ala_idx] = 0.0;
                }
            }
        }
        model
    }

    /// Builds a [`PairSurrogateModel`] directly from a [`ProbeReport`].
    pub fn from_probe_report(
        report: &ProbeReport,
        data_handedness: Handedness,
    ) -> Result<Self, DesignError> {
        let mut model = Self::new_empty(report.e_aa, data_handedness);

        let mut have_l = false;
        let mut have_d = false;
        let mut seen_plus1 = false;

        for row in &report.rows {
            let p_idx = match row.anchor_chirality {
                Chirality::L => {
                    have_l = true;
                    0
                }
                Chirality::D => {
                    have_d = true;
                    1
                }
                Chirality::Achiral => continue,
            };
            let ri = residue_index(residue_from_upper(row.res_i_code)?);
            let rj = residue_index(residue_from_upper(row.res_j_code)?);

            if row.res_j_code == 'A' {
                model.single_de[p_idx][ri] = row.e_xa - row.e_aa;
            }
            if let Some(off_idx) = FWD_OFFSETS.iter().position(|&o| o == row.offset) {
                if row.offset == 1 {
                    seen_plus1 = true;
                }
                model.pair_dde[p_idx][off_idx][ri][rj] = row.dde_dmc;
            }
        }

        // If a scan included offset -1 but not +1, populate +1 via the exact opposite-parity transpose.
        if !seen_plus1 {
            for row in &report.rows {
                if row.offset != -1 {
                    continue;
                }
                let partner_p = match row.anchor_chirality {
                    Chirality::L => 1,
                    Chirality::D => 0,
                    Chirality::Achiral => continue,
                };
                let ri = residue_index(residue_from_upper(row.res_i_code)?);
                let rj = residue_index(residue_from_upper(row.res_j_code)?);
                model.pair_dde[partner_p][0][rj][ri] = row.dde_dmc;
            }
        }

        if !have_l || !have_d {
            return Err(DesignError::IncompletePairTable(
                "surrogate requires both L and D anchor rows to evaluate handedness specificity"
                    .to_string(),
            ));
        }

        Ok(model)
    }

    /// Loads a [`PairSurrogateModel`] from a `probe-pairs` CSV file on disk.
    pub fn from_csv(path: &Path, data_handedness: Handedness) -> Result<Self, DesignError> {
        let text = fs::read_to_string(path).map_err(|e| DesignError::Io(e.to_string()))?;
        let mut lines = text.lines();
        let header_line = lines.next().ok_or_else(|| {
            DesignError::IncompletePairTable(format!("empty CSV file '{}'", path.display()))
        })?;
        let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();
        let col = |name: &str| -> Result<usize, DesignError> {
            headers.iter().position(|&h| h == name).ok_or_else(|| {
                DesignError::IncompletePairTable(format!(
                    "missing required CSV column '{name}' in '{}'",
                    path.display()
                ))
            })
        };
        let c_anchor = col("anchor")?;
        let c_off = col("offset")?;
        let c_ri = col("res_i")?;
        let c_rj = col("res_j")?;
        let c_dmc = col("ddE_dmc")?;
        let c_exa = col("e_xa")?;
        let c_eaa = col("e_aa")?;

        let mut model = Self::new_empty(0.0, data_handedness);
        let ala_idx = residue_index(CanonicalResidue::Ala);

        let mut have_l = false;
        let mut have_d = false;
        let mut set_ref = false;
        let mut seen_plus1 = false;
        let mut minus1_rows = Vec::new();

        for (line_idx, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            let parse_f = |c: usize| -> Result<f64, DesignError> {
                fields
                    .get(c)
                    .ok_or_else(|| {
                        DesignError::IncompletePairTable(format!("truncated row {}", line_idx + 2))
                    })?
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| {
                        DesignError::IncompletePairTable(format!(
                            "invalid float on row {}: {e}",
                            line_idx + 2
                        ))
                    })
            };
            let anchor_str = fields.get(c_anchor).map(|s| s.trim()).unwrap_or("");
            let p_idx = match anchor_str {
                "L" => {
                    have_l = true;
                    0
                }
                "D" => {
                    have_d = true;
                    1
                }
                _ => continue,
            };
            let off = fields
                .get(c_off)
                .ok_or_else(|| DesignError::IncompletePairTable("missing offset".to_string()))?
                .trim()
                .parse::<isize>()
                .map_err(|e| DesignError::IncompletePairTable(e.to_string()))?;
            let ri_ch = fields[c_ri]
                .trim()
                .chars()
                .next()
                .ok_or_else(|| DesignError::IncompletePairTable("empty res_i".to_string()))?;
            let rj_ch = fields[c_rj]
                .trim()
                .chars()
                .next()
                .ok_or_else(|| DesignError::IncompletePairTable("empty res_j".to_string()))?;
            let ri = residue_index(residue_from_upper(ri_ch)?);
            let rj = residue_index(residue_from_upper(rj_ch)?);
            let dde = parse_f(c_dmc)?;
            let e_xa = parse_f(c_exa)?;
            let e_aa = parse_f(c_eaa)?;
            if !set_ref {
                model.reference_e_aa = e_aa;
                set_ref = true;
            }
            if rj == ala_idx {
                model.single_de[p_idx][ri] = e_xa - e_aa;
            }
            if let Some(off_idx) = FWD_OFFSETS.iter().position(|&o| o == off) {
                if off == 1 {
                    seen_plus1 = true;
                }
                model.pair_dde[p_idx][off_idx][ri][rj] = dde;
            } else if off == -1 {
                minus1_rows.push((p_idx, ri, rj, dde));
            }
        }

        if !seen_plus1 {
            for (p_idx, ri, rj, dde) in minus1_rows {
                let partner_p = 1 - p_idx;
                model.pair_dde[partner_p][0][rj][ri] = dde;
            }
        }

        if !have_l || !have_d {
            return Err(DesignError::IncompletePairTable(format!(
                "CSV '{}' must contain both L and D anchor entries",
                path.display()
            )));
        }
        Ok(model)
    }

    /// Computes surrogate `(target_delta_vs_ala, decoy_delta_vs_ala)` for an alternating sequence
    /// `residues` (`even` 0-based = `L`, `odd` 0-based = `D`).
    pub fn evaluate_indices(&self, res_indices: &[usize], target_handedness: Handedness) -> (f64, f64) {
        let invert_target = target_handedness != self.data_handedness;
        let eval_for_parity = |invert: bool| -> f64 {
            let n = res_indices.len();
            let mut sum = 0.0;
            let base_shift = usize::from(invert);
            for (i, &ri) in res_indices.iter().enumerate() {
                let p = (i + base_shift) & 1;
                sum += self.single_de[p][ri];
            }
            for (off_idx, &off) in FWD_OFFSETS.iter().enumerate() {
                let step = off as usize;
                if step >= n {
                    continue;
                }
                for i in 0..(n - step) {
                    let p = (i + base_shift) & 1;
                    sum += self.pair_dde[p][off_idx][res_indices[i]][res_indices[i + step]];
                }
            }
            sum
        };
        (
            eval_for_parity(invert_target),
            eval_for_parity(!invert_target),
        )
    }
}

/// Configuration for repetitive stability & handedness-specific beta-helix design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignConfig {
    /// Designed peptide length in residues (default `30`, 5 turns of 6 residues).
    pub length: usize,
    /// N-terminal capping state (default [`NCap::Acetyl`]).
    pub n_cap: NCap,
    /// C-terminal capping state (default [`CCap::Amide`]).
    pub c_cap: CCap,
    /// Target helix backbone parameters (`HelixParams::beta_6_3()` default).
    pub target_params: HelixParams,
    /// Repeating template parameterization (default 12-mer alternating up to 2 cols).
    pub frame_mode: FrameMode,
    /// Chemical, charge, and composition constraints.
    pub constraints: DesignConstraints,
    /// Weight `lambda` balancing handedness specificity (`E_target - lambda * (E_decoy - E_target)`).
    pub specificity_weight: f64,
    /// Number of deterministic Stage-1 Simulated Annealing restarts (default `32`).
    pub sa_restarts: usize,
    /// Iterations per Stage-1 Simulated Annealing trajectory (default `6000`).
    pub sa_steps: usize,
    /// Number of diverse Stage-1 candidates evaluated in full 3D (`dASPR` + `BUDEFF`, default `25`).
    pub top_k_3d: usize,
    /// Run 3D coordinate-descent polish over first/last `terminal_window` residues (default `false`).
    pub refine_termini: bool,
    /// Number of top 3D candidates polished when `refine_termini` is enabled (default `4`).
    pub refine_top_n: usize,
    /// Deterministic PRNG seed for Stage-1 Simulated Annealing.
    pub seed: u64,
    /// Optional Rayon thread count limit.
    pub threads: Option<usize>,
}

impl Default for DesignConfig {
    fn default() -> Self {
        Self {
            length: 30,
            n_cap: NCap::Acetyl,
            c_cap: CCap::Amide,
            target_params: HelixParams::beta_6_3(),
            frame_mode: FrameMode::default(),
            constraints: DesignConstraints::default(),
            specificity_weight: 0.5,
            sa_restarts: 32,
            sa_steps: 6000,
            top_k_3d: 25,
            refine_termini: false,
            refine_top_n: 4,
            seed: 42,
            threads: None,
        }
    }
}

/// Complete Stage-1 and Stage-2 record for a designed beta-helix sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignCandidate {
    pub rank: usize,
    pub sequence_with_caps: String,
    pub sequence_core: String,
    pub frame_a: String,
    pub frame_b: String,
    pub alternating_cols: usize,
    pub net_charge: i32,
    pub trp_tyr_count: usize,
    pub aromatic_count: usize,
    pub aliphatic_count: usize,
    pub polar_count: usize,
    pub surr_target_de_vs_ala: f64,
    pub surr_decoy_de_vs_ala: f64,
    pub surr_hand_gap: f64,
    pub e_3d_target: f64,
    pub e_3d_target_steric: f64,
    pub e_3d_target_desolvation: f64,
    pub e_3d_target_charge: f64,
    pub e_3d_decoy: f64,
    pub e_3d_decoy_steric: f64,
    pub e_3d_decoy_desolvation: f64,
    pub e_3d_decoy_charge: f64,
    pub e_3d_hand_gap: f64,
    pub de_3d_vs_ala_host: f64,
    pub score_3d: f64,
    pub pareto_optimal: bool,
    pub refined_termini: bool,
    #[serde(skip)]
    pub target_pdb: String,
    #[serde(skip)]
    pub decoy_pdb: String,
}

/// Full output report produced by [`design_sequences`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignReport {
    pub host_baseline_sequence: String,
    pub host_3d_target_energy: f64,
    pub host_3d_decoy_energy: f64,
    pub target_handedness: Handedness,
    pub specificity_weight: f64,
    pub stage1_unique_candidates: usize,
    pub candidates: Vec<DesignCandidate>,
}

impl DesignReport {
    /// Formats a concise table of top designed sequences and 3D validation metrics.
    pub fn format_summary(&self, top_n: usize) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Beta-Helix Sequence Design Summary (target={:?}, host={}, E_host_3D={:.3}, decoy_host={:.3}, w={:.2}, stage1_unique={}) ===\n",
            self.target_handedness,
            self.host_baseline_sequence,
            self.host_3d_target_energy,
            self.host_3d_decoy_energy,
            self.specificity_weight,
            self.stage1_unique_candidates,
        ));
        for c in self.candidates.iter().take(top_n) {
            let pareto_tag = if c.pareto_optimal { " [PARETO]" } else { "" };
            let ref_tag = if c.refined_termini { " [REFINED]" } else { "" };
            out.push_str(&format!(
                "  #{:2} {}{}{} | A={} B={} alt={} q={:+} Ar={} Al={} Pol={}\n      3D: E_tgt={:.3} (st={:.2}, de={:.2}, ch={:.2})  E_decoy={:.3}  gap={:+.3}  dE_vs_Ala={:+.3}  score={:.3}\n      Surr: dE_tgt={:+.3}  dE_decoy={:+.3}  gap={:+.3}\n",
                c.rank,
                c.sequence_with_caps,
                pareto_tag,
                ref_tag,
                c.frame_a,
                c.frame_b,
                c.alternating_cols,
                c.net_charge,
                c.aromatic_count,
                c.aliphatic_count,
                c.polar_count,
                c.e_3d_target,
                c.e_3d_target_steric,
                c.e_3d_target_desolvation,
                c.e_3d_target_charge,
                c.e_3d_decoy,
                c.e_3d_hand_gap,
                c.de_3d_vs_ala_host,
                c.score_3d,
                c.surr_target_de_vs_ala,
                c.surr_decoy_de_vs_ala,
                c.surr_hand_gap,
            ));
        }
        out
    }

    /// Exports all Stage-2 ranked candidates as CSV.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "rank,sequence_with_caps,sequence_core,frame_a,frame_b,alternating_cols,net_charge,trp_tyr_count,aromatic_count,aliphatic_count,polar_count,surr_target_dE,surr_decoy_dE,surr_hand_gap,e_3d_target,e_3d_target_steric,e_3d_target_desolvation,e_3d_target_charge,e_3d_decoy,e_3d_decoy_steric,e_3d_decoy_desolvation,e_3d_decoy_charge,e_3d_hand_gap,dE_3d_vs_ala_host,score_3d,pareto_optimal,refined_termini\n",
        );
        for c in &self.candidates {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                c.rank,
                c.sequence_with_caps,
                c.sequence_core,
                c.frame_a,
                c.frame_b,
                c.alternating_cols,
                c.net_charge,
                c.trp_tyr_count,
                c.aromatic_count,
                c.aliphatic_count,
                c.polar_count,
                c.surr_target_de_vs_ala,
                c.surr_decoy_de_vs_ala,
                c.surr_hand_gap,
                c.e_3d_target,
                c.e_3d_target_steric,
                c.e_3d_target_desolvation,
                c.e_3d_target_charge,
                c.e_3d_decoy,
                c.e_3d_decoy_steric,
                c.e_3d_decoy_desolvation,
                c.e_3d_decoy_charge,
                c.e_3d_hand_gap,
                c.de_3d_vs_ala_host,
                c.score_3d,
                c.pareto_optimal,
                c.refined_termini
            ));
        }
        out
    }
}

#[derive(Debug, Clone)]
struct RepeatTemplate {
    col_a: [CanonicalResidue; 6],
    col_b: [CanonicalResidue; 6],
}

impl RepeatTemplate {
    fn alternating_cols(&self) -> usize {
        (0..6).filter(|&k| self.col_a[k] != self.col_b[k]).count()
    }

    fn has_pure_aliphatic_ladder(&self) -> bool {
        (0..6).any(|k| is_aliphatic_vli(self.col_a[k]) && is_aliphatic_vli(self.col_b[k]))
    }

    fn expand_indices(&self, length: usize) -> Vec<usize> {
        (0..length)
            .map(|pos| {
                let col = pos % 6;
                let block = (pos / 6) % 2;
                let aa = if block == 0 {
                    self.col_a[col]
                } else {
                    self.col_b[col]
                };
                residue_index(aa)
            })
            .collect()
    }

    fn expand_sequence(&self, length: usize, n_cap: NCap, c_cap: CCap) -> Sequence {
        indices_to_sequence(&self.expand_indices(length), n_cap, c_cap)
    }

    fn frame_string(cols: &[CanonicalResidue; 6]) -> String {
        cols.iter()
            .enumerate()
            .map(|(i, aa)| {
                let parity = if i % 2 == 0 { Chirality::L } else { Chirality::D };
                let ch = if *aa == CanonicalResidue::Gly {
                    Chirality::Achiral
                } else {
                    parity
                };
                aa.case_char(ch)
            })
            .collect()
    }
}

fn indices_to_sequence(res_indices: &[usize], n_cap: NCap, c_cap: CCap) -> Sequence {
    let mut raw = String::with_capacity(res_indices.len() + 8);
    if n_cap == NCap::Acetyl {
        raw.push_str("Ac-");
    }
    for (i, &idx) in res_indices.iter().enumerate() {
        let aa = CHEM_GROUP_ORDER[idx];
        let parity = if i % 2 == 0 { Chirality::L } else { Chirality::D };
        let ch = if aa == CanonicalResidue::Gly {
            Chirality::Achiral
        } else {
            parity
        };
        raw.push(aa.case_char(ch));
    }
    if c_cap == CCap::Amide {
        raw.push_str("-NH2");
    }
    Sequence::parse(&raw).expect("internally constructed alternating sequence always parses")
}

fn format_with_caps(seq: &Sequence) -> String {
    let mut s = String::new();
    if seq.n_cap == NCap::Acetyl {
        s.push_str("Ac-");
    }
    s.push_str(&seq.to_daspr_string());
    if seq.c_cap == CCap::Amide {
        s.push_str("-NH2");
    }
    s
}

#[derive(Debug, Clone)]
struct Stage1Candidate {
    indices: Vec<usize>,
    frame_a: String,
    frame_b: String,
    alternating_cols: usize,
    net_charge: i32,
    trp_tyr_count: usize,
    aromatic_count: usize,
    aliphatic_count: usize,
    polar_count: usize,
    surr_target_de: f64,
    surr_decoy_de: f64,
    surr_hand_gap: f64,
}

fn validate_design_config(config: &DesignConfig) -> Result<(), DesignError> {
    if config.length < 6 {
        return Err(DesignError::InvalidConfig(format!(
            "designed peptide length must be >= 6, got {}",
            config.length
        )));
    }
    config
        .target_params
        .validate(config.length)
        .map_err(|e| DesignError::InvalidConfig(e.to_string()))?;
    if config.constraints.min_net_charge > config.constraints.max_net_charge {
        return Err(DesignError::InvalidConfig(format!(
            "min_net_charge ({}) exceeds max_net_charge ({})",
            config.constraints.min_net_charge, config.constraints.max_net_charge
        )));
    }
    if config.top_k_3d == 0 {
        return Err(DesignError::InvalidConfig(
            "top_k_3d must be >= 1".to_string(),
        ));
    }
    if config.sa_restarts == 0 || config.sa_steps == 0 {
        return Err(DesignError::InvalidConfig(
            "sa_restarts and sa_steps must be >= 1".to_string(),
        ));
    }
    if let FrameMode::Strict = config.frame_mode {
        // Strict repeat with multiplicity M has charges divisible by M when length is divisible by 6.
        let n_turns = config.length / 6;
        if config.length.is_multiple_of(6) {
            let feasible_multiple = (config.constraints.min_net_charge
                ..=config.constraints.max_net_charge)
                .any(|q| q.rem_euclid(n_turns as i32) == 0);
            if !feasible_multiple {
                return Err(DesignError::InvalidConfig(format!(
                    "strict 6-mer repeat of length {} forces net charge to be a multiple of {}, which cannot satisfy [{}..={}]; use `--frame-mode alternating` or adjust charge bounds",
                    config.length,
                    n_turns,
                    config.constraints.min_net_charge,
                    config.constraints.max_net_charge
                )));
            }
        }
    }
    Ok(())
}

fn build_seed_template(
    rng: &mut ChaCha8Rng,
    config: &DesignConfig,
    core_alpha: &[CanonicalResidue],
) -> Result<RepeatTemplate, DesignError> {
    let max_alt = match config.frame_mode {
        FrameMode::Strict => 0,
        FrameMode::Alternating {
            max_alternating_cols,
        } => max_alternating_cols.min(6),
    };
    let neutrals: Vec<CanonicalResidue> = core_alpha
        .iter()
        .copied()
        .filter(|&a| residue_formal_charge(a) == 0 && a != CanonicalResidue::Gly)
        .collect();
    let positives: Vec<CanonicalResidue> = core_alpha
        .iter()
        .copied()
        .filter(|&a| residue_formal_charge(a) > 0)
        .collect();
    let negatives: Vec<CanonicalResidue> = core_alpha
        .iter()
        .copied()
        .filter(|&a| residue_formal_charge(a) < 0)
        .collect();
    let aromatics: Vec<CanonicalResidue> = core_alpha
        .iter()
        .copied()
        .filter(|&a| is_trp_or_tyr(a))
        .collect();

    if config.constraints.min_trp_tyr > 0 && aromatics.is_empty() {
        return Err(DesignError::InvalidConfig(
            "min_trp_tyr > 0 requires Trp or Tyr in the allowed alphabet".to_string(),
        ));
    }

    for _ in 0..5000 {
        let mut tpl = RepeatTemplate {
            col_a: [*core_alpha.choose(rng).unwrap(); 6],
            col_b: [CanonicalResidue::Ala; 6],
        };
        let polar_safe: Vec<CanonicalResidue> = core_alpha
            .iter()
            .copied()
            .filter(|&a| residue_formal_charge(a) == 0 && !is_aliphatic_vli(a) && !is_aromatic_fyw(a))
            .collect();

        for (i, slot) in tpl.col_a.iter_mut().enumerate() {
            let prefer_safe = (config.constraints.no_aliphatic_ladders
                || config.constraints.max_aliphatic.is_some_and(|c| c < 5)
                || config.constraints.max_aromatic.is_some_and(|c| c < 5))
                && !polar_safe.is_empty();
            if prefer_safe && rng.random_bool(0.85) {
                *slot = *polar_safe.choose(rng).unwrap();
            } else if !neutrals.is_empty() && rng.random_bool(0.7) {
                *slot = *neutrals.choose(rng).unwrap();
            } else {
                *slot = *core_alpha.choose(rng).unwrap();
            }
            tpl.col_b[i] = *slot;
        }

        let alt_col = rng.random_range(0..6);
        let aliph_list: Vec<CanonicalResidue> = core_alpha
            .iter()
            .copied()
            .filter(|&a| is_aliphatic_vli(a))
            .collect();
        if max_alt > 0 && !positives.is_empty() && !negatives.is_empty() {
            let n_alt = rng.random_range(1..=max_alt);
            let mut cols: Vec<usize> = (0..6).collect();
            for j in 0..n_alt {
                let pick = rng.random_range(j..6);
                cols.swap(j, pick);
                let col = cols[j];
                if j == 1 && !aromatics.is_empty() && config.constraints.max_aromatic.is_some_and(|c| c < 5) {
                    let ar = *aromatics.choose(rng).unwrap();
                    let filler = if !aliph_list.is_empty() && rng.random_bool(0.5) {
                        *aliph_list.choose(rng).unwrap()
                    } else if neutrals.contains(&CanonicalResidue::Gln) {
                        CanonicalResidue::Gln
                    } else {
                        CanonicalResidue::Ala
                    };
                    tpl.col_a[col] = filler;
                    tpl.col_b[col] = ar; // sparse alternating aromatic (2 copies, or paired with Leu/Gln)
                } else if rng.random_bool(0.5) {
                    tpl.col_a[col] = *positives.choose(rng).unwrap();
                    tpl.col_b[col] = *negatives.choose(rng).unwrap();
                } else {
                    tpl.col_a[col] = *negatives.choose(rng).unwrap();
                    tpl.col_b[col] = *positives.choose(rng).unwrap();
                }
            }
        }

        if config.constraints.min_trp_tyr > 0 {
            let idx = tpl.expand_indices(config.length);
            let stats = compute_seq_stats_iter(idx.iter().map(|&ix| CHEM_GROUP_ORDER[ix]));
            if stats.trp_tyr < config.constraints.min_trp_tyr {
                let ar = *aromatics.choose(rng).unwrap();
                if max_alt >= 2 && config.constraints.max_aromatic.is_some_and(|c| c < 5) {
                    let c = (alt_col + 1) % 6;
                    tpl.col_a[c] = CanonicalResidue::Gln;
                    tpl.col_b[c] = ar;
                } else {
                    let col = rng.random_range(0..6);
                    tpl.col_a[col] = ar;
                    tpl.col_b[col] = ar;
                }
            }
        }

        let seq = tpl.expand_sequence(config.length, config.n_cap, config.c_cap);
        if tpl.alternating_cols() <= max_alt
            && !(config.constraints.no_aliphatic_ladders && tpl.has_pure_aliphatic_ladder())
            && config.constraints.satisfies_sequence(&seq)
        {
            return Ok(tpl);
        }
    }

    // Deterministic fallback template mixing polar fillers (`Gln`/`Asn`/`Ser`), positive/negative salt bridge, and sparse `Trp`/`Tyr`.
    let polar_fillers = [
        CanonicalResidue::Gln,
        CanonicalResidue::Asn,
        CanonicalResidue::Ser,
        CanonicalResidue::Thr,
    ];
    for &pos_aa in &positives {
        for &neg_aa in &negatives {
            for &ar in &aromatics {
                for &pol1 in &polar_fillers {
                    for &pol2 in &[CanonicalResidue::Asn, CanonicalResidue::Ser, CanonicalResidue::Gln] {
                        let mut tpl = RepeatTemplate {
                            col_a: [
                                CanonicalResidue::Gln,
                                pol2,
                                pol1,
                                CanonicalResidue::Thr,
                                pos_aa,
                                CanonicalResidue::Asn,
                            ],
                            col_b: [
                                CanonicalResidue::Gln,
                                pol2,
                                pol1,
                                CanonicalResidue::Thr,
                                pos_aa,
                                CanonicalResidue::Asn,
                            ],
                        };
                        if max_alt >= 1 {
                            tpl.col_b[4] = neg_aa;
                        }
                        if config.constraints.max_aromatic.is_some_and(|c| c < 5) && max_alt >= 2 {
                            tpl.col_b[0] = ar;
                        } else {
                            tpl.col_a[0] = ar;
                            tpl.col_b[0] = ar;
                        }
                        if config.constraints.min_net_charge >= 5 {
                            tpl.col_b[4] = pos_aa;
                        }
                        let seq = tpl.expand_sequence(config.length, config.n_cap, config.c_cap);
                        if tpl.alternating_cols() <= max_alt
                            && !(config.constraints.no_aliphatic_ladders
                                && tpl.has_pure_aliphatic_ladder())
                            && config.constraints.satisfies_sequence(&seq)
                        {
                            return Ok(tpl);
                        }
                    }
                }
            }
        }
    }

    Err(DesignError::InvalidConfig(
        "could not construct a feasible initial repeat template satisfying constraints"
            .to_string(),
    ))
}

fn mutate_template(
    rng: &mut ChaCha8Rng,
    tpl: &RepeatTemplate,
    max_alt: usize,
    core_alpha: &[CanonicalResidue],
) -> RepeatTemplate {
    let mut next = tpl.clone();
    let move_type = if max_alt == 0 {
        if rng.random_bool(0.85) { 0 } else { 3 }
    } else {
        let p: f64 = rng.random();
        if p < 0.45 {
            0 // lock-mutate both A[k] and B[k] identically
        } else if p < 0.78 {
            1 // mutate A[k] or B[k] independently (staying within max_alt)
        } else if p < 0.90 {
            2 // collapse an alternating column back to A[k] == B[k] or copy a complementary pair
        } else {
            3 // swap two same-parity columns
        }
    };

    match move_type {
        0 => {
            let col = rng.random_range(0..6);
            let aa = *core_alpha.choose(rng).unwrap();
            if next.col_a[col] == next.col_b[col] || rng.random_bool(0.5) {
                next.col_a[col] = aa;
                next.col_b[col] = aa;
            } else if rng.random_bool(0.5) {
                next.col_a[col] = aa;
            } else {
                next.col_b[col] = aa;
            }
        }
        1 => {
            let alt_cols: Vec<usize> = (0..6)
                .filter(|&k| next.col_a[k] != next.col_b[k])
                .collect();
            let col = if alt_cols.len() >= max_alt && !alt_cols.is_empty() {
                *alt_cols.choose(rng).unwrap()
            } else {
                rng.random_range(0..6)
            };
            let aa = *core_alpha.choose(rng).unwrap();
            if rng.random_bool(0.5) {
                next.col_a[col] = aa;
            } else {
                next.col_b[col] = aa;
            }
        }
        2 => {
            let alt_cols: Vec<usize> = (0..6)
                .filter(|&k| next.col_a[k] != next.col_b[k])
                .collect();
            if let Some(&col) = alt_cols.choose(rng) {
                if rng.random_bool(0.5) {
                    next.col_b[col] = next.col_a[col];
                } else {
                    next.col_a[col] = next.col_b[col];
                }
            }
        }
        _ => {
            let parity = rng.random_range(0..2);
            let candidates = [parity, parity + 2, parity + 4];
            let c1 = candidates[rng.random_range(0..3)];
            let c2 = candidates[rng.random_range(0..3)];
            next.col_a.swap(c1, c2);
            next.col_b.swap(c1, c2);
        }
    }
    next
}

fn mark_pareto_flags<T, F, G>(items: &[T], get_target: F, get_gap: G) -> Vec<bool>
where
    F: Fn(&T) -> f64,
    G: Fn(&T) -> f64,
{
    let mut pareto = vec![true; items.len()];
    for (i, item_i) in items.iter().enumerate() {
        let ti = get_target(item_i);
        let gi = get_gap(item_i);
        for (j, item_j) in items.iter().enumerate() {
            if i == j {
                continue;
            }
            let tj = get_target(item_j);
            let gj = get_gap(item_j);
            if tj <= ti + 1e-12 && gj >= gi - 1e-12 && (tj < ti - 1e-12 || gj > gi + 1e-12) {
                pareto[i] = false;
                break;
            }
        }
    }
    pareto
}

fn run_stage1_sa(
    surrogate: &PairSurrogateModel,
    config: &DesignConfig,
) -> Result<Vec<Stage1Candidate>, DesignError> {
    let core_alpha = config.constraints.core_alphabet();
    let max_alt = match config.frame_mode {
        FrameMode::Strict => 0,
        FrameMode::Alternating {
            max_alternating_cols,
        } => max_alternating_cols.min(6),
    };
    let target_hand = config.target_params.handedness;
    let weight_ladder: [f64; 6] = if config.specificity_weight <= 0.0 {
        [0.0; 6]
    } else {
        let w = config.specificity_weight;
        [0.0, 0.25 * w, 0.5 * w, w, 1.5 * w, 2.5 * w]
    };

    let per_restart: Result<Vec<HashMap<String, Stage1Candidate>>, DesignError> =
        (0..config.sa_restarts)
            .into_par_iter()
            .map(|restart| {
                let mut rng = ChaCha8Rng::seed_from_u64(
                    config.seed ^ ((restart as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let lambda = weight_ladder[restart % weight_ladder.len()];
                let mut tpl = build_seed_template(&mut rng, config, &core_alpha)?;
                let mut idx = tpl.expand_indices(config.length);
                let (mut tgt, mut dec) = surrogate.evaluate_indices(&idx, target_hand);
                let mut gap = dec - tgt;
                let mut cur_score = tgt - lambda * gap;

                let mut local_archive: HashMap<String, Stage1Candidate> = HashMap::new();
                let mut record = |t: &RepeatTemplate, ix: &[usize], tg: f64, dc: f64| {
                    let seq = indices_to_sequence(ix, config.n_cap, config.c_cap);
                    let key = seq.to_daspr_string();
                    local_archive.entry(key).or_insert_with(|| {
                        let st_stats =
                            compute_seq_stats_iter(ix.iter().map(|&k| CHEM_GROUP_ORDER[k]));
                        Stage1Candidate {
                            indices: ix.to_vec(),
                            frame_a: RepeatTemplate::frame_string(&t.col_a),
                            frame_b: RepeatTemplate::frame_string(&t.col_b),
                            alternating_cols: t.alternating_cols(),
                            net_charge: st_stats.net_charge,
                            trp_tyr_count: st_stats.trp_tyr,
                            aromatic_count: st_stats.aromatic_fyw,
                            aliphatic_count: st_stats.aliphatic_vli,
                            polar_count: st_stats.polar_count,
                            surr_target_de: tg,
                            surr_decoy_de: dc,
                            surr_hand_gap: dc - tg,
                        }
                    });
                };
                record(&tpl, &idx, tgt, dec);

                let t_start = 4.0_f64;
                let t_end = 0.05_f64;
                let steps_f = (config.sa_steps.max(1) - 1).max(1) as f64;
                for step in 0..config.sa_steps {
                    let frac = step as f64 / steps_f;
                    let temp = t_start * (t_end / t_start).powf(frac);
                    let cand_tpl = mutate_template(&mut rng, &tpl, max_alt, &core_alpha);
                    if cand_tpl.alternating_cols() > max_alt
                        || (config.constraints.no_aliphatic_ladders
                            && cand_tpl.has_pure_aliphatic_ladder())
                    {
                        continue;
                    }
                    let cand_idx = cand_tpl.expand_indices(config.length);
                    if !config.constraints.satisfies_indices(&cand_idx) {
                        continue;
                    }
                    let (cand_tgt, cand_dec) = surrogate.evaluate_indices(&cand_idx, target_hand);
                    let cand_gap = cand_dec - cand_tgt;
                    let cand_score = cand_tgt - lambda * cand_gap;
                    let delta = cand_score - cur_score;
                    if delta <= 0.0 || rng.random_bool((-delta / temp).exp().clamp(0.0, 1.0)) {
                        tpl = cand_tpl;
                        idx = cand_idx;
                        tgt = cand_tgt;
                        dec = cand_dec;
                        gap = cand_gap;
                        cur_score = cand_score;
                        record(&tpl, &idx, tgt, dec);
                    }
                    let _ = gap;
                }
                Ok(local_archive)
            })
            .collect();

    let mut merged: HashMap<String, Stage1Candidate> = HashMap::new();
    for map in per_restart? {
        for (k, v) in map {
            merged.entry(k).or_insert(v);
        }
    }

    let mut pool: Vec<Stage1Candidate> = merged.into_values().collect();
    if pool.is_empty() {
        return Err(DesignError::InvalidConfig(
            "Stage-1 search generated no feasible candidates".to_string(),
        ));
    }

    // Sort cleanly before selection.
    pool.sort_by(|a, b| {
        let sa = a.surr_target_de - config.specificity_weight * a.surr_hand_gap;
        let sb = b.surr_target_de - config.specificity_weight * b.surr_hand_gap;
        sa.total_cmp(&sb).then_with(|| a.indices.cmp(&b.indices))
    });
    Ok(pool)
}

fn select_shortlist(pool: Vec<Stage1Candidate>, top_k: usize, weight: f64) -> Vec<Stage1Candidate> {
    let pareto_flags = mark_pareto_flags(&pool, |c| c.surr_target_de, |c| c.surr_hand_gap);
    let mut selected = Vec::with_capacity(top_k.min(pool.len()));
    let mut seen: HashSet<Vec<usize>> = HashSet::new();

    // First include Pareto-optimal frontier points sorted evenly across stability/specificity.
    let mut pareto_indices: Vec<usize> = (0..pool.len()).filter(|&i| pareto_flags[i]).collect();
    pareto_indices.sort_by(|&i, &j| pool[i].surr_target_de.total_cmp(&pool[j].surr_target_de));
    if pareto_indices.len() > top_k / 2 && top_k >= 2 {
        let take_p = (top_k / 2).max(1);
        for slot in 0..take_p {
            let pick = slot * (pareto_indices.len() - 1) / (take_p - 1).max(1);
            let idx = pareto_indices[pick];
            if seen.insert(pool[idx].indices.clone()) {
                selected.push(pool[idx].clone());
            }
        }
    } else {
        for idx in pareto_indices {
            if selected.len() >= top_k {
                break;
            }
            if seen.insert(pool[idx].indices.clone()) {
                selected.push(pool[idx].clone());
            }
        }
    }

    // Fill remainder by scalarized Stage-1 objective.
    let mut scalar_order: Vec<usize> = (0..pool.len()).collect();
    scalar_order.sort_by(|&i, &j| {
        let si = pool[i].surr_target_de - weight * pool[i].surr_hand_gap;
        let sj = pool[j].surr_target_de - weight * pool[j].surr_hand_gap;
        si.total_cmp(&sj)
    });
    for idx in scalar_order {
        if selected.len() >= top_k {
            break;
        }
        if seen.insert(pool[idx].indices.clone()) {
            selected.push(pool[idx].clone());
        }
    }
    selected
}

struct Eval3dResult {
    e_tgt: EnergyScore,
    e_dec: EnergyScore,
    tgt_pdb: String,
    dec_pdb: String,
}

fn evaluate_pair_3d(
    seq: &Sequence,
    target_params: &HelixParams,
    decoy_params: &HelixParams,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<Eval3dResult, DesignError> {
    let bb_seq = seq.as_poly_ala(true);
    let tgt_chain = build_beta_helix(&bb_seq, target_params, 'A')
        .map_err(|e| DesignError::Evaluation(e.to_string()))?;
    let dec_chain = build_beta_helix(&bb_seq, decoy_params, 'A')
        .map_err(|e| DesignError::Evaluation(e.to_string()))?;

    let tgt_pdb = pack_chain(&tgt_chain, seq, daspr_cfg)?;
    let dec_pdb = pack_chain(&dec_chain, seq, daspr_cfg)?;

    let tgt_struct = ScoredStructure::from_pdb_str(&tgt_pdb)
        .map_err(|e| DesignError::Evaluation(e.to_string()))?;
    let dec_struct = ScoredStructure::from_pdb_str(&dec_pdb)
        .map_err(|e| DesignError::Evaluation(e.to_string()))?;

    Ok(Eval3dResult {
        e_tgt: internal_energy(&tgt_struct, ff, energy_opts),
        e_dec: internal_energy(&dec_struct, ff, energy_opts),
        tgt_pdb,
        dec_pdb,
    })
}

struct Eval3dCtx<'a> {
    surrogate: &'a PairSurrogateModel,
    config: &'a DesignConfig,
    decoy_params: &'a HelixParams,
    host_3d_target_energy: f64,
    ff: &'a BudeForceField,
    energy_opts: &'a EnergyOptions,
    daspr_cfg: &'a DasprConfig,
}

fn apply_3d_result(
    cand: &mut DesignCandidate,
    indices: &[usize],
    ctx: &Eval3dCtx<'_>,
    res: Eval3dResult,
) {
    let seq = indices_to_sequence(indices, ctx.config.n_cap, ctx.config.c_cap);
    let stats = compute_seq_stats_iter(indices.iter().map(|&ix| CHEM_GROUP_ORDER[ix]));
    let (st, sd) = ctx
        .surrogate
        .evaluate_indices(indices, ctx.config.target_params.handedness);
    let gap = res.e_dec.total - res.e_tgt.total;
    cand.sequence_with_caps = format_with_caps(&seq);
    cand.sequence_core = seq.to_daspr_string();
    cand.net_charge = stats.net_charge;
    cand.trp_tyr_count = stats.trp_tyr;
    cand.aromatic_count = stats.aromatic_fyw;
    cand.aliphatic_count = stats.aliphatic_vli;
    cand.polar_count = stats.polar_count;
    cand.surr_target_de_vs_ala = st;
    cand.surr_decoy_de_vs_ala = sd;
    cand.surr_hand_gap = sd - st;
    cand.e_3d_target = res.e_tgt.total;
    cand.e_3d_target_steric = res.e_tgt.steric;
    cand.e_3d_target_desolvation = res.e_tgt.desolvation;
    cand.e_3d_target_charge = res.e_tgt.charge;
    cand.e_3d_decoy = res.e_dec.total;
    cand.e_3d_decoy_steric = res.e_dec.steric;
    cand.e_3d_decoy_desolvation = res.e_dec.desolvation;
    cand.e_3d_decoy_charge = res.e_dec.charge;
    cand.e_3d_hand_gap = gap;
    cand.de_3d_vs_ala_host = res.e_tgt.total - ctx.host_3d_target_energy;
    cand.score_3d = res.e_tgt.total - ctx.config.specificity_weight * gap;
    cand.target_pdb = res.tgt_pdb;
    cand.decoy_pdb = res.dec_pdb;
}

fn refine_candidate_termini_3d(
    cand: &mut DesignCandidate,
    mut cur_indices: Vec<usize>,
    ctx: &Eval3dCtx<'_>,
) -> Result<(), DesignError> {
    let config = ctx.config;
    let n = config.length;
    let w = config.constraints.terminal_window.min(n / 2);
    if w == 0 {
        return Ok(());
    }
    let term_alpha = config.constraints.terminal_alphabet();
    let mut term_positions: Vec<usize> = (0..w).chain((n - w)..n).collect();
    term_positions.sort_unstable();
    term_positions.dedup();

    for _pass in 0..2 {
        let mut trial_proposals: Vec<(Vec<usize>, f64)> = Vec::new();
        for &pos in &term_positions {
            let mut per_pos: Vec<(Vec<usize>, f64)> = Vec::new();
            for &aa in &term_alpha {
                let aa_idx = residue_index(aa);
                if cur_indices[pos] == aa_idx {
                    continue;
                }
                let mut mutated = cur_indices.clone();
                mutated[pos] = aa_idx;
                if !config.constraints.satisfies_indices(&mutated) {
                    continue;
                }
                let (st, sd) = ctx
                    .surrogate
                    .evaluate_indices(&mutated, config.target_params.handedness);
                let s_score = st - config.specificity_weight * (sd - st);
                per_pos.push((mutated, s_score));
            }
            per_pos.sort_by(|a, b| a.1.total_cmp(&b.1));

            let mut picked: Vec<(Vec<usize>, f64)> = per_pos.iter().take(4).cloned().collect();
            if config.constraints.allow_terminal_pro {
                let pro_idx = residue_index(CanonicalResidue::Pro);
                if let Some(item) = per_pos.into_iter().find(|(ix, _)| ix[pos] == pro_idx)
                    && !picked.iter().any(|(ix, _)| ix[pos] == pro_idx)
                {
                    picked.push(item);
                }
            }
            trial_proposals.extend(picked);
        }

        if trial_proposals.is_empty() {
            break;
        }

        let evals: Vec<Result<(Vec<usize>, Eval3dResult), DesignError>> = trial_proposals
            .into_par_iter()
            .map(|(ix, _)| {
                let seq = indices_to_sequence(&ix, config.n_cap, config.c_cap);
                let res = evaluate_pair_3d(
                    &seq,
                    &config.target_params,
                    ctx.decoy_params,
                    ctx.ff,
                    ctx.energy_opts,
                    ctx.daspr_cfg,
                )?;
                Ok((ix, res))
            })
            .collect();

        let mut best_improvement: Option<(f64, Vec<usize>, Eval3dResult)> = None;
        for outcome in evals {
            let Ok((ix, res)) = outcome else {
                continue;
            };
            let gap = res.e_dec.total - res.e_tgt.total;
            let s3d = res.e_tgt.total - config.specificity_weight * gap;
            if s3d < cand.score_3d - 1e-6 {
                let replace = match &best_improvement {
                    None => true,
                    Some((best_s, _, _)) => s3d < *best_s,
                };
                if replace {
                    best_improvement = Some((s3d, ix, res));
                }
            }
        }

        if let Some((_, new_ix, res)) = best_improvement {
            apply_3d_result(cand, &new_ix, ctx, res);
            cand.refined_termini = true;
            cur_indices = new_ix;
        } else {
            break;
        }
    }
    Ok(())
}

fn run_design_pipeline(
    config: &DesignConfig,
    surrogate: &PairSurrogateModel,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<DesignReport, DesignError> {
    let pool = run_stage1_sa(surrogate, config)?;
    let stage1_unique_candidates = pool.len();
    let shortlist = select_shortlist(pool, config.top_k_3d, config.specificity_weight);

    let host_indices = vec![residue_index(CanonicalResidue::Ala); config.length];
    let host_seq = indices_to_sequence(&host_indices, config.n_cap, config.c_cap);
    let decoy_params = config.target_params.mirror_enantiomer();
    let host_3d = evaluate_pair_3d(
        &host_seq,
        &config.target_params,
        &decoy_params,
        ff,
        energy_opts,
        daspr_cfg,
    )?;

    let mut stage2_items: Vec<(Vec<usize>, DesignCandidate)> = shortlist
        .par_iter()
        .filter_map(|s1| {
            let seq = indices_to_sequence(&s1.indices, config.n_cap, config.c_cap);
            let res = evaluate_pair_3d(
                &seq,
                &config.target_params,
                &decoy_params,
                ff,
                energy_opts,
                daspr_cfg,
            )
            .ok()?;
            let gap = res.e_dec.total - res.e_tgt.total;
            let score_3d = res.e_tgt.total - config.specificity_weight * gap;
            Some((
                s1.indices.clone(),
                DesignCandidate {
                    rank: 0,
                    sequence_with_caps: format_with_caps(&seq),
                    sequence_core: seq.to_daspr_string(),
                    frame_a: s1.frame_a.clone(),
                    frame_b: s1.frame_b.clone(),
                    alternating_cols: s1.alternating_cols,
                    net_charge: s1.net_charge,
                    trp_tyr_count: s1.trp_tyr_count,
                    aromatic_count: s1.aromatic_count,
                    aliphatic_count: s1.aliphatic_count,
                    polar_count: s1.polar_count,
                    surr_target_de_vs_ala: s1.surr_target_de,
                    surr_decoy_de_vs_ala: s1.surr_decoy_de,
                    surr_hand_gap: s1.surr_hand_gap,
                    e_3d_target: res.e_tgt.total,
                    e_3d_target_steric: res.e_tgt.steric,
                    e_3d_target_desolvation: res.e_tgt.desolvation,
                    e_3d_target_charge: res.e_tgt.charge,
                    e_3d_decoy: res.e_dec.total,
                    e_3d_decoy_steric: res.e_dec.steric,
                    e_3d_decoy_desolvation: res.e_dec.desolvation,
                    e_3d_decoy_charge: res.e_dec.charge,
                    e_3d_hand_gap: gap,
                    de_3d_vs_ala_host: res.e_tgt.total - host_3d.e_tgt.total,
                    score_3d,
                    pareto_optimal: false,
                    refined_termini: false,
                    target_pdb: res.tgt_pdb,
                    decoy_pdb: res.dec_pdb,
                },
            ))
        })
        .collect();

    if stage2_items.is_empty() {
        return Err(DesignError::Evaluation(
            "all Stage-2 3D packing evaluations failed".to_string(),
        ));
    }

    stage2_items.sort_by(|a, b| a.1.score_3d.total_cmp(&b.1.score_3d));

    if config.refine_termini {
        let refine_ctx = Eval3dCtx {
            surrogate,
            config,
            decoy_params: &decoy_params,
            host_3d_target_energy: host_3d.e_tgt.total,
            ff,
            energy_opts,
            daspr_cfg,
        };
        let refine_n = config.refine_top_n.min(stage2_items.len());
        for (ix, cand) in stage2_items.iter_mut().take(refine_n) {
            refine_candidate_termini_3d(cand, ix.clone(), &refine_ctx)?;
        }
    }

    let mut candidates: Vec<DesignCandidate> = stage2_items.into_iter().map(|(_, c)| c).collect();
    candidates.sort_by(|a, b| {
        a.score_3d
            .total_cmp(&b.score_3d)
            .then_with(|| a.sequence_core.cmp(&b.sequence_core))
    });
    let pareto_3d = mark_pareto_flags(&candidates, |c| c.e_3d_target, |c| c.e_3d_hand_gap);
    for (idx, cand) in candidates.iter_mut().enumerate() {
        cand.rank = idx + 1;
        cand.pareto_optimal = pareto_3d[idx];
    }

    Ok(DesignReport {
        host_baseline_sequence: format_with_caps(&host_seq),
        host_3d_target_energy: host_3d.e_tgt.total,
        host_3d_decoy_energy: host_3d.e_dec.total,
        target_handedness: config.target_params.handedness,
        specificity_weight: config.specificity_weight,
        stage1_unique_candidates,
        candidates,
    })
}

/// Runs the two-stage stability & handedness-specific sequence design pipeline.
///
/// If `surrogate` is `None`, automatically runs [`probe_pair_interactions`] on
/// `config.target_params` (22-mer host, both L/D anchors) first.
pub fn design_sequences(
    config: &DesignConfig,
    surrogate: Option<&PairSurrogateModel>,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<DesignReport, DesignError> {
    validate_design_config(config)?;
    resolve_daspr_binary(daspr_cfg)?;

    let owned_surrogate;
    let model = match surrogate {
        Some(s) => s,
        None => {
            let probe_cfg = ProbeConfig {
                minor_params: config.target_params.clone(),
                anchor: AnchorParity::Both,
                threads: config.threads,
                ..ProbeConfig::default()
            };
            let report = probe_pair_interactions(&probe_cfg, ff, energy_opts, daspr_cfg)?;
            owned_surrogate =
                PairSurrogateModel::from_probe_report(&report, config.target_params.handedness)?;
            &owned_surrogate
        }
    };

    if let Some(threads) = config.threads {
        if threads == 0 {
            return Err(DesignError::InvalidConfig(
                "threads must be >= 1".to_string(),
            ));
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| DesignError::Evaluation(e.to_string()))?;
        pool.install(|| run_design_pipeline(config, model, ff, energy_opts, daspr_cfg))
    } else {
        run_design_pipeline(config, model, ff, energy_opts, daspr_cfg)
    }
}

/// Errors returned during beta-helix sequence design.
#[derive(Debug)]
pub enum DesignError {
    InvalidConfig(String),
    IncompletePairTable(String),
    Io(String),
    Input(InputError),
    Probe(ProbeError),
    Sidechain(SidechainError),
    Evaluation(String),
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid design config: {msg}"),
            Self::IncompletePairTable(msg) => write!(f, "incomplete pair table: {msg}"),
            Self::Io(msg) => write!(f, "design I/O error: {msg}"),
            Self::Input(err) => write!(f, "{err}"),
            Self::Probe(err) => write!(f, "{err}"),
            Self::Sidechain(err) => write!(f, "{err}"),
            Self::Evaluation(msg) => write!(f, "design evaluation error: {msg}"),
        }
    }
}

impl std::error::Error for DesignError {}

impl From<InputError> for DesignError {
    fn from(err: InputError) -> Self {
        Self::Input(err)
    }
}

impl From<ProbeError> for DesignError {
    fn from(err: ProbeError) -> Self {
        Self::Probe(err)
    }
}

impl From<SidechainError> for DesignError {
    fn from(err: SidechainError) -> Self {
        Self::Sidechain(err)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::probe::{PairProbeRow, ProbeReport, RankBy};

    pub(crate) fn synthetic_surrogate() -> PairSurrogateModel {
        let make_row = |anchor: Chirality, off: isize, ri: char, rj: char, dde: f64, exa: f64| PairProbeRow {
            anchor_chirality: anchor,
            anchor_seq_id: 9,
            offset: off,
            partner_seq_id: (9_isize + off) as usize,
            res_i_code: ri,
            res_j_code: rj,
            res_i_case: ri,
            res_j_case: rj,
            res_i_pdb: "ALA".to_string(),
            res_j_pdb: "ALA".to_string(),
            dde_dmc: dde,
            de_vs_ala: (exa + 100.0) + dde,
            e_xy: exa + dde,
            e_xa: exa,
            e_ay: -100.0,
            e_aa: -100.0,
            direct_total: 0.0,
            direct_steric: 0.0,
            direct_desolvation: 0.0,
            direct_charge: 0.0,
            direct_sc_total: 0.0,
            direct_sc_steric: 0.0,
            direct_sc_desolvation: 0.0,
            direct_sc_charge: 0.0,
        };
        let rows = vec![
            make_row(Chirality::L, 6, 'W', 'A', 0.0, -110.0),
            make_row(Chirality::D, 6, 'W', 'A', 0.0, -108.0),
            make_row(Chirality::L, 6, 'W', 'W', -4.0, -110.0),
            make_row(Chirality::D, 6, 'W', 'W', 4.0, -108.0),
            make_row(Chirality::L, 6, 'R', 'A', 0.0, -104.0),
            make_row(Chirality::D, 6, 'R', 'A', 0.0, -104.0),
            make_row(Chirality::L, 6, 'E', 'A', 0.0, -103.5),
            make_row(Chirality::D, 6, 'E', 'A', 0.0, -103.5),
            make_row(Chirality::L, 6, 'R', 'E', -13.0, -104.0),
            make_row(Chirality::D, 6, 'R', 'E', -5.0, -104.0),
            make_row(Chirality::L, 6, 'E', 'R', -12.0, -103.5),
            make_row(Chirality::D, 6, 'E', 'R', -4.0, -103.5),
        ];
        let rep = ProbeReport {
            base_sequence: "Ac-AaAaAaAaAaAaAaAaAaAaAa-NH2".to_string(),
            e_aa: -100.0,
            offsets: vec![-1, 1, 5, 6, 7],
            anchor_mode: AnchorParity::Both,
            rank_by: RankBy::Dmc,
            parity_max_diff_threshold: 0.5,
            parity_rmsd_threshold: 0.15,
            alphabet_codes: vec!['A', 'W', 'R', 'E'],
            rows,
            parity_comparisons: Vec::new(),
            skipped_pairs: Vec::new(),
        };
        PairSurrogateModel::from_probe_report(&rep, Handedness::Right).unwrap()
    }

    #[test]
    fn enforces_cys_met_pro_exclusions_and_charge_wy_constraints() {
        let cons = DesignConstraints::default();
        let core = cons.core_alphabet();
        assert!(!core.contains(&CanonicalResidue::Cys));
        assert!(!core.contains(&CanonicalResidue::Met));
        assert!(!core.contains(&CanonicalResidue::Pro));
        assert_eq!(core.len(), 17);

        let term_only = DesignConstraints {
            allow_terminal_pro: true,
            ..DesignConstraints::default()
        };
        let term = term_only.terminal_alphabet();
        assert!(term.contains(&CanonicalResidue::Pro));

        // 30-mer alternating repeat: K(+3) and E(-2) -> net +1, includes W.
        let good = Sequence::parse("Ac-WaLqKvWaLqEvWaLqKvWaLqEvWaLqKv-NH2").unwrap();
        assert!(cons.satisfies_sequence(&good));

        // Negative charge violates [1, 4].
        let neg = Sequence::parse("Ac-WaLqDvWaLqEvWaLqDvWaLqEvWaLqDv-NH2").unwrap();
        assert!(!cons.satisfies_sequence(&neg));

        // Missing W/Y violates min_trp_tyr = 1.
        let no_wy = Sequence::parse("Ac-LaLqKvLaLqEvLaLqKvLaLqEvLaLqKv-NH2").unwrap();
        assert!(!cons.satisfies_sequence(&no_wy));

        // Pro in core violates allow_pro=false even with allow_terminal_pro=true.
        let core_pro = Sequence::parse("Ac-WaLqKvWaLqPvWaLqKvWaLqEvWaLqKv-NH2").unwrap();
        assert!(!term_only.satisfies_sequence(&core_pro));

        // Pro at residue 2 (within first 4) passes when allow_terminal_pro=true.
        let term_pro = Sequence::parse("Ac-WpLqKvWaLqEvWaLqKvWaLqEvWaLqKv-NH2").unwrap();
        assert!(term_only.satisfies_sequence(&term_pro));
        assert!(!cons.satisfies_sequence(&term_pro));
    }

    #[test]
    fn stage1_sa_is_deterministic_and_respects_all_constraints() {
        let model = synthetic_surrogate();
        let cfg = DesignConfig {
            sa_restarts: 6,
            sa_steps: 400,
            seed: 123,
            ..DesignConfig::default()
        };
        let p1 = run_stage1_sa(&model, &cfg).unwrap();
        let p2 = run_stage1_sa(&model, &cfg).unwrap();
        assert!(!p1.is_empty());
        assert_eq!(p1[0].indices, p2[0].indices);
        assert!((p1[0].surr_target_de - p2[0].surr_target_de).abs() < 1e-12);

        for c in &p1 {
            assert!(c.alternating_cols <= 2);
            assert!((1..=4).contains(&c.net_charge));
            assert!(c.trp_tyr_count >= 1);
            assert_eq!(c.indices.len(), 30);
        }
    }

    #[test]
    fn strict_30mer_incompatible_charge_range_is_rejected_cleanly() {
        let cfg = DesignConfig {
            frame_mode: FrameMode::Strict,
            ..DesignConfig::default()
        };
        assert!(validate_design_config(&cfg).is_err());
    }

    #[test]
    fn excludes_arg_and_caps_aromatics_with_polar_minimum() {
        let cons = DesignConstraints {
            allow_arg: false,
            max_aromatic: Some(3),
            max_his: Some(5),
            max_aliphatic: Some(4),
            no_aliphatic_ladders: true,
            min_polar: 14,
            ..DesignConstraints::default()
        };
        assert!(!cons.core_alphabet().contains(&CanonicalResidue::Arg));

        let model = synthetic_surrogate();
        let cfg = DesignConfig {
            constraints: cons,
            sa_restarts: 6,
            sa_steps: 400,
            seed: 7,
            ..DesignConfig::default()
        };
        let pool = run_stage1_sa(&model, &cfg).unwrap();
        assert!(!pool.is_empty());
        for c in &pool {
            let seq = indices_to_sequence(&c.indices, cfg.n_cap, cfg.c_cap);
            assert!(cfg.constraints.satisfies_sequence(&seq));
            assert!(!seq.to_daspr_string().contains(['R', 'r']));
            assert!(c.aromatic_count <= 3);
            assert!(c.aliphatic_count <= 4);
            assert!(c.polar_count >= 14);
        }
    }
}
