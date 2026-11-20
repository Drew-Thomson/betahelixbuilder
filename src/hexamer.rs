//! Two-interface asymmetric parallel hexamer (`C6`) design and scalable multi-state screening.
//!
//! Exploits `beta_6_3`'s ~6.20 residues/turn pitch (~58.1 deg/residue) so consecutive turn sectors
//! `(interface_a, lumen, interface_b)` span ~116 deg — matching the ~120 deg inter-neighbor wedge of
//! a parallel `C6` hexamer. Opposite-charge and steric complementarity between `A` and `B`
//! destabilizes homotypic `AA` and `BB` contacts (tested explicitly via `C2_AA` and `C2_BB` controls)
//! while stabilizing cyclic `AB` packing around an aromatic or salt-bridged central water channel.
//!
//! Scales across large sequence pools via a flat-parallel Stage 3A coarse `(state x R x pitch)`
//! grid batch (`rayon`) followed by Stage 3B Differential Evolution refinement on shortlisted hits.

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bundle::{build_bundle, bundle_target_sequences};
use crate::design::{PairSurrogateModel, residue_formal_charge};
use crate::energy::{BudeForceField, EnergyOptions, ScoredStructure, evaluate_bundle};
use crate::input::{
    BundleSpec, CCap, CanonicalResidue, HelixParams, HelixPlacementParams, NCap, Orientation,
    Sequence, Symmetry, TwistMode,
};
use crate::oligomer::{
    OligomerError, POLAR_EXTERIOR, SeamPattern, estimated_major_radius_for_sym,
    format_capped_sequence, indices_to_sequence, measure_min_ca_distance, net_charge_of, res_idx,
    score_monomer_right_left_3d,
};
use crate::optimization::{
    ConvergenceStats, DeConfig, OptHelixSpec, OptObjective, OptimizeSpec, ParamValue,
    optimize_bundle,
};
use crate::probe::{AnchorParity, CHEM_GROUP_ORDER, ProbeConfig, probe_pair_interactions};
use crate::sidechain::{DasprConfig, pack_bundle, resolve_daspr_binary};

const ALIPHATIC_SEAM: [CanonicalResidue; 3] = [
    CanonicalResidue::Val,
    CanonicalResidue::Leu,
    CanonicalResidue::Ile,
];

const AROMATIC_CORE: [CanonicalResidue; 3] = [
    CanonicalResidue::Trp,
    CanonicalResidue::Tyr,
    CanonicalResidue::Phe,
];

const NEGATIVE_RES: [CanonicalResidue; 2] = [CanonicalResidue::Asp, CanonicalResidue::Glu];
const POLAR_NEUTRAL: [CanonicalResidue; 4] = [
    CanonicalResidue::Gln,
    CanonicalResidue::Asn,
    CanonicalResidue::Ser,
    CanonicalResidue::Thr,
];

/// Chemistry grammar for residues lining the central channel (`s + 1` lumen face).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LumenMode {
    /// Aromatic/hydrophobic lumen (`Trp`/`Tyr`/`Phe`, minor `Leu`/`Gln`) for pi-mediated pores.
    Aromatic,
    /// Mixed oppositely charged hydrophilic channel (`Lys`/`Arg` + `Asp`/`Glu` + `Gln`/`Asn`).
    SaltBridge,
    /// Uncharged polar lumen control (`Gln`/`Asn`/`Ser`/`Thr`/`His`).
    PolarControl,
}

impl LumenMode {
    /// Parses `aromatic`, `salt-bridge`/`salt_bridge`, or `polar`/`polar-control`.
    pub fn parse(raw: &str) -> Result<Self, OligomerError> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "aromatic" => Ok(Self::Aromatic),
            "salt-bridge" | "saltbridge" => Ok(Self::SaltBridge),
            "polar" | "polar-control" | "control" => Ok(Self::PolarControl),
            other => Err(OligomerError::InvalidConfig(format!(
                "unknown lumen mode '{other}' (expected aromatic, salt-bridge, or polar)"
            ))),
        }
    }

    /// Short CSV identifier.
    pub fn label(self) -> &'static str {
        match self {
            Self::Aromatic => "aromatic",
            Self::SaltBridge => "salt_bridge",
            Self::PolarControl => "polar_control",
        }
    }
}

/// Residue role layout across `[0..length)` for a `(interface_a, lumen, interface_b)` helix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HexSeamLayout {
    pub length: usize,
    pub pattern: SeamPattern,
    /// Residue index `s` per turn forming Interface A (`~ -58 deg` when lumen faces inward).
    pub interface_a_indices: Vec<usize>,
    /// Residue index `s + 1` per turn facing the central pore (`~ 0 deg` inward).
    pub lumen_indices: Vec<usize>,
    /// Residue index `s + 2` per turn forming Interface B (`~ +58 deg` when lumen faces inward).
    pub interface_b_indices: Vec<usize>,
    /// Solvent-exposed exterior positions outside `(A, Lumen, B)`.
    pub exterior_indices: Vec<usize>,
}

impl HexSeamLayout {
    /// Builds non-overlapping `(s, s+1, s+2)` sector triples across turns for `length >= 26`.
    pub fn build(length: usize, pattern: SeamPattern) -> Result<Self, OligomerError> {
        if length < 26 {
            return Err(OligomerError::InvalidConfig(format!(
                "length must be >= 26 for 5 two-interface rungs, got {length}"
            )));
        }

        // Start at 0-based index 1 (`1-based 2`) so the final 5th rung ends at `25..=27` within a 28-30mer.
        let mut rung_starts = vec![1usize];
        let mut cur = 1usize;
        for step in pattern.steps() {
            cur += step;
            if cur + 2 < length {
                rung_starts.push(cur);
            }
        }

        let mut interface_a_indices = Vec::with_capacity(rung_starts.len());
        let mut lumen_indices = Vec::with_capacity(rung_starts.len());
        let mut interface_b_indices = Vec::with_capacity(rung_starts.len());
        let mut core_set = HashSet::new();

        for s in rung_starts {
            interface_a_indices.push(s);
            lumen_indices.push(s + 1);
            interface_b_indices.push(s + 2);
            core_set.insert(s);
            core_set.insert(s + 1);
            core_set.insert(s + 2);
        }

        let exterior_indices = (0..length).filter(|i| !core_set.contains(i)).collect();

        Ok(Self {
            length,
            pattern,
            interface_a_indices,
            lumen_indices,
            interface_b_indices,
            exterior_indices,
        })
    }

    fn circular_mean_azimuth_rad(&self, indices: &[usize], params: &HelixParams) -> f64 {
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;
        let dtheta = params.handedness.sign() * 2.0 * std::f64::consts::PI / params.residues_per_turn;
        for &idx in indices {
            let angle = (idx as f64) * dtheta + params.phase_offset_rad;
            cos_sum += angle.cos();
            sin_sum += angle.sin();
        }
        sin_sum.atan2(cos_sum)
    }

    /// `phi_c_alpha_deg` pointing Interface A straight along `radial_in` (`0 deg`).
    pub fn phi_a_inward_deg(&self, params: &HelixParams) -> f64 {
        -self
            .circular_mean_azimuth_rad(&self.interface_a_indices, params)
            .to_degrees()
    }

    /// `phi_c_alpha_deg` pointing the central Lumen face straight along `radial_in` (`0 deg`).
    pub fn phi_lumen_inward_deg(&self, params: &HelixParams) -> f64 {
        -self
            .circular_mean_azimuth_rad(&self.lumen_indices, params)
            .to_degrees()
    }

    /// `phi_c_alpha_deg` pointing Interface B straight along `radial_in` (`0 deg`).
    pub fn phi_b_inward_deg(&self, params: &HelixParams) -> f64 {
        -self
            .circular_mean_azimuth_rad(&self.interface_b_indices, params)
            .to_degrees()
    }
}

/// Symmetry target and interface constraint mode evaluated in Stage 3.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HexStateKind {
    /// Standard unconstrained cyclic/dihedral oligomer state.
    Standard,
    /// Homotypic `C2` dimer constrained with Interface A pointing inward (`A <-> A`).
    HomotypicC2AA,
    /// Homotypic `C2` dimer constrained with Interface B pointing inward (`B <-> B`).
    HomotypicC2BB,
}

/// Oligomer target evaluated in the two-interface hexamer workflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HexOligomerTarget {
    pub label: String,
    pub symmetry: Symmetry,
    pub kind: HexStateKind,
}

impl HexOligomerTarget {
    /// Parses `c2_aa`, `c2_bb`, or any standard `Cn`/`Dn` state (`c2..c8`, `d1..d4`).
    pub fn parse(raw: &str) -> Result<Self, OligomerError> {
        let norm = raw.trim().to_ascii_lowercase().replace('-', "_");
        match norm.as_str() {
            "c2_aa" | "c2aa" => Ok(Self {
                label: "C2_AA".to_string(),
                symmetry: Symmetry::Cn(2),
                kind: HexStateKind::HomotypicC2AA,
            }),
            "c2_bb" | "c2bb" => Ok(Self {
                label: "C2_BB".to_string(),
                symmetry: Symmetry::Cn(2),
                kind: HexStateKind::HomotypicC2BB,
            }),
            _ => {
                let sym = norm.parse::<Symmetry>().map_err(|e| {
                    OligomerError::InvalidConfig(format!("invalid oligomer target '{raw}': {e}"))
                })?;
                let label = match sym {
                    Symmetry::Cn(n) => format!("C{n}"),
                    Symmetry::Dn(n) => format!("D{n}"),
                    Symmetry::None => {
                        return Err(OligomerError::InvalidConfig(
                            "hexamer target requires Cn or Dn symmetry".to_string(),
                        ));
                    }
                };
                Ok(Self {
                    label,
                    symmetry: sym,
                    kind: HexStateKind::Standard,
                })
            }
        }
    }

    /// Default higher-oligomer panel `[C2_AA, C2_BB, C2, C3, C4, C5, C6, C7, D2, D3]`.
    pub fn default_panel() -> Vec<Self> {
        [
            "c2_aa", "c2_bb", "c2", "c3", "c4", "c5", "c6", "c7", "d2", "d3",
        ]
        .into_iter()
        .map(|s| Self::parse(s).expect("valid built-in target"))
        .collect()
    }

    /// Total peptide chains in this assembly.
    pub fn num_chains(&self) -> usize {
        match self.symmetry {
            Symmetry::Cn(n) => n,
            Symmetry::Dn(n) => 2 * n,
            Symmetry::None => 1,
        }
    }

    /// Returns `true` for parallel `Cn` cyclic assemblies (including `C2_AA`/`C2_BB`).
    pub fn is_parallel(&self) -> bool {
        matches!(self.symmetry, Symmetry::Cn(_))
    }

    /// Returns `true` for forced `C2_AA` or `C2_BB` homotypic dimer controls.
    pub fn is_homotypic_control(&self) -> bool {
        !matches!(self.kind, HexStateKind::Standard)
    }
}

/// Configuration for the two-interface hexamer design and scalable Stage-3 workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HexamerWorkflowConfig {
    /// Peptide length (`28..=30`, default `30`).
    pub length: usize,
    pub n_cap: NCap,
    pub c_cap: CCap,
    /// Target right-handed monomer parameters (`HelixParams::beta_6_3()` default).
    pub target_params: HelixParams,
    /// Target parallel cyclic stoichiometry for coarse/DE ranking (default `6` for `C6`).
    pub target_oligomer_n: usize,
    /// Seam step schedules explored (`Strict6`, `Staggered6676`).
    pub seam_patterns: Vec<SeamPattern>,
    /// Central lumen chemistries explored (`Aromatic`, `SaltBridge`).
    pub lumen_modes: Vec<LumenMode>,
    /// Exclude Arginine (`R`/`r`), leaving Lysine as positive residue.
    pub exclude_arg: bool,
    /// Weight on monomer chiral specificity gap (`E_tgt - weight * (E_decoy - E_tgt)`).
    pub monomer_specificity_weight: f64,
    /// Weight on Stage-1 analytical `AB` complementarity & `AA`/`BB` repulsion bonus.
    pub stage1_ab_weight: f64,
    /// Minimum required 3D right-handed monomer gap `E_left - E_right` (default `5.0`).
    pub min_monomer_hand_gap: f64,
    /// Stage-1 SA restarts per `(seam_pattern, lumen_mode)` class.
    pub sa_restarts_per_class: usize,
    /// Stage-1 SA iterations per restart.
    pub sa_steps: usize,
    /// Maximum monomer-gated candidates pushed through Stage 3A coarse multi-state grid.
    pub coarse_pool_size: usize,
    /// Top diverse coarse-screen winners promoted to full multi-state Stage 3B DE.
    pub top_de: usize,
    /// Oligomer target panel (`default: C2_AA, C2_BB, C2, C3, C4, C5, C6, C7, D2, D3`).
    pub states: Vec<HexOligomerTarget>,
    /// DE population size per `(candidate, state)` refinement.
    pub de_pop_size: usize,
    /// DE generations per `(candidate, state)` refinement.
    pub de_generations: usize,
    /// Tail energy drop/chain tolerance flagging `still_minimizing`.
    pub tail_drop_tol_per_chain: f64,
    /// Last-generation energy drop/chain tolerance flagging `still_minimizing`.
    pub last_drop_tol_per_chain: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    pub threads: Option<usize>,
}

impl Default for HexamerWorkflowConfig {
    fn default() -> Self {
        Self {
            length: 30,
            n_cap: NCap::Acetyl,
            c_cap: CCap::Amide,
            target_params: HelixParams::beta_6_3(),
            target_oligomer_n: 6,
            seam_patterns: vec![SeamPattern::Strict6, SeamPattern::Staggered6676],
            lumen_modes: vec![LumenMode::Aromatic, LumenMode::SaltBridge],
            exclude_arg: false,
            monomer_specificity_weight: 0.45,
            stage1_ab_weight: 8.0,
            min_monomer_hand_gap: 5.0,
            sa_restarts_per_class: 10,
            sa_steps: 2500,
            coarse_pool_size: 20,
            top_de: 8,
            states: HexOligomerTarget::default_panel(),
            de_pop_size: 16,
            de_generations: 12,
            tail_drop_tol_per_chain: 0.25,
            last_drop_tol_per_chain: 0.05,
            seed: 42,
            threads: None,
        }
    }
}

/// Monomer gating & multi-state summary per candidate in Stage 3A/3B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HexMonomerCandidate {
    pub candidate_id: usize,
    pub sequence_with_caps: String,
    pub sequence_core: String,
    pub pattern: SeamPattern,
    pub lumen_mode: LumenMode,
    pub interface_a_residues: String,
    pub lumen_residues: String,
    pub interface_b_residues: String,
    pub q_interface_a: i32,
    pub q_lumen: i32,
    pub q_interface_b: i32,
    pub net_charge: i32,
    pub phi_lumen_inward_deg: f64,
    pub surr_target_de: f64,
    pub surr_hand_gap: f64,
    pub stage1_ab_score: f64,
    pub e_3d_right: f64,
    pub e_3d_left_decoy: f64,
    pub e_3d_hand_gap: f64,
    pub passed_hand_gate: bool,
    pub promoted_to_de: bool,
    pub coarse_target_de_assoc: f64,
    pub coarse_ab_vs_aa_bb_gap: f64,
    pub best_oligomer_state: String,
    pub best_assoc_per_chain: f64,
    pub target_assoc_per_chain: f64,
    pub best_homotypic_c2_assoc: f64,
    pub ab_vs_aa_bb_gap: f64,
    pub target_vs_competitors_gap: f64,
    pub target_pore_diameter_a: f64,
}

/// Fast coarse 2D `(radius, pitch)` grid evaluation for one `(candidate, target)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HexCoarseEval {
    pub candidate_id: usize,
    pub sequence_with_caps: String,
    pub pattern: SeamPattern,
    pub lumen_mode: LumenMode,
    pub state_label: String,
    pub num_chains: usize,
    pub is_homotypic_control: bool,
    pub coarse_best_radius: f64,
    pub coarse_best_pitch: f64,
    pub coarse_best_phi_deg: f64,
    pub coarse_e_internal: f64,
    pub coarse_de_assoc_per_chain: f64,
}

/// Full DE oligomer evaluation for one `(promoted candidate, target)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HexOligomerEval {
    pub candidate_id: usize,
    pub sequence_with_caps: String,
    pub pattern: SeamPattern,
    pub lumen_mode: LumenMode,
    pub state_label: String,
    pub num_chains: usize,
    pub is_parallel: bool,
    pub is_homotypic_control: bool,
    pub coarse_best_radius: f64,
    pub coarse_best_pitch: f64,
    pub coarse_best_phi_deg: f64,
    pub opt_radius: f64,
    pub opt_pitch: f64,
    pub opt_phi_c_alpha_deg: f64,
    pub opt_z_offset: f64,
    pub min_interchain_ca_dist: f64,
    pub min_pore_diameter_a: f64,
    pub lumen_mean_radius_a: f64,
    pub e_bundle_internal: f64,
    pub e_per_chain: f64,
    pub de_assoc_per_chain: f64,
    pub e_inter_per_chain: f64,
    pub still_minimizing: bool,
    pub tail_drop_per_chain: f64,
    pub last_gen_drop_per_chain: f64,
    pub final_pop_std_per_chain: f64,
    pub boundary_hits: usize,
    #[serde(skip)]
    pub best_pdb: String,
}

/// Full report produced by [`design_hexamers`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HexamerWorkflowReport {
    pub target_state_label: String,
    pub monomers: Vec<HexMonomerCandidate>,
    pub coarse_evaluations: Vec<HexCoarseEval>,
    pub de_evaluations: Vec<HexOligomerEval>,
}

impl HexamerWorkflowReport {
    /// Formats a summary of monomer gating, coarse throughput pool, and DE hexamer winners.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Two-Interface {} Hexamer Workflow ({} coarse candidates, {} coarse runs, {} DE runs) ===\n",
            self.target_state_label,
            self.monomers.len(),
            self.coarse_evaluations.len(),
            self.de_evaluations.len(),
        ));
        for m in self.monomers.iter().filter(|m| m.promoted_to_de) {
            let gate_str = if m.passed_hand_gate { "PASS" } else { "LOW_GAP" };
            out.push_str(&format!(
                "\n[Cand #{:02}] {} ({}, {}, qA={:+} qLum={:+} qB={:+} net={:+})\n  A=[{}] Lum=[{}] B=[{}]\n  Monomer 3D: E_R={:.3} E_L={:.3} gap={:+.3} [{}] | phi_lum={:+.1}°\n  Hexamer {}: dE={:+.3}/ch | C2_AA/BB_min={:+.3}/ch | AB_vs_AA/BB={:+.3} | best={} ({:+.3}, sel={:+.3}) | pore={:.2} Å\n",
                m.candidate_id,
                m.sequence_with_caps,
                m.pattern.label(),
                m.lumen_mode.label(),
                m.q_interface_a,
                m.q_lumen,
                m.q_interface_b,
                m.net_charge,
                m.interface_a_residues,
                m.lumen_residues,
                m.interface_b_residues,
                m.e_3d_right,
                m.e_3d_left_decoy,
                m.e_3d_hand_gap,
                gate_str,
                m.phi_lumen_inward_deg,
                self.target_state_label,
                m.target_assoc_per_chain,
                m.best_homotypic_c2_assoc,
                m.ab_vs_aa_bb_gap,
                m.best_oligomer_state,
                m.best_assoc_per_chain,
                m.target_vs_competitors_gap,
                m.target_pore_diameter_a,
            ));
            for ev in self
                .de_evaluations
                .iter()
                .filter(|e| e.candidate_id == m.candidate_id)
            {
                let mut flags = Vec::new();
                if ev.still_minimizing {
                    flags.push("STILL_MINIMIZING");
                }
                if ev.boundary_hits > 0 {
                    flags.push("BOUNDARY");
                }
                let flag_str = if flags.is_empty() {
                    String::from("CONVERGED")
                } else {
                    flags.join("|")
                };
                out.push_str(&format!(
                    "    {:>5} (M={}): dE={:+8.3}/ch E_inter={:+8.3}/ch | r={:5.2} P={:+7.1} phi={:+6.1} z={:+.2} | pore={:5.2} Å | tailDrop={:.3} [{}]\n",
                    ev.state_label,
                    ev.num_chains,
                    ev.de_assoc_per_chain,
                    ev.e_inter_per_chain,
                    ev.opt_radius,
                    ev.opt_pitch,
                    ev.opt_phi_c_alpha_deg,
                    ev.opt_z_offset,
                    ev.min_pore_diameter_a,
                    ev.tail_drop_per_chain,
                    flag_str,
                ));
            }
        }
        out
    }

    /// Exports all monomer candidates (coarse + DE summaries) to CSV.
    pub fn monomer_csv(&self) -> String {
        let mut out = String::from(
            "candidate_id,sequence_with_caps,sequence_core,pattern,lumen_mode,interface_a_residues,lumen_residues,interface_b_residues,q_interface_a,q_lumen,q_interface_b,net_charge,phi_lumen_inward_deg,surr_target_dE,surr_hand_gap,stage1_ab_score,e_3d_right,e_3d_left_decoy,e_3d_hand_gap,passed_hand_gate,promoted_to_de,coarse_target_de_assoc,coarse_ab_vs_aa_bb_gap,best_oligomer_state,best_assoc_per_chain,target_assoc_per_chain,best_homotypic_c2_assoc,ab_vs_aa_bb_gap,target_vs_competitors_gap,target_pore_diameter_a\n",
        );
        for m in &self.monomers {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3}\n",
                m.candidate_id,
                m.sequence_with_caps,
                m.sequence_core,
                m.pattern.label(),
                m.lumen_mode.label(),
                m.interface_a_residues,
                m.lumen_residues,
                m.interface_b_residues,
                m.q_interface_a,
                m.q_lumen,
                m.q_interface_b,
                m.net_charge,
                m.phi_lumen_inward_deg,
                m.surr_target_de,
                m.surr_hand_gap,
                m.stage1_ab_score,
                m.e_3d_right,
                m.e_3d_left_decoy,
                m.e_3d_hand_gap,
                m.passed_hand_gate,
                m.promoted_to_de,
                m.coarse_target_de_assoc,
                m.coarse_ab_vs_aa_bb_gap,
                m.best_oligomer_state,
                m.best_assoc_per_chain,
                m.target_assoc_per_chain,
                m.best_homotypic_c2_assoc,
                m.ab_vs_aa_bb_gap,
                m.target_vs_competitors_gap,
                m.target_pore_diameter_a,
            ));
        }
        out
    }

    /// Exports all flat-parallel Stage 3A coarse evaluations to CSV.
    pub fn coarse_csv(&self) -> String {
        let mut out = String::from(
            "candidate_id,sequence_with_caps,pattern,lumen_mode,state,num_chains,is_homotypic_control,coarse_best_radius,coarse_best_pitch,coarse_best_phi_deg,coarse_e_internal,coarse_de_assoc_per_chain\n",
        );
        for c in &self.coarse_evaluations {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{:.4},{:.2},{:.2},{:.6},{:.6}\n",
                c.candidate_id,
                c.sequence_with_caps,
                c.pattern.label(),
                c.lumen_mode.label(),
                c.state_label,
                c.num_chains,
                c.is_homotypic_control,
                c.coarse_best_radius,
                c.coarse_best_pitch,
                c.coarse_best_phi_deg,
                c.coarse_e_internal,
                c.coarse_de_assoc_per_chain,
            ));
        }
        out
    }

    /// Exports full Stage 3B DE evaluations to CSV.
    pub fn hexamer_csv(&self) -> String {
        let mut out = String::from(
            "candidate_id,sequence_with_caps,pattern,lumen_mode,state,num_chains,is_parallel,is_homotypic_control,coarse_best_radius,coarse_best_pitch,coarse_best_phi_deg,opt_radius,opt_pitch,opt_phi_c_alpha_deg,opt_z_offset,min_interchain_ca_dist,min_pore_diameter_a,lumen_mean_radius_a,e_bundle_internal,e_per_chain,de_assoc_per_chain,e_inter_per_chain,still_minimizing,tail_drop_per_chain,last_gen_drop_per_chain,final_pop_std_per_chain,boundary_hits\n",
        );
        for e in &self.de_evaluations {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.4},{:.2},{:.2},{:.4},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{}\n",
                e.candidate_id,
                e.sequence_with_caps,
                e.pattern.label(),
                e.lumen_mode.label(),
                e.state_label,
                e.num_chains,
                e.is_parallel,
                e.is_homotypic_control,
                e.coarse_best_radius,
                e.coarse_best_pitch,
                e.coarse_best_phi_deg,
                e.opt_radius,
                e.opt_pitch,
                e.opt_phi_c_alpha_deg,
                e.opt_z_offset,
                e.min_interchain_ca_dist,
                e.min_pore_diameter_a,
                e.lumen_mean_radius_a,
                e.e_bundle_internal,
                e.e_per_chain,
                e.de_assoc_per_chain,
                e.e_inter_per_chain,
                e.still_minimizing,
                e.tail_drop_per_chain,
                e.last_gen_drop_per_chain,
                e.final_pop_std_per_chain,
                e.boundary_hits,
            ));
        }
        out
    }
}

/// Measures central pore diameter and mean lumen side-chain radius from a packed bundle PDB.
///
/// Assumes the bundle axis is centered on `(0, 0, z)` (as constructed by `supercoil_samples`).
/// Computes `min_pore_diameter_A = 2 * min sqrt(x^2 + y^2)` across all core heavy atoms (`1..=N`)
/// and `lumen_mean_radius_A` across non-backbone heavy atoms of `lumen_1based` residues.
pub fn measure_pore_diameter_angstrom(pdb_str: &str, lumen_1based: &HashSet<usize>) -> (f64, f64) {
    let Ok(st) = ScoredStructure::from_pdb_str(pdb_str) else {
        return (f64::NAN, f64::NAN);
    };
    let mut min_r = f64::INFINITY;
    let mut lumen_r_sum = 0.0;
    let mut lumen_count = 0usize;

    for chain in &st.chains {
        for res in &chain.residues {
            // Skip ACE (seq_id <= 0) and NH2 markers for central pore checks
            if res.seq_id <= 0 || res.res_name == "ACE" || res.res_name == "NH2" {
                continue;
            }
            let is_lumen = usize::try_from(res.seq_id).is_ok_and(|id| lumen_1based.contains(&id));
            for atom in &res.atoms {
                if atom.atom_name.starts_with('H') {
                    continue;
                }
                let r_xy = atom.pos.x.hypot(atom.pos.y);
                if r_xy < min_r {
                    min_r = r_xy;
                }
                if is_lumen
                    && !matches!(
                        atom.atom_name.as_str(),
                        "N" | "CA" | "C" | "O" | "OXT"
                    )
                {
                    lumen_r_sum += r_xy;
                    lumen_count += 1;
                }
            }
        }
    }

    let min_diam = if min_r.is_finite() { 2.0 * min_r } else { f64::NAN };
    let lum_mean = if lumen_count > 0 {
        lumen_r_sum / (lumen_count as f64)
    } else {
        f64::NAN
    };
    (min_diam, lum_mean)
}

fn validate_config(config: &HexamerWorkflowConfig) -> Result<(), OligomerError> {
    if !(26..=36).contains(&config.length) {
        return Err(OligomerError::InvalidConfig(format!(
            "length must be in 26..=36 (recommended 28-30), got {}",
            config.length
        )));
    }
    if config.target_oligomer_n < 3 {
        return Err(OligomerError::InvalidConfig(format!(
            "target_oligomer_n must be >= 3, got {}",
            config.target_oligomer_n
        )));
    }
    config
        .target_params
        .validate(config.length)
        .map_err(|e| OligomerError::InvalidConfig(e.to_string()))?;
    if config.seam_patterns.is_empty()
        || config.lumen_modes.is_empty()
        || config.states.is_empty()
    {
        return Err(OligomerError::InvalidConfig(
            "seam_patterns, lumen_modes, and states must be non-empty".to_string(),
        ));
    }
    if config.coarse_pool_size == 0 || config.top_de == 0 || config.de_pop_size < 4 || config.de_generations == 0 {
        return Err(OligomerError::InvalidConfig(
            "coarse_pool_size >= 1, top_de >= 1, de_pop_size >= 4, de_generations >= 1 required"
                .to_string(),
        ));
    }
    Ok(())
}

fn format_sector_codes(indices: &[usize], res_indices: &[usize]) -> String {
    indices
        .iter()
        .map(|&idx| {
            let aa = CHEM_GROUP_ORDER[res_indices[idx]];
            let is_l = idx % 2 == 0;
            let ch = if aa == CanonicalResidue::Gly {
                crate::input::Chirality::Achiral
            } else if is_l {
                crate::input::Chirality::L
            } else {
                crate::input::Chirality::D
            };
            aa.case_char(ch)
        })
        .collect()
}

fn interface_complementarity_score(
    layout: &HexSeamLayout,
    res_indices: &[usize],
    lumen_mode: LumenMode,
) -> Option<(f64, i32, i32, i32)> {
    let mut qa = 0i32;
    let mut qb = 0i32;
    let mut qlum = 0i32;
    let mut hydro_a = 0usize;
    let mut hydro_b = 0usize;
    let mut pos_a = 0usize;
    let mut neg_b = 0usize;

    for &idx in &layout.interface_a_indices {
        let aa = CHEM_GROUP_ORDER[res_indices[idx]];
        let q = residue_formal_charge(aa);
        if q < 0 {
            return None; // Keep A free of negative charges so AB/AA contrast stays sharp
        }
        qa += q;
        if q > 0 {
            pos_a += 1;
        }
        if ALIPHATIC_SEAM.contains(&aa) || AROMATIC_CORE.contains(&aa) {
            hydro_a += 1;
        }
    }

    for &idx in &layout.interface_b_indices {
        let aa = CHEM_GROUP_ORDER[res_indices[idx]];
        let q = residue_formal_charge(aa);
        if q > 0 {
            return None; // Keep B free of positive charges so AB/BB contrast stays sharp
        }
        qb += q;
        if q < 0 {
            neg_b += 1;
        }
        if ALIPHATIC_SEAM.contains(&aa) || AROMATIC_CORE.contains(&aa) {
            hydro_b += 1;
        }
    }

    if pos_a < 2 || hydro_a < 2 || neg_b < 2 || hydro_b < 2 {
        return None;
    }

    let mut lum_arom = 0usize;
    let mut lum_pos = 0usize;
    let mut lum_neg = 0usize;
    for &idx in &layout.lumen_indices {
        let aa = CHEM_GROUP_ORDER[res_indices[idx]];
        let q = residue_formal_charge(aa);
        qlum += q;
        if AROMATIC_CORE.contains(&aa) {
            lum_arom += 1;
        }
        if q > 0 {
            lum_pos += 1;
        } else if q < 0 {
            lum_neg += 1;
        }
    }

    match lumen_mode {
        LumenMode::Aromatic => {
            if lum_arom < 3 || qlum != 0 {
                return None;
            }
        }
        LumenMode::SaltBridge => {
            if lum_pos < 2 || lum_neg < 2 || !(-1..=1).contains(&qlum) {
                return None;
            }
        }
        LumenMode::PolarControl => {
            if qlum.abs() > 1 {
                return None;
            }
        }
    }

    let net_q = net_charge_of(res_indices);
    if !(-1..=4).contains(&net_q) {
        return None;
    }

    // Score AB electrostatic attraction + AA/BB repulsion + matched hydrophobic rung contact
    let elec_ab_advantage = ((-qa * qb) as f64) + 0.5 * ((qa * qa + qb * qb) as f64);
    let rung_pair_score = layout
        .interface_a_indices
        .iter()
        .zip(layout.interface_b_indices.iter())
        .map(|(&ia, &ib)| {
            let qa_i = residue_formal_charge(CHEM_GROUP_ORDER[res_indices[ia]]);
            let qb_i = residue_formal_charge(CHEM_GROUP_ORDER[res_indices[ib]]);
            let both_hydro = ALIPHATIC_SEAM.contains(&CHEM_GROUP_ORDER[res_indices[ia]])
                && (ALIPHATIC_SEAM.contains(&CHEM_GROUP_ORDER[res_indices[ib]])
                    || AROMATIC_CORE.contains(&CHEM_GROUP_ORDER[res_indices[ib]]));
            if qa_i > 0 && qb_i < 0 {
                2.5
            } else if both_hydro {
                1.8
            } else {
                0.2
            }
        })
        .sum::<f64>();

    let ab_score = 0.35 * elec_ab_advantage + rung_pair_score;
    Some((ab_score, qa, qlum, qb))
}

fn generate_stage1_hex_class(
    layout: &HexSeamLayout,
    lumen_mode: LumenMode,
    surrogate: &PairSurrogateModel,
    config: &HexamerWorkflowConfig,
    rng: &mut ChaCha8Rng,
) -> Vec<(Vec<usize>, f64, f64, f64)> {
    let n_rungs = layout.interface_a_indices.len();
    let pos_choices: Vec<CanonicalResidue> = if config.exclude_arg {
        vec![CanonicalResidue::Lys]
    } else {
        vec![CanonicalResidue::Lys, CanonicalResidue::Arg]
    };

    let mut a_pool: Vec<CanonicalResidue> = ALIPHATIC_SEAM.to_vec();
    a_pool.extend_from_slice(&pos_choices);
    let mut b_pool: Vec<CanonicalResidue> = ALIPHATIC_SEAM.to_vec();
    b_pool.extend_from_slice(&AROMATIC_CORE);
    b_pool.extend_from_slice(&NEGATIVE_RES);

    let lum_pool: Vec<CanonicalResidue> = match lumen_mode {
        LumenMode::Aromatic => vec![
            CanonicalResidue::Trp,
            CanonicalResidue::Tyr,
            CanonicalResidue::Phe,
            CanonicalResidue::Leu,
            CanonicalResidue::Gln,
        ],
        LumenMode::SaltBridge => {
            let mut v = vec![CanonicalResidue::Gln, CanonicalResidue::Asn];
            v.extend_from_slice(&pos_choices);
            v.extend_from_slice(&NEGATIVE_RES);
            v
        }
        LumenMode::PolarControl => {
            let mut v = POLAR_NEUTRAL.to_vec();
            v.push(CanonicalResidue::His);
            v
        }
    };

    let ext_choices: Vec<CanonicalResidue> = POLAR_EXTERIOR
        .iter()
        .copied()
        .filter(|&a| !config.exclude_arg || a != CanonicalResidue::Arg)
        .collect();

    // Build structured valid starting chromosome matching complementary A(+)/B(-) rungs
    let mut a_rungs = vec![CanonicalResidue::Leu; n_rungs];
    let mut b_rungs = vec![CanonicalResidue::Ile; n_rungs];
    let mut lum_rungs = vec![CanonicalResidue::Gln; n_rungs];
    for t in 0..n_rungs {
        if t % 2 == 1 {
            a_rungs[t] = *pos_choices.choose(rng).unwrap();
            b_rungs[t] = *NEGATIVE_RES.choose(rng).unwrap();
        } else {
            a_rungs[t] = *ALIPHATIC_SEAM.choose(rng).unwrap();
            b_rungs[t] = *ALIPHATIC_SEAM.choose(rng).unwrap();
        }
        lum_rungs[t] = match lumen_mode {
            LumenMode::Aromatic => {
                if t == 1 || t == 3 || t == 4 {
                    *AROMATIC_CORE.choose(rng).unwrap()
                } else {
                    CanonicalResidue::Leu
                }
            }
            LumenMode::SaltBridge => {
                if t == 0 || t == 2 {
                    *pos_choices.choose(rng).unwrap()
                } else if t == 1 || t == 3 {
                    *NEGATIVE_RES.choose(rng).unwrap()
                } else {
                    CanonicalResidue::Gln
                }
            }
            LumenMode::PolarControl => *POLAR_NEUTRAL.choose(rng).unwrap(),
        };
    }

    let mut ext_frame = [CanonicalResidue::Gln; 6];
    let assemble = |ar: &[CanonicalResidue],
                    lr: &[CanonicalResidue],
                    br: &[CanonicalResidue],
                    ef: &[CanonicalResidue; 6]|
     -> Option<(Vec<usize>, f64, f64, f64)> {
        let mut idx = vec![0usize; layout.length];
        for &pos in &layout.exterior_indices {
            idx[pos] = res_idx(ef[pos % 6]);
        }
        for (t, &pos) in layout.interface_a_indices.iter().enumerate() {
            idx[pos] = res_idx(ar[t]);
        }
        for (t, &pos) in layout.lumen_indices.iter().enumerate() {
            idx[pos] = res_idx(lr[t]);
        }
        for (t, &pos) in layout.interface_b_indices.iter().enumerate() {
            idx[pos] = res_idx(br[t]);
        }
        let (ab_score, _, _, _) = interface_complementarity_score(layout, &idx, lumen_mode)?;
        let (st, sd) = surrogate.evaluate_indices(&idx, config.target_params.handedness);
        Some((idx, st, sd - st, ab_score))
    };

    for _ in 0..80 {
        if assemble(&a_rungs, &lum_rungs, &b_rungs, &ext_frame).is_some() {
            break;
        }
        for slot in &mut ext_frame {
            *slot = *ext_choices.choose(rng).unwrap();
        }
    }

    let Some(mut cur) = assemble(&a_rungs, &lum_rungs, &b_rungs, &ext_frame) else {
        return Vec::new();
    };
    let mut best_archive = vec![cur.clone()];
    let score_of = |item: &(Vec<usize>, f64, f64, f64)| -> f64 {
        item.1 - config.monomer_specificity_weight * item.2 - config.stage1_ab_weight * item.3
    };
    let mut cur_score = score_of(&cur);

    let t_start = 4.0_f64;
    let t_end = 0.05_f64;
    let denom = (config.sa_steps.max(2) - 1) as f64;

    for step in 0..config.sa_steps {
        let temp = t_start * (t_end / t_start).powf(step as f64 / denom);
        let mut na = a_rungs.clone();
        let mut nl = lum_rungs.clone();
        let mut nb = b_rungs.clone();
        let mut nef = ext_frame;

        match rng.random_range(0..4) {
            0 => {
                let t = rng.random_range(0..n_rungs);
                na[t] = *a_pool.choose(rng).unwrap();
            }
            1 => {
                let t = rng.random_range(0..n_rungs);
                nb[t] = *b_pool.choose(rng).unwrap();
            }
            2 => {
                let t = rng.random_range(0..n_rungs);
                nl[t] = *lum_pool.choose(rng).unwrap();
            }
            _ => {
                let slot = rng.random_range(0..6);
                nef[slot] = *ext_choices.choose(rng).unwrap();
            }
        }

        let Some(next) = assemble(&na, &nl, &nb, &nef) else {
            continue;
        };
        let next_score = score_of(&next);
        let delta = next_score - cur_score;
        if delta <= 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
            a_rungs = na;
            lum_rungs = nl;
            b_rungs = nb;
            ext_frame = nef;
            cur = next;
            cur_score = next_score;
            best_archive.push(cur.clone());
        }
    }

    best_archive
}

fn coarse_grid_for_target(
    target: &HexOligomerTarget,
    layout: &HexSeamLayout,
    params: &HelixParams,
) -> (Vec<(f64, f64, f64)>, f64, f64) {
    let r0 = estimated_major_radius_for_sym(target.symmetry, params);
    let r_min_safe = 0.5 * (params.radius_l + params.radius_d) + 1.1;
    let m = target.num_chains();

    let radii = if m >= 5 {
        vec![
            (r0 - 2.0).max(r_min_safe),
            r0.max(r_min_safe),
            (r0 + 2.0).max(r_min_safe),
            (r0 + 4.0).max(r_min_safe),
        ]
    } else {
        vec![
            (r0 - 1.5).max(r_min_safe),
            r0.max(r_min_safe),
            (r0 + 1.5).max(r_min_safe),
            (r0 + 3.0).max(r_min_safe),
        ]
    };

    let pitches = [-140.0, -320.0, 900.0, 260.0];
    let phi_lum = layout.phi_lumen_inward_deg(params);
    let phi_a = layout.phi_a_inward_deg(params);
    let phi_b = layout.phi_b_inward_deg(params);

    let phis: Vec<f64> = match target.kind {
        HexStateKind::HomotypicC2AA => vec![phi_a],
        HexStateKind::HomotypicC2BB => vec![phi_b],
        HexStateKind::Standard if m == 2 && target.is_parallel() => vec![phi_lum, phi_a, phi_b],
        HexStateKind::Standard => vec![phi_lum],
    };

    let mut grid = Vec::with_capacity(radii.len() * pitches.len() * phis.len());
    for &r in &radii {
        for &p in &pitches {
            for &ph in &phis {
                grid.push((r, p, ph));
            }
        }
    }
    (grid, r_min_safe, r0)
}

struct HexMonomerSeed {
    layout: HexSeamLayout,
    lumen_mode: LumenMode,
    seq: Sequence,
    seq_with_caps: String,
    a_codes: String,
    lum_codes: String,
    b_codes: String,
    qa: i32,
    qlum: i32,
    qb: i32,
    net_q: i32,
    phi_lum: f64,
    surr_tgt: f64,
    surr_gap: f64,
    ab_score: f64,
    e_r: f64,
    e_l: f64,
    hand_gap: f64,
}

fn run_hex_workflow_inner(
    config: &HexamerWorkflowConfig,
    surrogate: &PairSurrogateModel,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<HexamerWorkflowReport, OligomerError> {
    let mut class_jobs = Vec::new();
    for &pat in &config.seam_patterns {
        let layout = HexSeamLayout::build(config.length, pat)?;
        for &lum in &config.lumen_modes {
            class_jobs.push((layout.clone(), lum));
        }
    }

    // Stage 1: Motif-constrained SA sampling per (pattern, lumen_mode) class
    struct RawHit {
        layout: HexSeamLayout,
        lumen_mode: LumenMode,
        indices: Vec<usize>,
        surr_tgt: f64,
        surr_gap: f64,
        ab_score: f64,
    }

    let per_class_keep = ((config.coarse_pool_size * 2) / class_jobs.len()).max(3);
    let mut class_hits = Vec::new();
    for (c_idx, (layout, lum)) in class_jobs.iter().enumerate() {
        let mut samples: Vec<(Vec<usize>, f64, f64, f64)> = (0..config.sa_restarts_per_class)
            .into_par_iter()
            .flat_map(|r_idx| {
                let seed = config.seed
                    ^ ((c_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((r_idx as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                generate_stage1_hex_class(layout, *lum, surrogate, config, &mut rng)
            })
            .collect();

        samples.sort_by(|a, b| {
            let sa = a.1 - config.monomer_specificity_weight * a.2 - config.stage1_ab_weight * a.3;
            let sb = b.1 - config.monomer_specificity_weight * b.2 - config.stage1_ab_weight * b.3;
            sa.total_cmp(&sb).then_with(|| a.0.cmp(&b.0))
        });

        let mut seen = HashSet::new();
        for (ix, st, sg, ab) in samples {
            if seen.insert(ix.clone()) {
                class_hits.push(RawHit {
                    layout: layout.clone(),
                    lumen_mode: *lum,
                    indices: ix,
                    surr_tgt: st,
                    surr_gap: sg,
                    ab_score: ab,
                });
                if seen.len() >= per_class_keep {
                    break;
                }
            }
        }
    }

    let mut seq_seen = HashSet::new();
    let unique_hits: Vec<RawHit> = class_hits
        .into_iter()
        .filter(|h| seq_seen.insert(h.indices.clone()))
        .collect();

    if unique_hits.is_empty() {
        return Err(OligomerError::Evaluation(
            "Stage 1 generated zero valid two-interface candidates".to_string(),
        ));
    }

    // Stage 2: Parallel 3D monomer right/left gating
    let decoy_params = config.target_params.mirror_enantiomer();
    let mut monomer_seeds: Vec<HexMonomerSeed> = unique_hits
        .into_par_iter()
        .filter_map(|hit| {
            let (_, qa, qlum, qb) =
                interface_complementarity_score(&hit.layout, &hit.indices, hit.lumen_mode)?;
            let seq = indices_to_sequence(&hit.indices, config.n_cap, config.c_cap);
            let (e_r, e_l) = score_monomer_right_left_3d(
                &seq,
                &config.target_params,
                &decoy_params,
                ff,
                energy_opts,
                daspr_cfg,
            )
            .ok()?;
            Some(HexMonomerSeed {
                a_codes: format_sector_codes(&hit.layout.interface_a_indices, &hit.indices),
                lum_codes: format_sector_codes(&hit.layout.lumen_indices, &hit.indices),
                b_codes: format_sector_codes(&hit.layout.interface_b_indices, &hit.indices),
                qa,
                qlum,
                qb,
                net_q: net_charge_of(&hit.indices),
                phi_lum: hit.layout.phi_lumen_inward_deg(&config.target_params),
                layout: hit.layout,
                lumen_mode: hit.lumen_mode,
                seq_with_caps: format_capped_sequence(&seq),
                seq,
                surr_tgt: hit.surr_tgt,
                surr_gap: hit.surr_gap,
                ab_score: hit.ab_score,
                e_r,
                e_l,
                hand_gap: e_l - e_r,
            })
        })
        .collect();

    if monomer_seeds.is_empty() {
        return Err(OligomerError::Evaluation(
            "failed to evaluate any monomer candidates in 3D".to_string(),
        ));
    }

    monomer_seeds.sort_by(|a, b| {
        let ga = a.hand_gap >= config.min_monomer_hand_gap;
        let gb = b.hand_gap >= config.min_monomer_hand_gap;
        gb.cmp(&ga)
            .then_with(|| {
                let sa = a.e_r
                    - config.monomer_specificity_weight * a.hand_gap
                    - config.stage1_ab_weight * a.ab_score;
                let sb = b.e_r
                    - config.monomer_specificity_weight * b.hand_gap
                    - config.stage1_ab_weight * b.ab_score;
                sa.total_cmp(&sb)
            })
            .then_with(|| a.seq_with_caps.cmp(&b.seq_with_caps))
    });
    monomer_seeds.truncate(config.coarse_pool_size);

    // Stage 3A: Flat-parallel coarse grid evaluation across all (candidate x target x grid)
    struct CoarseJob<'a> {
        cand_idx: usize,
        state_idx: usize,
        seed: &'a HexMonomerSeed,
        target: &'a HexOligomerTarget,
        r: f64,
        p: f64,
        phi: f64,
    }

    let mut coarse_jobs = Vec::new();
    for (cand_idx, seed) in monomer_seeds.iter().enumerate() {
        for (state_idx, target) in config.states.iter().enumerate() {
            let (grid, _, _) = coarse_grid_for_target(target, &seed.layout, &config.target_params);
            for (r, p, phi) in grid {
                coarse_jobs.push(CoarseJob {
                    cand_idx,
                    state_idx,
                    seed,
                    target,
                    r,
                    p,
                    phi,
                });
            }
        }
    }

    type CoarsePointResult = ((usize, usize), Option<(f64, f64, f64, f64)>);
    let coarse_raw: Vec<CoarsePointResult> = coarse_jobs
        .into_par_iter()
        .map(|job| {
            let eval = || -> Option<(f64, f64, f64, f64)> {
                let spec = BundleSpec {
                    symmetry: job.target.symmetry,
                    keep_gly: true,
                    pack: true,
                    helices: vec![HelixPlacementParams {
                        sequence: job.seed.seq_with_caps.clone(),
                        bundle_radius: job.r,
                        pitch: Some(job.p),
                        phi_c_alpha_deg: job.phi,
                        azimuth_deg: None,
                        z_offset: 0.0,
                        orientation: Orientation::Parallel,
                        twist_mode: TwistMode::UncoupledLabFrame,
                        minor_params: config.target_params.clone(),
                        chain_id: None,
                    }],
                };
                let bundle = build_bundle(&spec).ok()?;
                if let Some(d_min) = bundle.min_interchain_ca_distance()
                    && d_min < 3.4
                {
                    return None;
                }
                let seqs = bundle_target_sequences(&spec).ok()?;
                let packed = pack_bundle(&bundle, &seqs, daspr_cfg).ok()?;
                let st = ScoredStructure::from_pdb_str(&packed).ok()?;
                let rep = evaluate_bundle(&st, ff, energy_opts);
                Some((job.r, job.p, job.phi, rep.internal.total))
            };
            ((job.cand_idx, job.state_idx), eval())
        })
        .collect();

    let mut best_coarse: HashMap<(usize, usize), (f64, f64, f64, f64)> = HashMap::new();
    for (key, res) in coarse_raw {
        if let Some((r, p, ph, e)) = res {
            best_coarse
                .entry(key)
                .and_modify(|cur| {
                    if e < cur.3 {
                        *cur = (r, p, ph, e);
                    }
                })
                .or_insert((r, p, ph, e));
        }
    }

    let mut coarse_evals_out = Vec::with_capacity(monomer_seeds.len() * config.states.len());
    let mut coarse_target_assoc = vec![f64::INFINITY; monomer_seeds.len()];
    let mut coarse_ab_gap = vec![0.0_f64; monomer_seeds.len()];

    let target_label = format!("C{}", config.target_oligomer_n);
    for (cand_idx, seed) in monomer_seeds.iter().enumerate() {
        let mut homo_min = f64::INFINITY;
        let mut target_de = f64::INFINITY;
        for (state_idx, target) in config.states.iter().enumerate() {
            let r0 = estimated_major_radius_for_sym(target.symmetry, &config.target_params);
            let default_phi = match target.kind {
                HexStateKind::HomotypicC2AA => seed.layout.phi_a_inward_deg(&config.target_params),
                HexStateKind::HomotypicC2BB => seed.layout.phi_b_inward_deg(&config.target_params),
                HexStateKind::Standard => seed.phi_lum,
            };
            let (r, p, ph, e_int) = best_coarse
                .get(&(cand_idx, state_idx))
                .copied()
                .unwrap_or((r0, -140.0, default_phi, f64::INFINITY));
            let m_chains = target.num_chains();
            let de_assoc = if e_int.is_finite() {
                e_int / (m_chains as f64) - seed.e_r
            } else {
                f64::INFINITY
            };

            if target.is_homotypic_control() && de_assoc < homo_min {
                homo_min = de_assoc;
            }
            if target.label == target_label {
                target_de = de_assoc;
            }

            coarse_evals_out.push(HexCoarseEval {
                candidate_id: cand_idx + 1,
                sequence_with_caps: seed.seq_with_caps.clone(),
                pattern: seed.layout.pattern,
                lumen_mode: seed.lumen_mode,
                state_label: target.label.clone(),
                num_chains: m_chains,
                is_homotypic_control: target.is_homotypic_control(),
                coarse_best_radius: r,
                coarse_best_pitch: p,
                coarse_best_phi_deg: ph,
                coarse_e_internal: e_int,
                coarse_de_assoc_per_chain: de_assoc,
            });
        }

        coarse_target_assoc[cand_idx] = target_de;
        coarse_ab_gap[cand_idx] = if homo_min.is_finite() && target_de.is_finite() {
            homo_min - target_de
        } else {
            0.0
        };
    }

    // Select top diverse coarse winners to promote into Stage 3B full DE refinement
    let mut ranked_cands: Vec<usize> = (0..monomer_seeds.len()).collect();
    ranked_cands.sort_by(|&ia, &ib| {
        let sa = coarse_target_assoc[ia] - 0.35 * coarse_ab_gap[ia];
        let sb = coarse_target_assoc[ib] - 0.35 * coarse_ab_gap[ib];
        sa.total_cmp(&sb).then_with(|| ia.cmp(&ib))
    });

    let mut promoted_set = HashSet::new();
    // Guarantee at least top candidate per (lumen_mode, seam_pattern) when available
    for &(ref layout, lum) in &class_jobs {
        if promoted_set.len() >= config.top_de {
            break;
        }
        if let Some(&idx) = ranked_cands.iter().find(|&&ci| {
            !promoted_set.contains(&ci)
                && monomer_seeds[ci].lumen_mode == lum
                && monomer_seeds[ci].layout.pattern == layout.pattern
        }) {
            promoted_set.insert(idx);
        }
    }
    for &idx in &ranked_cands {
        if promoted_set.len() >= config.top_de {
            break;
        }
        promoted_set.insert(idx);
    }

    // Stage 3B: Full DE refinement across promoted candidates x states
    let mut promoted_indices: Vec<usize> = promoted_set.into_iter().collect();
    promoted_indices.sort_by(|&a, &b| {
        let sa = coarse_target_assoc[a] - 0.35 * coarse_ab_gap[a];
        let sb = coarse_target_assoc[b] - 0.35 * coarse_ab_gap[b];
        sa.total_cmp(&sb).then_with(|| a.cmp(&b))
    });

    let mut de_jobs = Vec::with_capacity(promoted_indices.len() * config.states.len());
    for &cand_idx in &promoted_indices {
        let seed = &monomer_seeds[cand_idx];
        let lumen_1based: HashSet<usize> =
            seed.layout.lumen_indices.iter().map(|&i| i + 1).collect();
        for (s_idx, target) in config.states.iter().enumerate() {
            de_jobs.push((cand_idx, s_idx, seed, target, lumen_1based.clone()));
        }
    }

    let de_raw: Result<Vec<(usize, HexOligomerEval)>, OligomerError> = de_jobs
        .into_par_iter()
        .map(|(cand_idx, s_idx, seed, target, lumen_1based)| {
            let cand_id = cand_idx + 1;
            let (coarse_r, coarse_p, coarse_phi, _) = best_coarse
                .get(&(cand_idx, s_idx))
                .copied()
                .unwrap_or_else(|| {
                    (
                        estimated_major_radius_for_sym(target.symmetry, &config.target_params),
                        -140.0,
                        seed.phi_lum,
                        f64::INFINITY,
                    )
                });

            let r_min_safe =
                0.5 * (config.target_params.radius_l + config.target_params.radius_d) + 1.1;
            let r_half = if target.num_chains() >= 5 {
                (coarse_r - r_min_safe).clamp(0.8, 3.0)
            } else {
                (coarse_r - r_min_safe).clamp(0.6, 2.2)
            };
            let pitch_half = if coarse_p.abs() > 450.0 { 450.0 } else { 150.0 };
            let phi_half = match target.kind {
                HexStateKind::HomotypicC2AA | HexStateKind::HomotypicC2BB => 20.0,
                HexStateKind::Standard if target.num_chains() == 2 && target.is_parallel() => 180.0,
                HexStateKind::Standard => 45.0,
            };
            let z_param = match target.symmetry {
                Symmetry::Dn(_) => ParamValue::dynamic(0.0, 3.0),
                _ => ParamValue::fixed(0.0),
            };

            let opt_spec = OptimizeSpec {
                symmetry: target.symmetry,
                objective: OptObjective::Internal,
                keep_gly: true,
                de: DeConfig {
                    pop_size: config.de_pop_size,
                    generations: config.de_generations,
                    cxpb: 0.7,
                    diff_weight: 0.5,
                    neighbours: None,
                    seed: config.seed ^ ((cand_id as u64 * 131) + s_idx as u64),
                    threads: None,
                    store_history: false,
                },
                helices: vec![OptHelixSpec {
                    sequence: seed.seq_with_caps.clone(),
                    bundle_radius: ParamValue::dynamic(coarse_r, r_half),
                    pitch: Some(ParamValue::dynamic(coarse_p, pitch_half)),
                    phi_c_alpha_deg: ParamValue::dynamic(coarse_phi, phi_half),
                    azimuth_deg: None,
                    z_offset: z_param,
                    residues_per_turn: None,
                    orientation: Orientation::Parallel,
                    twist_mode: TwistMode::UncoupledLabFrame,
                    minor_params: config.target_params.clone(),
                    chain_id: None,
                }],
            };

            let opt_res = optimize_bundle(&opt_spec, ff, energy_opts, daspr_cfg)
                .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
            let m_chains = target.num_chains();
            let conv: ConvergenceStats = opt_res.convergence_diagnostics(
                m_chains,
                3,
                config.tail_drop_tol_per_chain,
                config.last_drop_tol_per_chain,
            );
            let h0 = &opt_res.best_spec.helices[0];
            let e_int = opt_res.best_energy.internal.total;
            let e_per_chain = e_int / (m_chains as f64);
            let de_assoc = e_per_chain - seed.e_r;
            let e_inter_per_chain = opt_res.best_energy.interaction.total / (m_chains as f64);
            let min_ca = measure_min_ca_distance(&opt_res.best_spec);
            let (min_pore_diam, lum_mean_r) =
                measure_pore_diameter_angstrom(&opt_res.best_pdb, &lumen_1based);

            Ok((
                cand_idx,
                HexOligomerEval {
                    candidate_id: cand_id,
                    sequence_with_caps: seed.seq_with_caps.clone(),
                    pattern: seed.layout.pattern,
                    lumen_mode: seed.lumen_mode,
                    state_label: target.label.clone(),
                    num_chains: m_chains,
                    is_parallel: target.is_parallel(),
                    is_homotypic_control: target.is_homotypic_control(),
                    coarse_best_radius: coarse_r,
                    coarse_best_pitch: coarse_p,
                    coarse_best_phi_deg: coarse_phi,
                    opt_radius: h0.bundle_radius,
                    opt_pitch: h0.pitch.unwrap_or(0.0),
                    opt_phi_c_alpha_deg: h0.phi_c_alpha_deg,
                    opt_z_offset: h0.z_offset,
                    min_interchain_ca_dist: min_ca,
                    min_pore_diameter_a: min_pore_diam,
                    lumen_mean_radius_a: lum_mean_r,
                    e_bundle_internal: e_int,
                    e_per_chain,
                    de_assoc_per_chain: de_assoc,
                    e_inter_per_chain,
                    still_minimizing: conv.still_minimizing,
                    tail_drop_per_chain: conv.tail_drop_per_chain,
                    last_gen_drop_per_chain: conv.last_gen_drop_per_chain,
                    final_pop_std_per_chain: conv.final_pop_std_per_chain,
                    boundary_hits: conv.boundary_hits,
                    best_pdb: opt_res.best_pdb,
                },
            ))
        })
        .collect();

    let mut de_evals_out = Vec::with_capacity(promoted_indices.len() * config.states.len());
    let mut de_by_cand: HashMap<usize, Vec<HexOligomerEval>> = HashMap::new();
    for (cand_idx, eval) in de_raw? {
        de_evals_out.push(eval.clone());
        de_by_cand.entry(cand_idx).or_default().push(eval);
    }

    let mut monomers_out = Vec::with_capacity(monomer_seeds.len());
    for (cand_idx, seed) in monomer_seeds.iter().enumerate() {
        let evals_opt = de_by_cand.get(&cand_idx);
        let (best_label, best_assoc, target_assoc, homo_min, ab_gap, comp_gap, pore_d, promoted) =
            if let Some(evals) = evals_opt {
                let best_overall = evals
                    .iter()
                    .min_by(|a, b| a.de_assoc_per_chain.total_cmp(&b.de_assoc_per_chain))
                    .expect("non-empty evaluations");
                let target_eval = evals.iter().find(|e| e.state_label == target_label);
                let t_assoc = target_eval.map_or(coarse_target_assoc[cand_idx], |e| e.de_assoc_per_chain);
                let pore = target_eval.map_or(f64::NAN, |e| e.min_pore_diameter_a);
                let h_min = evals
                    .iter()
                    .filter(|e| e.is_homotypic_control)
                    .map(|e| e.de_assoc_per_chain)
                    .fold(f64::INFINITY, f64::min);
                let ab_g = if h_min.is_finite() && t_assoc.is_finite() {
                    h_min - t_assoc
                } else {
                    0.0
                };
                let best_comp = evals
                    .iter()
                    .filter(|e| e.state_label != target_label)
                    .map(|e| e.de_assoc_per_chain)
                    .fold(f64::INFINITY, f64::min);
                let c_gap = if best_comp.is_finite() && t_assoc.is_finite() {
                    best_comp - t_assoc
                } else {
                    0.0
                };
                (
                    best_overall.state_label.clone(),
                    best_overall.de_assoc_per_chain,
                    t_assoc,
                    h_min,
                    ab_g,
                    c_gap,
                    pore,
                    true,
                )
            } else {
                (
                    "coarse_only".to_string(),
                    f64::NAN,
                    coarse_target_assoc[cand_idx],
                    f64::NAN,
                    coarse_ab_gap[cand_idx],
                    f64::NAN,
                    f64::NAN,
                    false,
                )
            };

        monomers_out.push(HexMonomerCandidate {
            candidate_id: cand_idx + 1,
            sequence_with_caps: seed.seq_with_caps.clone(),
            sequence_core: seed.seq.to_daspr_string(),
            pattern: seed.layout.pattern,
            lumen_mode: seed.lumen_mode,
            interface_a_residues: seed.a_codes.clone(),
            lumen_residues: seed.lum_codes.clone(),
            interface_b_residues: seed.b_codes.clone(),
            q_interface_a: seed.qa,
            q_lumen: seed.qlum,
            q_interface_b: seed.qb,
            net_charge: seed.net_q,
            phi_lumen_inward_deg: seed.phi_lum,
            surr_target_de: seed.surr_tgt,
            surr_hand_gap: seed.surr_gap,
            stage1_ab_score: seed.ab_score,
            e_3d_right: seed.e_r,
            e_3d_left_decoy: seed.e_l,
            e_3d_hand_gap: seed.hand_gap,
            passed_hand_gate: seed.hand_gap >= config.min_monomer_hand_gap,
            promoted_to_de: promoted,
            coarse_target_de_assoc: coarse_target_assoc[cand_idx],
            coarse_ab_vs_aa_bb_gap: coarse_ab_gap[cand_idx],
            best_oligomer_state: best_label,
            best_assoc_per_chain: best_assoc,
            target_assoc_per_chain: target_assoc,
            best_homotypic_c2_assoc: homo_min,
            ab_vs_aa_bb_gap: ab_gap,
            target_vs_competitors_gap: comp_gap,
            target_pore_diameter_a: pore_d,
        });
    }

    Ok(HexamerWorkflowReport {
        target_state_label: target_label,
        monomers: monomers_out,
        coarse_evaluations: coarse_evals_out,
        de_evaluations: de_evals_out,
    })
}

/// Runs the two-interface parallel hexamer (`C6`) design and scalable multi-state assessment workflow.
pub fn design_hexamers(
    config: &HexamerWorkflowConfig,
    surrogate: Option<&PairSurrogateModel>,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<HexamerWorkflowReport, OligomerError> {
    validate_config(config)?;
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
            let report = probe_pair_interactions(&probe_cfg, ff, energy_opts, daspr_cfg)
                .map_err(|e| OligomerError::Design(crate::design::DesignError::Probe(e)))?;
            owned_surrogate =
                PairSurrogateModel::from_probe_report(&report, config.target_params.handedness)?;
            &owned_surrogate
        }
    };

    if let Some(threads) = config.threads {
        if threads == 0 {
            return Err(OligomerError::InvalidConfig(
                "threads must be >= 1".to_string(),
            ));
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
        pool.install(|| run_hex_workflow_inner(config, model, ff, energy_opts, daspr_cfg))
    } else {
        run_hex_workflow_inner(config, model, ff, energy_opts, daspr_cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdb::bundle_to_pdb_string;

    fn unit_vector_in_xy(chain_pts: &crate::geometry::Chain, idx: usize) -> (f64, f64) {
        let ca = chain_pts.residues[idx].ca;
        let mean_x = chain_pts.residues.iter().map(|r| r.ca.x).sum::<f64>()
            / (chain_pts.residues.len() as f64);
        let mean_y = chain_pts.residues.iter().map(|r| r.ca.y).sum::<f64>()
            / (chain_pts.residues.len() as f64);
        let dx = ca.x - mean_x;
        let dy = ca.y - mean_y;
        let norm = dx.hypot(dy);
        (dx / norm, dy / norm)
    }

    #[test]
    fn hex_seam_layout_separates_a_and_b_by_c6_neighbor_angle() {
        let params = HelixParams::beta_6_3();
        let layout = HexSeamLayout::build(30, SeamPattern::Strict6).unwrap();
        assert_eq!(layout.interface_a_indices, vec![1, 7, 13, 19, 25]);
        assert_eq!(layout.lumen_indices, vec![2, 8, 14, 20, 26]);
        assert_eq!(layout.interface_b_indices, vec![3, 9, 15, 21, 27]);

        // Place lumen inward along radial_in = (-1, 0, 0) on chain 0 of C6
        let phi_lum = layout.phi_lumen_inward_deg(&params);
        let spec = BundleSpec {
            symmetry: Symmetry::Cn(6),
            keep_gly: true,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: "Aa".repeat(15),
                bundle_radius: 13.5,
                pitch: None,
                phi_c_alpha_deg: phi_lum,
                azimuth_deg: Some(0.0),
                z_offset: 0.0,
                orientation: Orientation::Parallel,
                twist_mode: TwistMode::CoupledDarboux,
                minor_params: params,
                chain_id: None,
            }],
        };
        let bundle = build_bundle(&spec).unwrap();
        let ch0 = &bundle.chains[0];

        // At the midpoint rung (idx 13, 14, 15), lumen 14 points along radial_in (-x),
        // A (13) toward CCW chain 1 (+y, ~58 deg), B (15) toward CW chain 5 (-y, ~-58 deg).
        let (lx, _) = unit_vector_in_xy(ch0, 14);
        let (ax, ay) = unit_vector_in_xy(ch0, 13);
        let (bx, by) = unit_vector_in_xy(ch0, 15);
        assert!(lx < -0.95, "lumen midpoint should face radial_in (-x), got x={lx}");
        assert!(ay > 0.80, "interface A should face CCW neighbor (+y), got ({ax}, {ay})");
        assert!(by < -0.80, "interface B should face CW neighbor (-y), got ({bx}, {by})");
        // Angular separation cos(theta_AB) ≈ cos(116.1°) ≈ -0.44 ≈ cos(120°) = -0.5
        let dot_ab = ax * bx + ay * by;
        assert!(
            dot_ab < -0.35 && dot_ab > -0.60,
            "expected ~120 deg between A and B, dot={dot_ab}"
        );
    }

    #[test]
    fn hex_state_targets_parse_c2_controls_and_c6_pore_exceeds_c2() {
        let aa = HexOligomerTarget::parse("c2_aa").unwrap();
        let bb = HexOligomerTarget::parse("c2_bb").unwrap();
        let c6 = HexOligomerTarget::parse("c6").unwrap();
        assert!(aa.is_homotypic_control());
        assert!(bb.is_homotypic_control());
        assert!(!c6.is_homotypic_control());
        assert_eq!(c6.num_chains(), 6);

        let params = HelixParams::beta_6_3();
        let build_pdb = |sym: Symmetry, r: f64| -> String {
            let spec = BundleSpec {
                symmetry: sym,
                keep_gly: true,
                pack: false,
                helices: vec![HelixPlacementParams {
                    sequence: "Ac-AaAaAaAaAaAaAaAaAaAaAaAaAaAaAa-NH2".to_string(),
                    bundle_radius: r,
                    pitch: Some(-140.0),
                    phi_c_alpha_deg: 0.0,
                    azimuth_deg: None,
                    z_offset: 0.0,
                    orientation: Orientation::Parallel,
                    twist_mode: TwistMode::UncoupledLabFrame,
                    minor_params: params.clone(),
                    chain_id: None,
                }],
            };
            let b = build_bundle(&spec).unwrap();
            bundle_to_pdb_string(&b)
        };

        let lum_1based: HashSet<usize> = [3, 9, 15, 21, 27].into_iter().collect();
        let (d_c2, _) = measure_pore_diameter_angstrom(&build_pdb(Symmetry::Cn(2), 6.2), &lum_1based);
        let (d_c6, _) = measure_pore_diameter_angstrom(&build_pdb(Symmetry::Cn(6), 14.0), &lum_1based);
        assert!(
            d_c6 > d_c2 + 10.0 && d_c6 > 16.0,
            "C6 backbone ring should enclose open pore vs C2: d_c6={d_c6}, d_c2={d_c2}"
        );
    }

    #[test]
    fn stage1_generates_valid_aromatic_and_salt_bridge_hex_candidates() {
        let model = crate::design::tests::synthetic_surrogate();
        let cfg = HexamerWorkflowConfig {
            sa_steps: 200,
            ..HexamerWorkflowConfig::default()
        };
        let layout = HexSeamLayout::build(30, SeamPattern::Strict6).unwrap();

        for lum_mode in [LumenMode::Aromatic, LumenMode::SaltBridge] {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let res = generate_stage1_hex_class(&layout, lum_mode, &model, &cfg, &mut rng);
            assert!(!res.is_empty(), "expected hits for {lum_mode:?}");
            let (idx, _, _, _) = res.last().unwrap();
            let (_, qa, qlum, qb) =
                interface_complementarity_score(&layout, idx, lum_mode).expect("valid scoring");
            assert!(qa > 0 && qb < 0, "expected opposite A/B polarity: qa={qa}, qb={qb}");
            if lum_mode == LumenMode::Aromatic {
                assert_eq!(qlum, 0);
            } else {
                assert!((-1..=1).contains(&qlum));
            }
            for &ext in &layout.exterior_indices {
                let aa = CHEM_GROUP_ORDER[idx[ext]];
                assert!(POLAR_EXTERIOR.contains(&aa), "exterior must be polar, got {aa:?}");
            }
        }
    }
}
