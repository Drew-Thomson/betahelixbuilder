//! Water-soluble oligomeric beta-helix design and multi-state DE assessment workflow.
//!
//! Builds capped 28-30 residue monomers with strict (`[6]`) or staggered (`[6, 6, 7, 6]`)
//! hydrophobic seams along one helical face and parallel-directing flanking charges, gates
//! monomers by 3D right-handed stability specificity, runs a fast coarse `(bundle_radius, pitch)`
//! pre-screen across target `Cn`/`Dn` states, and refines geometries with Differential Evolution
//! normalized per chain (`E_internal / M - E_monomer`) with DE convergence diagnostics.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use nalgebra::Point3;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bundle::{build_bundle, bundle_target_sequences};
use crate::design::{DesignError, PairSurrogateModel, residue_formal_charge};
use crate::energy::{BudeForceField, EnergyOptions, ScoredStructure, evaluate_bundle, internal_energy};
use crate::geometry::build_beta_helix;
use crate::input::{
    BundleSpec, CCap, CanonicalResidue, Chirality, HelixParams, HelixPlacementParams, NCap,
    Orientation, Sequence, Symmetry, TwistMode,
};
use crate::optimization::{
    ConvergenceStats, DeConfig, OptHelixSpec, OptObjective, OptimizeSpec, ParamValue,
    optimize_bundle,
};
use crate::probe::{AnchorParity, CHEM_GROUP_ORDER, ProbeConfig, probe_pair_interactions};
use crate::sidechain::{DasprConfig, pack_bundle, pack_chain, resolve_daspr_binary};

pub(crate) const POLAR_EXTERIOR: [CanonicalResidue; 9] = [
    CanonicalResidue::Gln,
    CanonicalResidue::Asn,
    CanonicalResidue::Ser,
    CanonicalResidue::Thr,
    CanonicalResidue::His,
    CanonicalResidue::Asp,
    CanonicalResidue::Glu,
    CanonicalResidue::Lys,
    CanonicalResidue::Ala,
];

pub(crate) fn res_idx(aa: CanonicalResidue) -> usize {
    CHEM_GROUP_ORDER
        .iter()
        .position(|&r| r == aa)
        .expect("all canonical residues exist in CHEM_GROUP_ORDER")
}

/// Repeating step schedule generating a contiguous hydrophobic seam along one beta-helix face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeamPattern {
    /// Constant 6-residue step (`[6, 6, 6, 6]`), drifting `-11.6°/turn` on `beta_6_3`.
    Strict6,
    /// Staggered 6/7-residue schedule (`[6, 6, 7, 6]`), averaging `6.25 res/turn`.
    Staggered6676,
}

impl SeamPattern {
    /// Short identifier string.
    pub fn label(self) -> &'static str {
        match self {
            Self::Strict6 => "strict6",
            Self::Staggered6676 => "staggered6676",
        }
    }

    pub(crate) fn steps(self) -> [usize; 4] {
        match self {
            Self::Strict6 => [6, 6, 6, 6],
            Self::Staggered6676 => [6, 6, 7, 6],
        }
    }
}

/// Lateral width of each hydrophobic seam rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeamWidth {
    /// Single residue per helical turn.
    Single,
    /// Adjacent pair `(s, s + 1)` per helical turn (`L` + `D`).
    PairedLD,
}

impl SeamWidth {
    /// Number of residues per rung.
    pub fn width(self) -> usize {
        match self {
            Self::Single => 1,
            Self::PairedLD => 2,
        }
    }

    /// Short identifier string.
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "w1",
            Self::PairedLD => "w2",
        }
    }
}

/// Allowed hydrophobic side chains within the oligomerization seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeamComposition {
    /// Pure aliphatic seam (`Val`, `Leu`, `Ile`).
    AliphaticOnly,
    /// Mixed aliphatic + aromatic seam (`Val`, `Leu`, `Ile`, `Phe`, `Tyr`, `Trp`).
    MixedAliphaticAromatic,
}

impl SeamComposition {
    /// Candidate residues permitted inside the hydrophobic seam.
    pub fn allowed_residues(self) -> Vec<CanonicalResidue> {
        match self {
            Self::AliphaticOnly => {
                vec![CanonicalResidue::Val, CanonicalResidue::Leu, CanonicalResidue::Ile]
            }
            Self::MixedAliphaticAromatic => vec![
                CanonicalResidue::Val,
                CanonicalResidue::Leu,
                CanonicalResidue::Ile,
                CanonicalResidue::Phe,
                CanonicalResidue::Tyr,
                CanonicalResidue::Trp,
            ],
        }
    }

    /// Short identifier string.
    pub fn label(self) -> &'static str {
        match self {
            Self::AliphaticOnly => "aliphatic",
            Self::MixedAliphaticAromatic => "mixed",
        }
    }
}

/// Flanking charge grammar bordering the hydrophobic seam edges (`s - 1` and `s + width`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgePattern {
    /// Positive left edge (`Lys`/`Arg`), negative right edge (`Asp`/`Glu`) across all turns
    /// to promote parallel `Cn` inter-chain salt bridges.
    ParallelBias,
    /// Turn-by-turn alternating `(+/-)` and `(-/+)` seam edges.
    Alternating,
    /// Polar uncharged/mixed flanks (`Gln`/`Asn`/`Ser`/`Thr`) without engineered edge salt bridges.
    NeutralPolar,
}

impl EdgePattern {
    /// Short identifier string.
    pub fn label(self) -> &'static str {
        match self {
            Self::ParallelBias => "parallel_bias",
            Self::Alternating => "alternating",
            Self::NeutralPolar => "neutral_polar",
        }
    }
}

/// Residue role assignment across `[0..length)` for a seam layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeamLayout {
    pub length: usize,
    pub pattern: SeamPattern,
    pub width: SeamWidth,
    pub seam_rungs: Vec<Vec<usize>>,
    pub seam_indices: Vec<usize>,
    pub left_flanks: Vec<usize>,
    pub right_flanks: Vec<usize>,
    pub exterior_indices: Vec<usize>,
}

impl SeamLayout {
    /// Computes seam, left/right flanking edge, and polar exterior 0-based indices for `length`.
    pub fn build(
        length: usize,
        pattern: SeamPattern,
        width: SeamWidth,
    ) -> Result<Self, OligomerError> {
        if length < 26 {
            return Err(OligomerError::InvalidConfig(format!(
                "length must be >= 26 to accommodate 5 helical rungs, got {length}"
            )));
        }
        // Start at 0-based index 1 (1-based residue 2) so turn-0 left flank (index 0) is interior.
        let mut rung_starts = vec![1usize];
        let mut cur = 1usize;
        for step in pattern.steps() {
            cur += step;
            if cur < length {
                rung_starts.push(cur);
            }
        }

        let w = width.width();
        let mut seam_set = HashSet::new();
        let mut seam_rungs = Vec::with_capacity(rung_starts.len());
        let mut seam_indices = Vec::new();
        for &start in &rung_starts {
            let mut rung = Vec::new();
            for off in 0..w {
                if start + off < length {
                    let idx = start + off;
                    seam_set.insert(idx);
                    seam_indices.push(idx);
                    rung.push(idx);
                }
            }
            if !rung.is_empty() {
                seam_rungs.push(rung);
            }
        }

        let mut left_flanks = Vec::new();
        let mut right_flanks = Vec::new();
        let mut flank_set = HashSet::new();
        for &start in &rung_starts {
            if let Some(lf) = start.checked_sub(1)
                && !seam_set.contains(&lf)
                && flank_set.insert(lf)
            {
                left_flanks.push(lf);
            }
            let rf = start + w;
            if rf < length && !seam_set.contains(&rf) && flank_set.insert(rf) {
                right_flanks.push(rf);
            }
        }

        let exterior_indices = (0..length)
            .filter(|i| !seam_set.contains(i) && !flank_set.contains(i))
            .collect();

        Ok(Self {
            length,
            pattern,
            width,
            seam_rungs,
            seam_indices,
            left_flanks,
            right_flanks,
            exterior_indices,
        })
    }

    /// Circular mean azimuth (radians) of seam residues on `params`, relative to `phi_c_alpha = 0`.
    pub fn seam_mean_azimuth_rad(&self, params: &HelixParams) -> f64 {
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;
        let dtheta = params.handedness.sign() * 2.0 * std::f64::consts::PI / params.residues_per_turn;
        for &idx in &self.seam_indices {
            let angle = (idx as f64) * dtheta + params.phase_offset_rad;
            cos_sum += angle.cos();
            sin_sum += angle.sin();
        }
        sin_sum.atan2(cos_sum)
    }

    /// `phi_c_alpha_deg` orienting the circular mean seam vector directly along `radial_in` (`0°`).
    pub fn aligned_phi_c_alpha_deg(&self, params: &HelixParams) -> f64 {
        -self.seam_mean_azimuth_rad(params).to_degrees()
    }
}

/// Oligomer symmetry target evaluated in Stage 3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OligomerState {
    pub label: String,
    pub symmetry: Symmetry,
}

impl OligomerState {
    /// Parses a symmetry string such as `c2`, `d1`, `c3`, `d2`, `c4`, `c6`, `d3`.
    pub fn parse(raw: &str) -> Result<Self, OligomerError> {
        let trimmed = raw.trim();
        let sym = Symmetry::from_str(trimmed).map_err(|e| {
            OligomerError::InvalidConfig(format!("invalid oligomer state '{trimmed}': {e}"))
        })?;
        let label = match sym {
            Symmetry::Cn(n) => format!("C{n}"),
            Symmetry::Dn(n) => format!("D{n}"),
            Symmetry::None => {
                return Err(OligomerError::InvalidConfig(
                    "oligomer state panel requires Cn or Dn symmetry".to_string(),
                ));
            }
        };
        Ok(Self { label, symmetry: sym })
    }

    /// Default oligomer screening panel `[C2, D1, C3, D2, C4]`.
    pub fn default_panel() -> Vec<Self> {
        ["C2", "D1", "C3", "D2", "C4"]
            .into_iter()
            .map(|s| Self::parse(s).expect("valid built-in symmetry"))
            .collect()
    }

    /// Total number of peptide chains in this oligomeric assembly.
    pub fn num_chains(&self) -> usize {
        match self.symmetry {
            Symmetry::Cn(n) => n,
            Symmetry::Dn(n) => 2 * n,
            Symmetry::None => 1,
        }
    }

    /// Returns `true` for parallel `Cn` cyclic assemblies.
    pub fn is_parallel(&self) -> bool {
        matches!(self.symmetry, Symmetry::Cn(_))
    }
}

/// Configuration for the end-to-end oligomeric beta-helix design workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OligomerWorkflowConfig {
    /// Designed peptide length (`28..=30`, default `30`).
    pub length: usize,
    pub n_cap: NCap,
    pub c_cap: CCap,
    /// Target right-handed monomer backbone geometry (`HelixParams::beta_6_3()` default).
    pub target_params: HelixParams,
    /// Seam step schedules evaluated during candidate generation.
    pub seam_patterns: Vec<SeamPattern>,
    /// Seam widths evaluated during candidate generation.
    pub seam_widths: Vec<SeamWidth>,
    /// Allowed seam hydrophobic compositions evaluated.
    pub seam_compositions: Vec<SeamComposition>,
    /// Flanking edge charge grammars evaluated.
    pub edge_patterns: Vec<EdgePattern>,
    /// Exclude Arginine (`R`/`r`), restricting positive charges to Lysine (`K`/`k`).
    pub exclude_arg: bool,
    /// Allow terminal hydrophobic aromatic anchor (`W` or `Y`) when a pure aliphatic seam is chosen.
    pub allow_terminal_wy: bool,
    /// Monomer specificity weight (`E_tgt - weight * (E_decoy - E_tgt)`).
    pub monomer_specificity_weight: f64,
    /// Minimum required 3D right-handed monomer gap `E_left - E_right` (default `5.0`).
    pub min_monomer_hand_gap: f64,
    /// Stage-1 Simulated Annealing restarts per `(pattern, width, comp, edge)` class.
    pub sa_restarts_per_class: usize,
    /// Stage-1 SA iterations per restart.
    pub sa_steps: usize,
    /// Maximum shortlisted monomer candidates carried into 3D oligomer optimization.
    pub top_monomers: usize,
    /// Oligomeric symmetries tested per shortlisted monomer.
    pub states: Vec<OligomerState>,
    /// Run a coarse `(bundle_radius, pitch)` 2D screen before launching DE per state (default `true`).
    pub run_coarse_screen: bool,
    /// DE population size per `(monomer, state)` optimization.
    pub de_pop_size: usize,
    /// DE generations per `(monomer, state)` optimization.
    pub de_generations: usize,
    /// Per-chain tail-drop tolerance for flagging `still_minimizing` (default `0.25`).
    pub tail_drop_tol_per_chain: f64,
    /// Per-chain final-generation improvement tolerance for flagging `still_minimizing` (default `0.05`).
    pub last_drop_tol_per_chain: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    pub threads: Option<usize>,
}

impl Default for OligomerWorkflowConfig {
    fn default() -> Self {
        Self {
            length: 30,
            n_cap: NCap::Acetyl,
            c_cap: CCap::Amide,
            target_params: HelixParams::beta_6_3(),
            seam_patterns: vec![SeamPattern::Strict6, SeamPattern::Staggered6676],
            seam_widths: vec![SeamWidth::Single, SeamWidth::PairedLD],
            seam_compositions: vec![
                SeamComposition::AliphaticOnly,
                SeamComposition::MixedAliphaticAromatic,
            ],
            edge_patterns: vec![EdgePattern::ParallelBias],
            exclude_arg: false,
            allow_terminal_wy: true,
            monomer_specificity_weight: 0.5,
            min_monomer_hand_gap: 5.0,
            sa_restarts_per_class: 6,
            sa_steps: 2500,
            top_monomers: 6,
            states: OligomerState::default_panel(),
            run_coarse_screen: true,
            de_pop_size: 14,
            de_generations: 12,
            tail_drop_tol_per_chain: 0.25,
            last_drop_tol_per_chain: 0.05,
            seed: 42,
            threads: None,
        }
    }
}

/// Gated monomeric candidate carrying seam layout metadata and 3D right/left verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonomerGateCandidate {
    pub candidate_id: usize,
    pub sequence_with_caps: String,
    pub sequence_core: String,
    pub pattern: SeamPattern,
    pub width: SeamWidth,
    pub composition: SeamComposition,
    pub edge_pattern: EdgePattern,
    pub seam_indices: Vec<usize>,
    pub aligned_phi_c_alpha_deg: f64,
    pub net_charge: i32,
    pub surr_target_de: f64,
    pub surr_hand_gap: f64,
    pub e_3d_right: f64,
    pub e_3d_left_decoy: f64,
    pub e_3d_hand_gap: f64,
    pub passed_hand_gate: bool,
    pub best_oligomer_state: String,
    pub best_assoc_per_chain: f64,
    pub best_parallel_state: String,
    pub best_parallel_assoc_per_chain: f64,
    pub parallel_preference_gap: f64,
    pub state_selectivity_gap: f64,
}

/// Optimized 3D oligomer evaluation for one `(MonomerGateCandidate, OligomerState)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OligomerEval {
    pub candidate_id: usize,
    pub sequence_with_caps: String,
    pub pattern: SeamPattern,
    pub width: SeamWidth,
    pub composition: SeamComposition,
    pub edge_pattern: EdgePattern,
    pub state_label: String,
    pub num_chains: usize,
    pub is_parallel: bool,
    pub coarse_best_radius: f64,
    pub coarse_best_pitch: f64,
    pub coarse_best_internal: f64,
    pub opt_radius: f64,
    pub opt_pitch: f64,
    pub opt_phi_c_alpha_deg: f64,
    pub opt_z_offset: f64,
    pub min_interchain_ca_dist: f64,
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

/// Full report produced by [`design_oligomers`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OligomerWorkflowReport {
    pub monomers: Vec<MonomerGateCandidate>,
    pub evaluations: Vec<OligomerEval>,
}

impl OligomerWorkflowReport {
    /// Formats a readable summary across monomer gating and multi-state DE oligomer scores.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Oligomeric Beta-Helix Workflow ({} shortlisted monomers, {} state DE runs) ===\n",
            self.monomers.len(),
            self.evaluations.len()
        ));
        for m in &self.monomers {
            let gate_str = if m.passed_hand_gate { "PASS" } else { "LOW_GAP" };
            out.push_str(&format!(
                "\n[Cand #{:02}] {} ({}, {}, {}, {}, q={:+}, phi_seam={:+.1}°)\n  Monomer 3D: E_R={:.3} E_L={:.3} hand_gap={:+.3} [{}]\n  Oligomer: best={} (dE_assoc/chain={:+.3}), best_parallel={} ({:+.3}), par_vs_anti_gap={:+.3}, state_selectivity={:+.3}\n",
                m.candidate_id,
                m.sequence_with_caps,
                m.pattern.label(),
                m.width.label(),
                m.composition.label(),
                m.edge_pattern.label(),
                m.net_charge,
                m.aligned_phi_c_alpha_deg,
                m.e_3d_right,
                m.e_3d_left_decoy,
                m.e_3d_hand_gap,
                gate_str,
                m.best_oligomer_state,
                m.best_assoc_per_chain,
                m.best_parallel_state,
                m.best_parallel_assoc_per_chain,
                m.parallel_preference_gap,
                m.state_selectivity_gap,
            ));
            for ev in self
                .evaluations
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
                    "    {:>3} (M={}): dE_assoc/ch={:+8.3} E_inter/ch={:+8.3} | r={:5.2} P={:+7.1} phi={:+6.1} z={:+.2} minCA={:4.2} | tailDrop/ch={:.3} [{}]\n",
                    ev.state_label,
                    ev.num_chains,
                    ev.de_assoc_per_chain,
                    ev.e_inter_per_chain,
                    ev.opt_radius,
                    ev.opt_pitch,
                    ev.opt_phi_c_alpha_deg,
                    ev.opt_z_offset,
                    ev.min_interchain_ca_dist,
                    ev.tail_drop_per_chain,
                    flag_str
                ));
            }
        }
        out
    }

    /// Exports gated monomer rankings and oligomer selectivity summaries to CSV.
    pub fn monomer_csv(&self) -> String {
        let mut out = String::from(
            "candidate_id,sequence_with_caps,sequence_core,pattern,width,composition,edge_pattern,aligned_phi_c_alpha_deg,net_charge,surr_target_dE,surr_hand_gap,e_3d_right,e_3d_left_decoy,e_3d_hand_gap,passed_hand_gate,best_oligomer_state,best_assoc_per_chain,best_parallel_state,best_parallel_assoc_per_chain,parallel_preference_gap,state_selectivity_gap\n",
        );
        for m in &self.monomers {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{:.3},{},{:.4},{:.4},{:.6},{:.6},{:.6},{},{},{:.6},{},{:.6},{:.6},{:.6}\n",
                m.candidate_id,
                m.sequence_with_caps,
                m.sequence_core,
                m.pattern.label(),
                m.width.label(),
                m.composition.label(),
                m.edge_pattern.label(),
                m.aligned_phi_c_alpha_deg,
                m.net_charge,
                m.surr_target_de,
                m.surr_hand_gap,
                m.e_3d_right,
                m.e_3d_left_decoy,
                m.e_3d_hand_gap,
                m.passed_hand_gate,
                m.best_oligomer_state,
                m.best_assoc_per_chain,
                m.best_parallel_state,
                m.best_parallel_assoc_per_chain,
                m.parallel_preference_gap,
                m.state_selectivity_gap,
            ));
        }
        out
    }

    /// Exports all `(candidate, state)` coarse/DE evaluations and convergence flags to CSV.
    pub fn oligomer_csv(&self) -> String {
        let mut out = String::from(
            "candidate_id,sequence_with_caps,pattern,width,composition,edge_pattern,state,num_chains,is_parallel,coarse_best_radius,coarse_best_pitch,coarse_best_internal,opt_radius,opt_pitch,opt_phi_c_alpha_deg,opt_z_offset,min_interchain_ca_dist,e_bundle_internal,e_per_chain,de_assoc_per_chain,e_inter_per_chain,still_minimizing,tail_drop_per_chain,last_gen_drop_per_chain,final_pop_std_per_chain,boundary_hits\n",
        );
        for e in &self.evaluations {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.6},{:.4},{:.2},{:.2},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{}\n",
                e.candidate_id,
                e.sequence_with_caps,
                e.pattern.label(),
                e.width.label(),
                e.composition.label(),
                e.edge_pattern.label(),
                e.state_label,
                e.num_chains,
                e.is_parallel,
                e.coarse_best_radius,
                e.coarse_best_pitch,
                e.coarse_best_internal,
                e.opt_radius,
                e.opt_pitch,
                e.opt_phi_c_alpha_deg,
                e.opt_z_offset,
                e.min_interchain_ca_dist,
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

fn validate_config(config: &OligomerWorkflowConfig) -> Result<(), OligomerError> {
    if !(26..=36).contains(&config.length) {
        return Err(OligomerError::InvalidConfig(format!(
            "length must be in 26..=36 (recommended 28-30), got {}",
            config.length
        )));
    }
    config
        .target_params
        .validate(config.length)
        .map_err(|e| OligomerError::InvalidConfig(e.to_string()))?;
    if config.seam_patterns.is_empty()
        || config.seam_widths.is_empty()
        || config.seam_compositions.is_empty()
        || config.edge_patterns.is_empty()
        || config.states.is_empty()
    {
        return Err(OligomerError::InvalidConfig(
            "seam patterns, widths, compositions, edge patterns, and states must be non-empty"
                .to_string(),
        ));
    }
    if config.top_monomers == 0 || config.de_pop_size < 4 || config.de_generations == 0 {
        return Err(OligomerError::InvalidConfig(
            "top_monomers >= 1, de_pop_size >= 4, de_generations >= 1 required".to_string(),
        ));
    }
    Ok(())
}

fn build_indices_from_roles(
    layout: &SeamLayout,
    seam_pair: &[CanonicalResidue],
    edge_pos: CanonicalResidue,
    edge_neg: CanonicalResidue,
    edge_pattern: EdgePattern,
    exterior_frame: &[CanonicalResidue; 6],
    terminal_wy: Option<CanonicalResidue>,
) -> Vec<usize> {
    let mut indices = vec![0usize; layout.length];
    for &idx in &layout.exterior_indices {
        indices[idx] = res_idx(exterior_frame[idx % 6]);
    }
    for (turn, rung) in layout.seam_rungs.iter().enumerate() {
        for (slot, &idx) in rung.iter().enumerate() {
            let pick = if seam_pair.len() == 2 && (turn % 2 == 1 || slot % 2 == 1) {
                seam_pair[1]
            } else {
                seam_pair[0]
            };
            indices[idx] = res_idx(pick);
        }
    }
    for (t, &idx) in layout.left_flanks.iter().enumerate() {
        indices[idx] = match edge_pattern {
            EdgePattern::ParallelBias => res_idx(edge_pos),
            EdgePattern::Alternating => {
                res_idx(if t % 2 == 0 { edge_pos } else { edge_neg })
            }
            EdgePattern::NeutralPolar => res_idx(exterior_frame[idx % 6]),
        };
    }
    for (t, &idx) in layout.right_flanks.iter().enumerate() {
        indices[idx] = match edge_pattern {
            EdgePattern::ParallelBias => res_idx(edge_neg),
            EdgePattern::Alternating => {
                res_idx(if t % 2 == 0 { edge_neg } else { edge_pos })
            }
            EdgePattern::NeutralPolar => res_idx(exterior_frame[idx % 6]),
        };
    }
    if let Some(wy) = terminal_wy {
        // Place optional Trp/Tyr marker at the final C-terminal exterior residue when aliphatic-only.
        if let Some(&last_ext) = layout.exterior_indices.last() {
            indices[last_ext] = res_idx(wy);
        }
    }
    indices
}

pub(crate) fn indices_to_sequence(res_indices: &[usize], n_cap: NCap, c_cap: CCap) -> Sequence {
    let mut raw = String::with_capacity(res_indices.len() + 8);
    if n_cap == NCap::Acetyl {
        raw.push_str("Ac-");
    }
    for (i, &ix) in res_indices.iter().enumerate() {
        let aa = CHEM_GROUP_ORDER[ix];
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
    Sequence::parse(&raw).expect("internally constructed alternating sequence parses")
}

pub(crate) fn net_charge_of(indices: &[usize]) -> i32 {
    indices
        .iter()
        .map(|&ix| residue_formal_charge(CHEM_GROUP_ORDER[ix]))
        .sum()
}

fn generate_stage1_for_class(
    layout: &SeamLayout,
    comp: SeamComposition,
    edge_pattern: EdgePattern,
    surrogate: &PairSurrogateModel,
    config: &OligomerWorkflowConfig,
    rng: &mut ChaCha8Rng,
) -> Vec<(Vec<usize>, f64, f64)> {
    let seam_choices = comp.allowed_residues();
    let pos_choices: Vec<CanonicalResidue> = if config.exclude_arg {
        vec![CanonicalResidue::Lys]
    } else {
        vec![CanonicalResidue::Lys, CanonicalResidue::Arg]
    };
    let neg_choices = [CanonicalResidue::Asp, CanonicalResidue::Glu];
    let ext_choices: Vec<CanonicalResidue> = POLAR_EXTERIOR
        .iter()
        .copied()
        .filter(|&a| !config.exclude_arg || a != CanonicalResidue::Arg)
        .collect();
    let wy_opt = if config.allow_terminal_wy && comp == SeamComposition::AliphaticOnly {
        Some(CanonicalResidue::Trp)
    } else {
        None
    };

    let mut seam_pair = [
        *seam_choices.choose(rng).unwrap(),
        *seam_choices.choose(rng).unwrap(),
    ];
    let mut edge_pos = *pos_choices.choose(rng).unwrap();
    let mut edge_neg = *neg_choices.choose(rng).unwrap();
    let mut ext_frame = [CanonicalResidue::Gln; 6];
    for slot in &mut ext_frame {
        *slot = *ext_choices.choose(rng).unwrap();
    }

    // Ensure initial net charge is near neutral-positive [-1..=+5]
    let mut best_archive: Vec<(Vec<usize>, f64, f64)> = Vec::new();
    let eval_fn = |sp: &[CanonicalResidue; 2], ep: CanonicalResidue, en: CanonicalResidue, ef: &[CanonicalResidue; 6]| -> Option<(Vec<usize>, f64, f64)> {
        let idx = build_indices_from_roles(layout, sp, ep, en, edge_pattern, ef, wy_opt);
        let q = net_charge_of(&idx);
        if !(-1..=5).contains(&q) {
            return None;
        }
        let (st, sd) = surrogate.evaluate_indices(&idx, config.target_params.handedness);
        Some((idx, st, sd - st))
    };

    // Repair initial frame charges cleanly if needed.
    for _ in 0..200 {
        if eval_fn(&seam_pair, edge_pos, edge_neg, &ext_frame).is_some() {
            break;
        }
        for slot in &mut ext_frame {
            *slot = *ext_choices.choose(rng).unwrap();
        }
    }
    let mut cur = match eval_fn(&seam_pair, edge_pos, edge_neg, &ext_frame) {
        Some(v) => v,
        None => {
            let neutral_ext = [CanonicalResidue::Gln; 6];
            ext_frame = neutral_ext;
            eval_fn(&seam_pair, edge_pos, edge_neg, &ext_frame).unwrap_or_else(|| {
                let idx = build_indices_from_roles(
                    layout,
                    &seam_pair,
                    edge_pos,
                    edge_neg,
                    edge_pattern,
                    &ext_frame,
                    wy_opt,
                );
                let (st, sd) = surrogate.evaluate_indices(&idx, config.target_params.handedness);
                (idx, st, sd - st)
            })
        }
    };
    best_archive.push(cur.clone());
    let mut cur_score = cur.1 - config.monomer_specificity_weight * cur.2;

    let t_start = 4.0_f64;
    let t_end = 0.05_f64;
    let denom = (config.sa_steps.max(2) - 1) as f64;

    for step in 0..config.sa_steps {
        let temp = t_start * (t_end / t_start).powf(step as f64 / denom);
        let mut next_sp = seam_pair;
        let mut next_ep = edge_pos;
        let mut next_en = edge_neg;
        let mut next_ef = ext_frame;

        let mv: f64 = rng.random();
        if mv < 0.28 {
            let s_idx = rng.random_range(0..2);
            next_sp[s_idx] = *seam_choices.choose(rng).unwrap();
            if rng.random_bool(0.5) {
                next_sp[1 - s_idx] = next_sp[s_idx];
            }
        } else if mv < 0.44 {
            if rng.random_bool(0.5) {
                next_ep = *pos_choices.choose(rng).unwrap();
            } else {
                next_en = *neg_choices.choose(rng).unwrap();
            }
        } else {
            let col = rng.random_range(0..6);
            next_ef[col] = *ext_choices.choose(rng).unwrap();
        }

        let Some(next) = eval_fn(&next_sp, next_ep, next_en, &next_ef) else {
            continue;
        };
        let next_score = next.1 - config.monomer_specificity_weight * next.2;
        let delta = next_score - cur_score;
        if delta <= 0.0 || rng.random_bool((-delta / temp).exp().clamp(0.0, 1.0)) {
            seam_pair = next_sp;
            edge_pos = next_ep;
            edge_neg = next_en;
            ext_frame = next_ef;
            cur = next;
            cur_score = next_score;
            best_archive.push(cur.clone());
        }
    }

    best_archive
}

pub(crate) fn score_monomer_right_left_3d(
    seq: &Sequence,
    target_params: &HelixParams,
    decoy_params: &HelixParams,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<(f64, f64), OligomerError> {
    let bb = seq.as_poly_ala(true);
    let r_chain = build_beta_helix(&bb, target_params, 'A')
        .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
    let l_chain = build_beta_helix(&bb, decoy_params, 'A')
        .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
    let r_pdb = pack_chain(&r_chain, seq, daspr_cfg)?;
    let l_pdb = pack_chain(&l_chain, seq, daspr_cfg)?;
    let r_st = ScoredStructure::from_pdb_str(&r_pdb)
        .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
    let l_st = ScoredStructure::from_pdb_str(&l_pdb)
        .map_err(|e| OligomerError::Evaluation(e.to_string()))?;
    Ok((
        internal_energy(&r_st, ff, energy_opts).total,
        internal_energy(&l_st, ff, energy_opts).total,
    ))
}

pub(crate) fn estimated_major_radius_for_sym(symmetry: Symmetry, minor: &HelixParams) -> f64 {
    let r_min = 0.5 * (minor.radius_l + minor.radius_d);
    let m = match symmetry {
        Symmetry::Cn(n) => n,
        Symmetry::Dn(n) => 2 * n,
        Symmetry::None => 1,
    };
    if m <= 2 {
        r_min + 2.4
    } else {
        (r_min + 2.6) / (std::f64::consts::PI / (m as f64)).sin()
    }
}

fn estimated_major_radius(state: &OligomerState, minor: &HelixParams) -> f64 {
    estimated_major_radius_for_sym(state.symmetry, minor)
}

fn coarse_screen_oligomer(
    seq_daspr: &str,
    params: &HelixParams,
    state: &OligomerState,
    phi_center: f64,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<(f64, f64, f64), OligomerError> {
    let r0 = estimated_major_radius(state, params);
    let r_min_bound = 0.5 * (params.radius_l + params.radius_d) + 1.2;
    let radii = [
        (r0 - 1.5).max(r_min_bound),
        r0.max(r_min_bound),
        (r0 + 1.5).max(r_min_bound),
        (r0 + 3.0).max(r_min_bound),
    ];
    let pitches = [-140.0, -320.0, 900.0, 260.0];

    let mut grid = Vec::with_capacity(radii.len() * pitches.len());
    for &r in &radii {
        for &p in &pitches {
            grid.push((r, p));
        }
    }

    let evals: Vec<Option<(f64, f64, f64)>> = grid
        .par_iter()
        .map(|&(r, p)| {
            let spec = BundleSpec {
                symmetry: state.symmetry,
                keep_gly: true,
                pack: true,
                helices: vec![HelixPlacementParams {
                    sequence: seq_daspr.to_string(),
                    bundle_radius: r,
                    pitch: Some(p),
                    phi_c_alpha_deg: phi_center,
                    azimuth_deg: None,
                    z_offset: 0.0,
                    orientation: Orientation::Parallel,
                    twist_mode: TwistMode::UncoupledLabFrame,
                    minor_params: params.clone(),
                    chain_id: None,
                }],
            };
            let bundle = build_bundle(&spec).ok()?;
            // Skip severe backbone inter-chain overlaps in coarse screen
            if let Some(d_min) = bundle.min_interchain_ca_distance()
                && d_min < 3.4
            {
                return None;
            }
            let seqs = bundle_target_sequences(&spec).ok()?;
            let packed = pack_bundle(&bundle, &seqs, daspr_cfg).ok()?;
            let st = ScoredStructure::from_pdb_str(&packed).ok()?;
            let rep = evaluate_bundle(&st, ff, energy_opts);
            Some((r, p, rep.internal.total))
        })
        .collect();

    let mut best: Option<(f64, f64, f64)> = None;
    for item in evals.into_iter().flatten() {
        let replace = match best {
            None => true,
            Some((_, _, best_e)) => item.2 < best_e,
        };
        if replace {
            best = Some(item);
        }
    }
    Ok(best.unwrap_or((r0, -140.0, f64::INFINITY)))
}

pub(crate) fn measure_min_ca_distance(spec: &BundleSpec) -> f64 {
    build_bundle(spec)
        .ok()
        .and_then(|b| b.min_interchain_ca_distance())
        .unwrap_or(f64::NAN)
}

pub(crate) fn format_capped_sequence(seq: &Sequence) -> String {
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

fn min_ca_point_helper(pts: &[Point3<f64>]) -> usize {
    pts.len()
}

fn run_workflow_inner(
    config: &OligomerWorkflowConfig,
    surrogate: &PairSurrogateModel,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<OligomerWorkflowReport, OligomerError> {
    let _ = min_ca_point_helper(&[]);
    let mut class_jobs = Vec::new();
    for &pat in &config.seam_patterns {
        for &wid in &config.seam_widths {
            let layout = SeamLayout::build(config.length, pat, wid)?;
            for &comp in &config.seam_compositions {
                for &edge in &config.edge_patterns {
                    class_jobs.push((layout.clone(), comp, edge));
                }
            }
        }
    }

    // Step 1: Stage-1 SA per motif class
    struct ClassHit {
        layout: SeamLayout,
        comp: SeamComposition,
        edge: EdgePattern,
        indices: Vec<usize>,
        surr_tgt: f64,
        surr_gap: f64,
    }

    let mut per_class_best: Vec<ClassHit> = Vec::new();
    for (c_idx, (layout, comp, edge)) in class_jobs.iter().enumerate() {
        let mut all_class_samples: Vec<(Vec<usize>, f64, f64)> = (0..config.sa_restarts_per_class)
            .into_par_iter()
            .flat_map(|r_idx| {
                let seed = config.seed
                    ^ ((c_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((r_idx as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                generate_stage1_for_class(layout, *comp, *edge, surrogate, config, &mut rng)
            })
            .collect();

        all_class_samples.sort_by(|a, b| {
            let sa = a.1 - config.monomer_specificity_weight * a.2;
            let sb = b.1 - config.monomer_specificity_weight * b.2;
            sa.total_cmp(&sb).then_with(|| a.0.cmp(&b.0))
        });

        let mut seen = HashSet::new();
        for (ix, st, sg) in all_class_samples {
            if seen.insert(ix.clone()) {
                per_class_best.push(ClassHit {
                    layout: layout.clone(),
                    comp: *comp,
                    edge: *edge,
                    indices: ix,
                    surr_tgt: st,
                    surr_gap: sg,
                });
                if seen.len() >= 2 {
                    break;
                }
            }
        }
    }

    // Deduplicate across motif classes by exact sequence while keeping best diverse pool
    let mut seq_seen = HashSet::new();
    let mut unique_hits = Vec::new();
    for hit in per_class_best {
        if seq_seen.insert(hit.indices.clone()) {
            unique_hits.push(hit);
        }
    }

    // Step 2: 3D monomer stability + right-handed specificity verification & gating
    let decoy_params = config.target_params.mirror_enantiomer();
    let mut monomer_scored: Vec<(ClassHit, Sequence, f64, f64, f64)> = unique_hits
        .into_par_iter()
        .filter_map(|hit| {
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
            Some((hit, seq, e_r, e_l, e_l - e_r))
        })
        .collect();

    if monomer_scored.is_empty() {
        return Err(OligomerError::Evaluation(
            "failed to evaluate any monomer candidates in 3D".to_string(),
        ));
    }

    // Rank monomers balancing right-handed stability & handedness gate clearance
    monomer_scored.sort_by(|a, b| {
        let gate_a = a.4 >= config.min_monomer_hand_gap;
        let gate_b = b.4 >= config.min_monomer_hand_gap;
        gate_b
            .cmp(&gate_a)
            .then_with(|| {
                let sa = a.2 - config.monomer_specificity_weight * a.4;
                let sb = b.2 - config.monomer_specificity_weight * b.4;
                sa.total_cmp(&sb)
            })
            .then_with(|| a.1.to_daspr_string().cmp(&b.1.to_daspr_string()))
    });
    monomer_scored.truncate(config.top_monomers);

    // Step 3: Multi-state oligomer coarse screening + DE optimization
    let mut monomers_out = Vec::with_capacity(monomer_scored.len());
    let mut evals_out = Vec::new();

    for (m_idx, (hit, seq, e_r, e_l, hand_gap)) in monomer_scored.into_iter().enumerate() {
        let cand_id = m_idx + 1;
        let phi_center = hit.layout.aligned_phi_c_alpha_deg(&config.target_params);
        let seq_with_caps = {
            let mut s = String::new();
            if seq.n_cap == NCap::Acetyl {
                s.push_str("Ac-");
            }
            s.push_str(&seq.to_daspr_string());
            if seq.c_cap == CCap::Amide {
                s.push_str("-NH2");
            }
            s
        };
        let mut cand_evals = Vec::new();

        for (s_idx, state) in config.states.iter().enumerate() {
            let (coarse_r, coarse_p, coarse_e) = if config.run_coarse_screen {
                coarse_screen_oligomer(
                    &seq_with_caps,
                    &config.target_params,
                    state,
                    phi_center,
                    ff,
                    energy_opts,
                    daspr_cfg,
                )?
            } else {
                let r0 = estimated_major_radius(state, &config.target_params);
                (r0, -140.0, f64::NAN)
            };

            let r_min_safe = 0.5 * (config.target_params.radius_l + config.target_params.radius_d) + 1.1;
            let r_half = (coarse_r - r_min_safe).clamp(0.6, 2.2);
            let pitch_half = if coarse_p.abs() > 450.0 { 450.0 } else { 150.0 };
            let z_param = match state.symmetry {
                Symmetry::Dn(_) => ParamValue::dynamic(0.0, 3.0),
                _ => ParamValue::fixed(0.0),
            };

            let opt_spec = OptimizeSpec {
                symmetry: state.symmetry,
                objective: OptObjective::Internal,
                keep_gly: true,
                de: DeConfig {
                    pop_size: config.de_pop_size,
                    generations: config.de_generations,
                    cxpb: 0.7,
                    diff_weight: 0.5,
                    neighbours: None,
                    seed: config.seed ^ ((cand_id as u64 * 97) + s_idx as u64),
                    threads: config.threads,
                    store_history: false,
                },
                helices: vec![OptHelixSpec {
                    sequence: seq_with_caps.clone(),
                    bundle_radius: ParamValue::dynamic(coarse_r, r_half),
                    pitch: Some(ParamValue::dynamic(coarse_p, pitch_half)),
                    phi_c_alpha_deg: ParamValue::dynamic(phi_center, 40.0),
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
            let m_chains = state.num_chains();
            let conv: ConvergenceStats = opt_res.convergence_diagnostics(
                m_chains,
                3,
                config.tail_drop_tol_per_chain,
                config.last_drop_tol_per_chain,
            );
            let h0 = &opt_res.best_spec.helices[0];
            let e_int = opt_res.best_energy.internal.total;
            let e_per_chain = e_int / (m_chains as f64);
            let de_assoc = e_per_chain - e_r;
            let e_inter_per_chain = opt_res.best_energy.interaction.total / (m_chains as f64);
            let min_ca = measure_min_ca_distance(&opt_res.best_spec);

            cand_evals.push(OligomerEval {
                candidate_id: cand_id,
                sequence_with_caps: seq_with_caps.clone(),
                pattern: hit.layout.pattern,
                width: hit.layout.width,
                composition: hit.comp,
                edge_pattern: hit.edge,
                state_label: state.label.clone(),
                num_chains: m_chains,
                is_parallel: state.is_parallel(),
                coarse_best_radius: coarse_r,
                coarse_best_pitch: coarse_p,
                coarse_best_internal: coarse_e,
                opt_radius: h0.bundle_radius,
                opt_pitch: h0.pitch.unwrap_or(0.0),
                opt_phi_c_alpha_deg: h0.phi_c_alpha_deg,
                opt_z_offset: h0.z_offset,
                min_interchain_ca_dist: min_ca,
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
            });
        }

        let mut sorted_by_assoc: Vec<&OligomerEval> = cand_evals.iter().collect();
        sorted_by_assoc.sort_by(|a, b| a.de_assoc_per_chain.total_cmp(&b.de_assoc_per_chain));
        let best_overall = sorted_by_assoc[0];
        let state_selectivity_gap = if sorted_by_assoc.len() >= 2 {
            sorted_by_assoc[1].de_assoc_per_chain - best_overall.de_assoc_per_chain
        } else {
            0.0
        };

        let best_par = cand_evals
            .iter()
            .filter(|e| e.is_parallel)
            .min_by(|a, b| a.de_assoc_per_chain.total_cmp(&b.de_assoc_per_chain));
        let best_anti = cand_evals
            .iter()
            .filter(|e| !e.is_parallel)
            .min_by(|a, b| a.de_assoc_per_chain.total_cmp(&b.de_assoc_per_chain));

        let (best_par_label, best_par_assoc) =
            best_par.map_or_else(|| ("none".to_string(), 0.0), |e| (e.state_label.clone(), e.de_assoc_per_chain));
        let par_pref_gap = match (best_par, best_anti) {
            (Some(p), Some(a)) => a.de_assoc_per_chain - p.de_assoc_per_chain,
            _ => 0.0,
        };

        monomers_out.push(MonomerGateCandidate {
            candidate_id: cand_id,
            sequence_with_caps: seq_with_caps,
            sequence_core: seq.to_daspr_string(),
            pattern: hit.layout.pattern,
            width: hit.layout.width,
            composition: hit.comp,
            edge_pattern: hit.edge,
            seam_indices: hit.layout.seam_indices.clone(),
            aligned_phi_c_alpha_deg: phi_center,
            net_charge: net_charge_of(&hit.indices),
            surr_target_de: hit.surr_tgt,
            surr_hand_gap: hit.surr_gap,
            e_3d_right: e_r,
            e_3d_left_decoy: e_l,
            e_3d_hand_gap: hand_gap,
            passed_hand_gate: hand_gap >= config.min_monomer_hand_gap,
            best_oligomer_state: best_overall.state_label.clone(),
            best_assoc_per_chain: best_overall.de_assoc_per_chain,
            best_parallel_state: best_par_label,
            best_parallel_assoc_per_chain: best_par_assoc,
            parallel_preference_gap: par_pref_gap,
            state_selectivity_gap,
        });
        evals_out.extend(cand_evals);
    }

    Ok(OligomerWorkflowReport {
        monomers: monomers_out,
        evaluations: evals_out,
    })
}

/// Runs the water-soluble parallel oligomeric beta-helix design and multi-state assessment pipeline.
pub fn design_oligomers(
    config: &OligomerWorkflowConfig,
    surrogate: Option<&PairSurrogateModel>,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<OligomerWorkflowReport, OligomerError> {
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
                .map_err(|e| OligomerError::Design(DesignError::Probe(e)))?;
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
        pool.install(|| run_workflow_inner(config, model, ff, energy_opts, daspr_cfg))
    } else {
        run_workflow_inner(config, model, ff, energy_opts, daspr_cfg)
    }
}

/// Errors produced during oligomeric beta-helix design and assessment.
#[derive(Debug)]
pub enum OligomerError {
    InvalidConfig(String),
    Design(DesignError),
    Sidechain(crate::sidechain::SidechainError),
    Evaluation(String),
}

impl fmt::Display for OligomerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid oligomer workflow config: {msg}"),
            Self::Design(err) => write!(f, "{err}"),
            Self::Sidechain(err) => write!(f, "{err}"),
            Self::Evaluation(msg) => write!(f, "oligomer evaluation error: {msg}"),
        }
    }
}

impl std::error::Error for OligomerError {}

impl From<DesignError> for OligomerError {
    fn from(err: DesignError) -> Self {
        Self::Design(err)
    }
}

impl From<crate::sidechain::SidechainError> for OligomerError {
    fn from(err: crate::sidechain::SidechainError) -> Self {
        Self::Sidechain(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::HelixPlacementParams;

    #[test]
    fn seam_layouts_compute_exact_indices_and_center_azimuth_inward() {
        let s6 = SeamLayout::build(30, SeamPattern::Strict6, SeamWidth::Single).unwrap();
        assert_eq!(s6.seam_indices, vec![1, 7, 13, 19, 25]);
        assert_eq!(s6.left_flanks, vec![0, 6, 12, 18, 24]);
        assert_eq!(s6.right_flanks, vec![2, 8, 14, 20, 26]);

        let stag = SeamLayout::build(30, SeamPattern::Staggered6676, SeamWidth::PairedLD).unwrap();
        // Rungs at [1,2], [7,8], [13,14], [20,21], [26,27] -> left flanks and right flanks valid in [0..30).
        assert_eq!(stag.seam_indices, vec![1, 2, 7, 8, 13, 14, 20, 21, 26, 27]);
        assert_eq!(stag.left_flanks, vec![0, 6, 12, 19, 25]);
        assert_eq!(stag.right_flanks, vec![3, 9, 15, 22, 28]);

        // Placing aligned phi_c_alpha on a straight C2 bundle must point the seam circular centroid toward origin (-X for azimuth 0).
        let params = HelixParams::beta_6_3();
        let phi = s6.aligned_phi_c_alpha_deg(&params);
        let bundle = build_bundle(&BundleSpec {
            symmetry: Symmetry::Cn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: "Aa".repeat(15),
                bundle_radius: 7.5,
                pitch: None,
                phi_c_alpha_deg: phi,
                azimuth_deg: Some(0.0),
                z_offset: 0.0,
                orientation: Orientation::Parallel,
                twist_mode: TwistMode::CoupledDarboux,
                minor_params: params,
                chain_id: None,
            }],
        })
        .unwrap();
        // Chain 0 sits at (+7.5, 0, z), radial_in = (-1, 0, 0). Mean seam Cα X must be closer to origin than chain center 7.5.
        let seam_mean_x: f64 = s6
            .seam_indices
            .iter()
            .map(|&ix| bundle.chains[0].residues[ix].ca.x)
            .sum::<f64>()
            / (s6.seam_indices.len() as f64);
        assert!(
            seam_mean_x < 5.0,
            "expected seam to face inward (< 5.0 Å), got mean_x={seam_mean_x}"
        );
    }

    #[test]
    fn parses_arbitrary_cn_dn_states_including_c6_and_d3() {
        let c6 = OligomerState::parse("c6").unwrap();
        assert_eq!(c6.label, "C6");
        assert_eq!(c6.num_chains(), 6);
        assert!(c6.is_parallel());

        let d3 = OligomerState::parse("D3").unwrap();
        assert_eq!(d3.label, "D3");
        assert_eq!(d3.num_chains(), 6);
        assert!(!d3.is_parallel());
    }

    #[test]
    fn parallel_bias_assigns_consistent_opposite_flank_charges() {
        let layout = SeamLayout::build(30, SeamPattern::Strict6, SeamWidth::Single).unwrap();
        let ix = build_indices_from_roles(
            &layout,
            &[CanonicalResidue::Leu],
            CanonicalResidue::Lys,
            CanonicalResidue::Glu,
            EdgePattern::ParallelBias,
            &[CanonicalResidue::Gln; 6],
            None,
        );
        for &lf in &layout.left_flanks {
            assert_eq!(CHEM_GROUP_ORDER[ix[lf]], CanonicalResidue::Lys);
        }
        for &rf in &layout.right_flanks {
            assert_eq!(CHEM_GROUP_ORDER[ix[rf]], CanonicalResidue::Glu);
        }
        for &si in &layout.seam_indices {
            assert_eq!(CHEM_GROUP_ORDER[ix[si]], CanonicalResidue::Leu);
        }
    }
}
