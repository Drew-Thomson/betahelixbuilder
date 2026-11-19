//! Systematic pairwise amino-acid composition probing and double-mutant cycle ranking.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::energy::{
    BudeForceField, EnergyOptions, ScoredStructure, direct_residue_pair_energy, internal_energy,
};
use crate::geometry::build_beta_helix;
use crate::input::{
    CCap, CanonicalResidue, Chirality, HelixParams, InputError, NCap, ResidueSpec, Sequence,
};
use crate::sidechain::{DasprConfig, SidechainError, pack_chain, resolve_daspr_binary};

/// Canonical 20 amino acids ordered by physicochemical group for analysis and heatmaps.
pub const CHEM_GROUP_ORDER: [CanonicalResidue; 20] = [
    CanonicalResidue::Gly,
    CanonicalResidue::Ala,
    CanonicalResidue::Val,
    CanonicalResidue::Leu,
    CanonicalResidue::Ile,
    CanonicalResidue::Met,
    CanonicalResidue::Pro,
    CanonicalResidue::Phe,
    CanonicalResidue::Tyr,
    CanonicalResidue::Trp,
    CanonicalResidue::Ser,
    CanonicalResidue::Thr,
    CanonicalResidue::Asn,
    CanonicalResidue::Gln,
    CanonicalResidue::Cys,
    CanonicalResidue::His,
    CanonicalResidue::Asp,
    CanonicalResidue::Glu,
    CanonicalResidue::Lys,
    CanonicalResidue::Arg,
];

/// Anchor stereochemical position(s) to probe in an alternating L/D host helix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorParity {
    L,
    D,
    Both,
}

impl AnchorParity {
    /// Returns the active [`Chirality`] slots probed for this mode.
    pub fn active_chirality_slots(self) -> Vec<Chirality> {
        match self {
            Self::L => vec![Chirality::L],
            Self::D => vec![Chirality::D],
            Self::Both => vec![Chirality::L, Chirality::D],
        }
    }
}

/// Energetic metric used to sort pairwise scan results and construct 20x20 matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankBy {
    /// Double-mutant cycle coupling `ddE_DMC = E(XY) - E(XA) - E(AY) + E(AA)`.
    Dmc,
    /// Mutation energy relative to the poly-D/L-Ala host `dE_Ala = E(XY) - E(AA)`.
    VsAla,
    /// Direct residue-pair BUDEFF interaction between residues `i` and `j`.
    DirectPair,
}

impl RankBy {
    /// Extracts the scalar score corresponding to this metric from a [`PairProbeRow`].
    pub fn extract(self, row: &PairProbeRow) -> f64 {
        match self {
            Self::Dmc => row.dde_dmc,
            Self::VsAla => row.de_vs_ala,
            Self::DirectPair => row.direct_total,
        }
    }

    /// Short file-safe identifier for this metric.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Dmc => "ddE_dmc",
            Self::VsAla => "dE_vs_ala",
            Self::DirectPair => "direct_pair",
        }
    }
}

/// Configuration for systematic pairwise composition scanning along a beta-helix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Host poly-(L/D-Ala) peptide length (default `22`).
    pub host_length: usize,
    /// N-terminal capping state (default [`NCap::Acetyl`]).
    pub n_cap: NCap,
    /// C-terminal capping state (default [`CCap::Amide`]).
    pub c_cap: CCap,
    /// Minor helical backbone parameters defining the beta-helix host geometry.
    pub minor_params: HelixParams,
    /// Relative partner offsets `j - i` to evaluate (default `[-1, 1, 5, 6, 7]`).
    pub offsets: Vec<isize>,
    /// Which anchor chirality slot(s) to scan (default [`AnchorParity::Both`]).
    pub anchor: AnchorParity,
    /// Whether Proline is included in the default 20-residue alphabet (default `true`).
    pub include_proline: bool,
    /// Optional explicit residue subset; defaults to [`CHEM_GROUP_ORDER`].
    #[serde(skip)]
    pub residues: Option<Vec<CanonicalResidue>>,
    /// Primary sorting metric (default [`RankBy::Dmc`]).
    pub rank_by: RankBy,
    /// Flag L-vs-D difference as significant when `max |ddE_L - ddE_D|` exceeds this threshold.
    pub parity_max_diff_threshold: f64,
    /// Flag L-vs-D difference as significant when RMSD exceeds this threshold.
    pub parity_rmsd_threshold: f64,
    /// When `true` (default), record isolated external `dASPR` packing failures as skipped pairs (`NaN` in matrices) instead of aborting the sweep.
    pub skip_pack_errors: bool,
    /// Optional Rayon worker thread limit (`None` uses all available CPUs).
    pub threads: Option<usize>,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            host_length: 22,
            n_cap: NCap::Acetyl,
            c_cap: CCap::Amide,
            minor_params: HelixParams::beta_6_3(),
            offsets: vec![-1, 1, 5, 6, 7],
            anchor: AnchorParity::Both,
            include_proline: true,
            residues: None,
            rank_by: RankBy::Dmc,
            parity_max_diff_threshold: 0.5,
            parity_rmsd_threshold: 0.15,
            skip_pack_errors: true,
            threads: None,
        }
    }
}

impl ProbeConfig {
    /// Returns the ordered list of amino acids evaluated in this scan.
    pub fn effective_residues(&self) -> Result<Vec<CanonicalResidue>, ProbeError> {
        let mut out = match &self.residues {
            Some(custom) => custom.clone(),
            None => CHEM_GROUP_ORDER
                .iter()
                .copied()
                .filter(|&r| self.include_proline || r != CanonicalResidue::Pro)
                .collect(),
        };
        let mut seen = HashSet::new();
        out.retain(|r| seen.insert(*r));
        if out.is_empty() {
            return Err(ProbeError::InvalidConfig(
                "residue alphabet must contain at least one amino acid".to_string(),
            ));
        }
        Ok(out)
    }
}

/// Energetic and structural record for a single `(anchor, offset, res_i, res_j)` variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairProbeRow {
    pub anchor_chirality: Chirality,
    pub anchor_seq_id: usize,
    pub offset: isize,
    pub partner_seq_id: usize,
    pub res_i_code: char,
    pub res_j_code: char,
    pub res_i_case: char,
    pub res_j_case: char,
    pub res_i_pdb: String,
    pub res_j_pdb: String,
    pub dde_dmc: f64,
    pub de_vs_ala: f64,
    pub e_xy: f64,
    pub e_xa: f64,
    pub e_ay: f64,
    pub e_aa: f64,
    pub direct_total: f64,
    pub direct_steric: f64,
    pub direct_desolvation: f64,
    pub direct_charge: f64,
    pub direct_sc_total: f64,
    pub direct_sc_steric: f64,
    pub direct_sc_desolvation: f64,
    pub direct_sc_charge: f64,
}

/// Statistical comparison between L-anchor and D-anchor double-mutant couplings at one offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityComparison {
    pub offset: isize,
    pub n_pairs: usize,
    pub mean_signed_diff: f64,
    pub rmsd: f64,
    pub max_abs_diff: f64,
    pub max_diff_pair: String,
    pub pearson_r: f64,
    pub top10_overlap: usize,
    pub flagged_significant: bool,
}

/// Pairwise variant skipped because an external packing call failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedPair {
    pub anchor_chirality: Chirality,
    pub anchor_seq_id: usize,
    pub offset: isize,
    pub partner_seq_id: usize,
    pub res_i_code: char,
    pub res_j_code: char,
    pub reason: String,
}

/// Full result returned by [`probe_pair_interactions`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub base_sequence: String,
    pub e_aa: f64,
    pub offsets: Vec<isize>,
    pub anchor_mode: AnchorParity,
    pub rank_by: RankBy,
    pub parity_max_diff_threshold: f64,
    pub parity_rmsd_threshold: f64,
    pub alphabet_codes: Vec<char>,
    pub rows: Vec<PairProbeRow>,
    pub parity_comparisons: Vec<ParityComparison>,
    pub skipped_pairs: Vec<SkippedPair>,
}

impl ProbeReport {
    /// Returns `true` if any offset exhibited a significant L-vs-D anchor difference.
    pub fn any_significant_parity_diff(&self) -> bool {
        self.parity_comparisons
            .iter()
            .any(|p| p.flagged_significant)
    }

    /// Formats a human-readable summary of L-vs-D anchor agreement per offset.
    pub fn parity_report(&self) -> String {
        if self.parity_comparisons.is_empty() {
            return "L-vs-D parity comparison: skipped (single anchor mode)\n".to_string();
        }
        let mut out = String::new();
        out.push_str(&format!(
            "=== L-vs-D Anchor Parity Check (thresholds: max|d| <= {:.3}, RMSD <= {:.3}) ===\n",
            self.parity_max_diff_threshold, self.parity_rmsd_threshold
        ));
        for p in &self.parity_comparisons {
            let status = if p.flagged_significant {
                "SIGNIFICANT_DIFF"
            } else {
                "OK"
            };
            out.push_str(&format!(
                "  offset={:+2} n={:3} mean(L-D)={:+.4} rmsd={:.4} max|L-D|={:.4} ({}) r={:.4} top10_overlap={}/{:<2} [{}]\n",
                p.offset,
                p.n_pairs,
                p.mean_signed_diff,
                p.rmsd,
                p.max_abs_diff,
                p.max_diff_pair,
                p.pearson_r,
                p.top10_overlap,
                p.n_pairs.min(10),
                status
            ));
        }
        if self.any_significant_parity_diff() {
            out.push_str(
                "WARNING: Significant L-vs-D anchor asymmetry detected; keep both anchors or inspect chiral helix geometry.\n",
            );
        } else {
            out.push_str(
                "NOTE: L and D anchors agree within thresholds; single-anchor scans are consistent.\n",
            );
        }
        out
    }

    /// Formats top-`top_k` stabilizing and destabilizing pairings for each `(anchor, offset)` plus parity check.
    pub fn format_summary(&self, top_k: usize) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Pairwise Beta-Helix Scan (host={}, E_AA={:.4}, rank_by={}, evaluated={}, skipped={}) ===\n",
            self.base_sequence,
            self.e_aa,
            self.rank_by.slug(),
            self.rows.len(),
            self.skipped_pairs.len()
        ));
        if !self.skipped_pairs.is_empty() {
            out.push_str(&format!(
                "NOTE: {} pair(s) skipped due to external packer errors (NaN in matrices):\n",
                self.skipped_pairs.len()
            ));
            for sp in self.skipped_pairs.iter().take(15) {
                out.push_str(&format!(
                    "  skipped {:?} seq {} -> {} (offset {:+}) {}-{}: {}\n",
                    sp.anchor_chirality,
                    sp.anchor_seq_id,
                    sp.partner_seq_id,
                    sp.offset,
                    sp.res_i_code,
                    sp.res_j_code,
                    sp.reason
                ));
            }
            if self.skipped_pairs.len() > 15 {
                out.push_str(&format!("  ... and {} more\n", self.skipped_pairs.len() - 15));
            }
        }
        out.push_str(&self.parity_report());

        let mut groups: BTreeSet<(u8, isize)> = BTreeSet::new();
        for row in &self.rows {
            let key = match row.anchor_chirality {
                Chirality::L => 0,
                Chirality::D => 1,
                Chirality::Achiral => 2,
            };
            groups.insert((key, row.offset));
        }

        for (anchor_ord, offset) in groups {
            let chirality = match anchor_ord {
                0 => Chirality::L,
                1 => Chirality::D,
                _ => Chirality::Achiral,
            };
            let mut slice: Vec<&PairProbeRow> = self
                .rows
                .iter()
                .filter(|r| r.anchor_chirality == chirality && r.offset == offset)
                .collect();
            slice.sort_by(|a, b| {
                self.rank_by
                    .extract(a)
                    .total_cmp(&self.rank_by.extract(b))
                    .then_with(|| a.res_i_code.cmp(&b.res_i_code))
                    .then_with(|| a.res_j_code.cmp(&b.res_j_code))
            });
            let Some(first) = slice.first() else {
                continue;
            };
            out.push_str(&format!(
                "\n--- Anchor={:?} (seq {} -> {}, offset {:+}) [n={}] ---\n",
                chirality,
                first.anchor_seq_id,
                first.partner_seq_id,
                offset,
                slice.len()
            ));
            let k = top_k.min(slice.len());
            out.push_str("  Most stabilizing:\n");
            for (rank, row) in slice.iter().take(k).enumerate() {
                out.push_str(&format!(
                    "    {:2}. {}{} ({}-{}) ddE_DMC={:+8.4} dE_Ala={:+8.4} direct={:+8.4} (st={:.3}, de={:.3}, ch={:+.3})\n",
                    rank + 1,
                    row.res_i_case,
                    row.res_j_case,
                    row.res_i_pdb,
                    row.res_j_pdb,
                    row.dde_dmc,
                    row.de_vs_ala,
                    row.direct_total,
                    row.direct_steric,
                    row.direct_desolvation,
                    row.direct_charge
                ));
            }
            out.push_str("  Most destabilizing:\n");
            for (idx, row) in slice.iter().rev().take(k).enumerate() {
                out.push_str(&format!(
                    "    {:2}. {}{} ({}-{}) ddE_DMC={:+8.4} dE_Ala={:+8.4} direct={:+8.4} (st={:.3}, de={:.3}, ch={:+.3})\n",
                    idx + 1,
                    row.res_i_case,
                    row.res_j_case,
                    row.res_i_pdb,
                    row.res_j_pdb,
                    row.dde_dmc,
                    row.de_vs_ala,
                    row.direct_total,
                    row.direct_steric,
                    row.direct_desolvation,
                    row.direct_charge
                ));
            }
        }
        out
    }

    /// Serializes all rows (already sorted by `rank_by`) as tidy CSV.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "anchor,anchor_seq_id,offset,partner_seq_id,res_i,res_j,res_i_case,res_j_case,res_i_pdb,res_j_pdb,ddE_dmc,dE_vs_ala,e_xy,e_xa,e_ay,e_aa,direct_total,direct_steric,direct_desolvation,direct_charge,direct_sc_total,direct_sc_steric,direct_sc_desolvation,direct_sc_charge\n",
        );
        for r in &self.rows {
            let anchor_str = match r.anchor_chirality {
                Chirality::L => "L",
                Chirality::D => "D",
                Chirality::Achiral => "Achiral",
            };
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                anchor_str,
                r.anchor_seq_id,
                r.offset,
                r.partner_seq_id,
                r.res_i_code,
                r.res_j_code,
                r.res_i_case,
                r.res_j_case,
                r.res_i_pdb,
                r.res_j_pdb,
                r.dde_dmc,
                r.de_vs_ala,
                r.e_xy,
                r.e_xa,
                r.e_ay,
                r.e_aa,
                r.direct_total,
                r.direct_steric,
                r.direct_desolvation,
                r.direct_charge,
                r.direct_sc_total,
                r.direct_sc_steric,
                r.direct_sc_desolvation,
                r.direct_sc_charge
            ));
        }
        out
    }

    /// Exports a square `N x N` matrix CSV (`row = res_i`, `col = res_j`) for `(anchor, offset, metric)`.
    pub fn matrix_csv(
        &self,
        anchor: Chirality,
        offset: isize,
        metric: RankBy,
    ) -> Result<String, ProbeError> {
        let lookup: HashMap<(char, char), f64> = self
            .rows
            .iter()
            .filter(|r| r.anchor_chirality == anchor && r.offset == offset)
            .map(|r| ((r.res_i_code, r.res_j_code), metric.extract(r)))
            .collect();
        if lookup.is_empty() {
            return Err(ProbeError::InvalidConfig(format!(
                "no rows found for anchor={anchor:?} offset={offset}"
            )));
        }

        let mut out = String::from("res_i");
        for code in &self.alphabet_codes {
            out.push(',');
            out.push(*code);
        }
        out.push('\n');

        for &ri in &self.alphabet_codes {
            out.push(ri);
            for &rj in &self.alphabet_codes {
                match lookup.get(&(ri, rj)) {
                    Some(val) => out.push_str(&format!(",{val:.6}")),
                    None => out.push_str(",NaN"),
                }
            }
            out.push('\n');
        }
        Ok(out)
    }
}

/// Validated geometric anchor layout derived from [`ProbeConfig`].
#[derive(Debug, Clone)]
struct ProbePlan {
    base_sequence: Sequence,
    anchors: Vec<(Chirality, usize)>,
    alphabet: Vec<CanonicalResidue>,
    offsets: Vec<isize>,
}

fn build_host_sequence(config: &ProbeConfig) -> Result<Sequence, ProbeError> {
    if config.host_length < 4 {
        return Err(ProbeError::InvalidConfig(format!(
            "host_length must be >= 4, got {}",
            config.host_length
        )));
    }
    let mut raw = String::with_capacity(config.host_length + 8);
    if config.n_cap == NCap::Acetyl {
        raw.push_str("Ac-");
    }
    for i in 0..config.host_length {
        raw.push(if i % 2 == 0 { 'A' } else { 'a' });
    }
    if config.c_cap == CCap::Amide {
        raw.push_str("-NH2");
    }
    Ok(Sequence::parse(&raw)?)
}

fn pick_centered_anchor(
    chirality: Chirality,
    ideal: f64,
    host_length: usize,
    offsets: &[isize],
) -> Result<usize, ProbeError> {
    let want_odd = match chirality {
        Chirality::L => 1usize,
        Chirality::D => 0usize,
        Chirality::Achiral => {
            return Err(ProbeError::InvalidConfig(
                "anchor chirality cannot be Achiral".to_string(),
            ));
        }
    };
    let mut best: Option<(f64, usize)> = None;
    for seq_id in 2..host_length {
        if seq_id % 2 != want_odd {
            continue;
        }
        let valid = offsets.iter().all(|&off| {
            let p = seq_id as isize + off;
            p >= 2 && p <= (host_length as isize - 1)
        });
        if !valid {
            continue;
        }
        let dist = (seq_id as f64 - ideal).abs();
        let replace = match best {
            None => true,
            Some((best_dist, best_id)) => {
                dist < best_dist - 1e-9 || ((dist - best_dist).abs() < 1e-9 && seq_id > best_id)
            }
        };
        if replace {
            best = Some((dist, seq_id));
        }
    }
    best.map(|(_, id)| id).ok_or_else(|| {
        ProbeError::InvalidConfig(format!(
            "host_length={} cannot place an interior {:?}-anchor across offsets {:?}",
            host_length, chirality, offsets
        ))
    })
}

fn validate_and_plan(config: &ProbeConfig) -> Result<ProbePlan, ProbeError> {
    if config.offsets.is_empty() {
        return Err(ProbeError::InvalidConfig(
            "offsets list cannot be empty".to_string(),
        ));
    }
    let mut seen_offsets = HashSet::new();
    for &off in &config.offsets {
        if off == 0 {
            return Err(ProbeError::InvalidConfig(
                "offset 0 (self-interaction) is invalid".to_string(),
            ));
        }
        if !seen_offsets.insert(off) {
            return Err(ProbeError::InvalidConfig(format!(
                "duplicate offset {off} in ProbeConfig"
            )));
        }
    }
    let min_off = config.offsets.iter().copied().min().unwrap_or(0);
    let max_off = config.offsets.iter().copied().max().unwrap_or(0);
    let ideal = (config.host_length as f64 + 1.0 - (min_off + max_off) as f64) * 0.5;

    let base_sequence = build_host_sequence(config)?;
    let alphabet = config.effective_residues()?;
    let mut anchors = Vec::new();
    for parity in config.anchor.active_chirality_slots() {
        let seq_id = pick_centered_anchor(parity, ideal, config.host_length, &config.offsets)?;
        anchors.push((parity, seq_id));
    }

    Ok(ProbePlan {
        base_sequence,
        anchors,
        alphabet,
        offsets: config.offsets.clone(),
    })
}

fn mutate_sequence(base: &Sequence, mutations: &[(usize, CanonicalResidue)]) -> Sequence {
    let mut seq = base.clone();
    for &(seq_id, aa) in mutations {
        let res: &mut ResidueSpec = &mut seq.residues[seq_id - 1];
        res.residue = aa;
        res.chirality = if aa == CanonicalResidue::Gly {
            Chirality::Achiral
        } else {
            res.backbone_parity
        };
    }
    seq
}

fn pack_and_parse(
    sequence: &Sequence,
    params: &HelixParams,
    daspr_cfg: &DasprConfig,
) -> Result<ScoredStructure, ProbeError> {
    let backbone_seq = sequence.as_poly_ala(true);
    let chain = build_beta_helix(&backbone_seq, params, 'A')
        .map_err(|e| ProbeError::Evaluation(e.to_string()))?;
    let pdb = pack_chain(&chain, sequence, daspr_cfg).map_err(|e| {
        ProbeError::Evaluation(format!("seq={} failed: {e}", sequence.to_daspr_string()))
    })?;
    ScoredStructure::from_pdb_str(&pdb).map_err(|e| ProbeError::Evaluation(e.to_string()))
}

fn canonical_one_letter(aa: CanonicalResidue) -> char {
    aa.case_char(Chirality::L)
}

#[derive(Clone)]
struct SingleEval {
    structure: ScoredStructure,
    total_energy: f64,
}

type SingleKey = (usize, CanonicalResidue);
type SingleOutcome = Result<SingleEval, String>;
type SingleCache = HashMap<SingleKey, SingleOutcome>;

#[derive(Clone, Copy)]
struct DoubleJob {
    anchor_chirality: Chirality,
    anchor_seq_id: usize,
    offset: isize,
    partner_seq_id: usize,
    res_i: CanonicalResidue,
    res_j: CanonicalResidue,
}

fn score_pair_terms(
    structure: &ScoredStructure,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    anchor_seq_id: usize,
    partner_seq_id: usize,
) -> (
    crate::energy::EnergyScore,
    crate::energy::EnergyScore,
) {
    let direct_full = direct_residue_pair_energy(
        structure,
        ff,
        'A',
        anchor_seq_id as i32,
        partner_seq_id as i32,
        false,
        energy_opts,
    );
    let direct_sc = direct_residue_pair_energy(
        structure,
        ff,
        'A',
        anchor_seq_id as i32,
        partner_seq_id as i32,
        true,
        energy_opts,
    );
    (direct_full, direct_sc)
}

fn evaluate_all_variants(
    plan: &ProbePlan,
    config: &ProbeConfig,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<(f64, Vec<PairProbeRow>, Vec<SkippedPair>), ProbeError> {
    let baseline_struct = pack_and_parse(&plan.base_sequence, &config.minor_params, daspr_cfg)?;
    let e_aa = internal_energy(&baseline_struct, ff, energy_opts).total;
    let base_eval = SingleEval {
        structure: baseline_struct,
        total_energy: e_aa,
    };

    let mut single_keys = BTreeMap::new();
    for &(_parity, anchor_seq) in &plan.anchors {
        for &offset in &plan.offsets {
            let partner_seq = (anchor_seq as isize + offset) as usize;
            for &aa in &plan.alphabet {
                if aa != CanonicalResidue::Ala {
                    single_keys.insert((anchor_seq, canonical_one_letter(aa)), aa);
                    single_keys.insert((partner_seq, canonical_one_letter(aa)), aa);
                }
            }
        }
    }

    let single_jobs: Vec<SingleKey> =
        single_keys.into_iter().map(|((s, _), r)| (s, r)).collect();
    let single_results: Result<Vec<(SingleKey, SingleOutcome)>, ProbeError> = single_jobs
            .par_iter()
            .map(|&(seq_id, aa)| {
                let mut_seq = mutate_sequence(&plan.base_sequence, &[(seq_id, aa)]);
                match pack_and_parse(&mut_seq, &config.minor_params, daspr_cfg) {
                    Ok(structure) => {
                        let total_energy = internal_energy(&structure, ff, energy_opts).total;
                        Ok((
                            (seq_id, aa),
                            Ok(SingleEval {
                                structure,
                                total_energy,
                            }),
                        ))
                    }
                    Err(err) => {
                        if config.skip_pack_errors {
                            Ok(((seq_id, aa), Err(err.to_string())))
                        } else {
                            Err(err)
                        }
                    }
                }
            })
            .collect();
    let single_cache: SingleCache = single_results?.into_iter().collect();

    let lookup_single = |seq_id: usize, aa: CanonicalResidue| -> Result<&SingleEval, &str> {
        if aa == CanonicalResidue::Ala {
            Ok(&base_eval)
        } else {
            match single_cache
                .get(&(seq_id, aa))
                .expect("every non-Ala single-site mutation was precomputed")
            {
                Ok(eval) => Ok(eval),
                Err(msg) => Err(msg.as_str()),
            }
        }
    };

    let mut jobs = Vec::with_capacity(
        plan.anchors.len() * plan.offsets.len() * plan.alphabet.len() * plan.alphabet.len(),
    );
    for &(anchor_chirality, anchor_seq_id) in &plan.anchors {
        for &offset in &plan.offsets {
            let partner_seq_id = (anchor_seq_id as isize + offset) as usize;
            for &res_i in &plan.alphabet {
                for &res_j in &plan.alphabet {
                    jobs.push(DoubleJob {
                        anchor_chirality,
                        anchor_seq_id,
                        offset,
                        partner_seq_id,
                        res_i,
                        res_j,
                    });
                }
            }
        }
    }

    enum JobOutcome {
        Row(PairProbeRow),
        Skipped(SkippedPair),
    }

    let outcomes: Result<Vec<JobOutcome>, ProbeError> = jobs
        .par_iter()
        .map(|job| {
            let make_skipped = |reason: String| {
                JobOutcome::Skipped(SkippedPair {
                    anchor_chirality: job.anchor_chirality,
                    anchor_seq_id: job.anchor_seq_id,
                    offset: job.offset,
                    partner_seq_id: job.partner_seq_id,
                    res_i_code: canonical_one_letter(job.res_i),
                    res_j_code: canonical_one_letter(job.res_j),
                    reason,
                })
            };

            let xa = match lookup_single(job.anchor_seq_id, job.res_i) {
                Ok(v) => v,
                Err(msg) => return Ok(make_skipped(format!("single-site res_i failed: {msg}"))),
            };
            let ay = match lookup_single(job.partner_seq_id, job.res_j) {
                Ok(v) => v,
                Err(msg) => return Ok(make_skipped(format!("single-site res_j failed: {msg}"))),
            };
            let e_xa = xa.total_energy;
            let e_ay = ay.total_energy;

            let owned_xy: Option<(f64, ScoredStructure)> =
                if job.res_i == CanonicalResidue::Ala || job.res_j == CanonicalResidue::Ala {
                    None
                } else {
                    let mut_seq = mutate_sequence(
                        &plan.base_sequence,
                        &[
                            (job.anchor_seq_id, job.res_i),
                            (job.partner_seq_id, job.res_j),
                        ],
                    );
                    match pack_and_parse(&mut_seq, &config.minor_params, daspr_cfg) {
                        Ok(structure) => {
                            let energy = internal_energy(&structure, ff, energy_opts).total;
                            Some((energy, structure))
                        }
                        Err(err) => {
                            if config.skip_pack_errors {
                                return Ok(make_skipped(err.to_string()));
                            }
                            return Err(err);
                        }
                    }
                };

            let (e_xy, xy_struct): (f64, &ScoredStructure) = match &owned_xy {
                Some((e, s)) => (*e, s),
                None if job.res_i == CanonicalResidue::Ala => (e_ay, &ay.structure),
                None => (e_xa, &xa.structure),
            };

            let (direct_full, direct_sc) = score_pair_terms(
                xy_struct,
                ff,
                energy_opts,
                job.anchor_seq_id,
                job.partner_seq_id,
            );

            let res_i_spec = &plan.base_sequence.residues[job.anchor_seq_id - 1];
            let res_j_spec = &plan.base_sequence.residues[job.partner_seq_id - 1];
            let res_i_chir = if job.res_i == CanonicalResidue::Gly {
                Chirality::Achiral
            } else {
                res_i_spec.backbone_parity
            };
            let res_j_chir = if job.res_j == CanonicalResidue::Gly {
                Chirality::Achiral
            } else {
                res_j_spec.backbone_parity
            };

            Ok(JobOutcome::Row(PairProbeRow {
                anchor_chirality: job.anchor_chirality,
                anchor_seq_id: job.anchor_seq_id,
                offset: job.offset,
                partner_seq_id: job.partner_seq_id,
                res_i_code: canonical_one_letter(job.res_i),
                res_j_code: canonical_one_letter(job.res_j),
                res_i_case: job.res_i.case_char(res_i_chir),
                res_j_case: job.res_j.case_char(res_j_chir),
                res_i_pdb: job.res_i.pdb_code(res_i_chir).to_string(),
                res_j_pdb: job.res_j.pdb_code(res_j_chir).to_string(),
                dde_dmc: e_xy - e_xa - e_ay + e_aa,
                de_vs_ala: e_xy - e_aa,
                e_xy,
                e_xa,
                e_ay,
                e_aa,
                direct_total: direct_full.total,
                direct_steric: direct_full.steric,
                direct_desolvation: direct_full.desolvation,
                direct_charge: direct_full.charge,
                direct_sc_total: direct_sc.total,
                direct_sc_steric: direct_sc.steric,
                direct_sc_desolvation: direct_sc.desolvation,
                direct_sc_charge: direct_sc.charge,
            }))
        })
        .collect();

    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for item in outcomes? {
        match item {
            JobOutcome::Row(r) => rows.push(r),
            JobOutcome::Skipped(s) => skipped.push(s),
        }
    }
    Ok((e_aa, rows, skipped))
}

/// Computes per-offset statistical agreement between L-anchor and D-anchor `ddE_DMC` values.
pub fn compute_parity_comparisons(
    rows: &[PairProbeRow],
    offsets: &[isize],
    max_diff_threshold: f64,
    rmsd_threshold: f64,
) -> Vec<ParityComparison> {
    let mut comparisons = Vec::new();
    for &offset in offsets {
        let mut l_map: HashMap<(char, char), f64> = HashMap::new();
        let mut d_map: HashMap<(char, char), f64> = HashMap::new();
        for r in rows.iter().filter(|r| r.offset == offset) {
            match r.anchor_chirality {
                Chirality::L => {
                    l_map.insert((r.res_i_code, r.res_j_code), r.dde_dmc);
                }
                Chirality::D => {
                    d_map.insert((r.res_i_code, r.res_j_code), r.dde_dmc);
                }
                Chirality::Achiral => {}
            }
        }
        if l_map.is_empty() || d_map.is_empty() {
            continue;
        }

        let mut paired: Vec<((char, char), f64, f64)> = Vec::new();
        for (&pair, &vl) in &l_map {
            if let Some(&vd) = d_map.get(&pair) {
                paired.push((pair, vl, vd));
            }
        }
        if paired.is_empty() {
            continue;
        }

        let n = paired.len() as f64;
        let mean_l = paired.iter().map(|x| x.1).sum::<f64>() / n;
        let mean_d = paired.iter().map(|x| x.2).sum::<f64>() / n;
        let mut sum_signed = 0.0;
        let mut sum_sq = 0.0;
        let mut max_abs = -1.0;
        let mut max_pair = String::new();
        let mut cov = 0.0;
        let mut var_l = 0.0;
        let mut var_d = 0.0;

        for &((ci, cj), vl, vd) in &paired {
            let diff = vl - vd;
            sum_signed += diff;
            sum_sq += diff * diff;
            let abs_diff = diff.abs();
            if abs_diff > max_abs {
                max_abs = abs_diff;
                max_pair = format!("{ci}-{cj}");
            }
            let dl = vl - mean_l;
            let dd = vd - mean_d;
            cov += dl * dd;
            var_l += dl * dl;
            var_d += dd * dd;
        }

        let rmsd = (sum_sq / n).sqrt();
        let denom = (var_l * var_d).sqrt();
        let pearson_r = if denom > 1e-12 {
            cov / denom
        } else if rmsd < 1e-12 {
            1.0
        } else {
            0.0
        };

        let top_k = paired.len().min(10);
        let mut l_sorted: Vec<((char, char), f64)> =
            paired.iter().map(|&(p, vl, _)| (p, vl)).collect();
        let mut d_sorted: Vec<((char, char), f64)> =
            paired.iter().map(|&(p, _, vd)| (p, vd)).collect();
        l_sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        d_sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let top_l: HashSet<(char, char)> = l_sorted.into_iter().take(top_k).map(|x| x.0).collect();
        let top10_overlap = d_sorted
            .into_iter()
            .take(top_k)
            .filter(|x| top_l.contains(&x.0))
            .count();

        let flagged_significant = max_abs > max_diff_threshold || rmsd > rmsd_threshold;
        comparisons.push(ParityComparison {
            offset,
            n_pairs: paired.len(),
            mean_signed_diff: sum_signed / n,
            rmsd,
            max_abs_diff: max_abs,
            max_diff_pair: max_pair,
            pearson_r,
            top10_overlap,
            flagged_significant,
        });
    }
    comparisons
}

/// Runs the systematic pairwise side-chain scan using `dASPR` and native BUDEFF scoring.
pub fn probe_pair_interactions(
    config: &ProbeConfig,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<ProbeReport, ProbeError> {
    resolve_daspr_binary(daspr_cfg)?;
    let plan = validate_and_plan(config)?;

    let (e_aa, mut rows, skipped_pairs) = if let Some(threads) = config.threads {
        if threads == 0 {
            return Err(ProbeError::InvalidConfig(
                "threads must be >= 1".to_string(),
            ));
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| ProbeError::Evaluation(e.to_string()))?;
        pool.install(|| evaluate_all_variants(&plan, config, ff, energy_opts, daspr_cfg))?
    } else {
        evaluate_all_variants(&plan, config, ff, energy_opts, daspr_cfg)?
    };

    rows.sort_by(|a, b| {
        config
            .rank_by
            .extract(a)
            .total_cmp(&config.rank_by.extract(b))
            .then_with(|| a.offset.cmp(&b.offset))
            .then_with(|| a.res_i_code.cmp(&b.res_i_code))
            .then_with(|| a.res_j_code.cmp(&b.res_j_code))
    });

    let parity_comparisons = compute_parity_comparisons(
        &rows,
        &plan.offsets,
        config.parity_max_diff_threshold,
        config.parity_rmsd_threshold,
    );

    let alphabet_codes = plan
        .alphabet
        .iter()
        .map(|&aa| canonical_one_letter(aa))
        .collect();

    Ok(ProbeReport {
        base_sequence: plan.base_sequence.to_daspr_string(),
        e_aa,
        offsets: plan.offsets,
        anchor_mode: config.anchor,
        rank_by: config.rank_by,
        parity_max_diff_threshold: config.parity_max_diff_threshold,
        parity_rmsd_threshold: config.parity_rmsd_threshold,
        alphabet_codes,
        rows,
        parity_comparisons,
        skipped_pairs,
    })
}

/// Errors produced during pairwise beta-helix composition scanning.
#[derive(Debug)]
pub enum ProbeError {
    InvalidConfig(String),
    Input(InputError),
    Sidechain(SidechainError),
    Evaluation(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid probe configuration: {msg}"),
            Self::Input(err) => write!(f, "{err}"),
            Self::Sidechain(err) => write!(f, "{err}"),
            Self::Evaluation(msg) => write!(f, "probe evaluation error: {msg}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<InputError> for ProbeError {
    fn from(err: InputError) -> Self {
        Self::Input(err)
    }
}

impl From<SidechainError> for ProbeError {
    fn from(err: SidechainError) -> Self {
        Self::Sidechain(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_plan_places_interior_anchors_and_includes_proline() {
        let cfg = ProbeConfig::default();
        let plan = validate_and_plan(&cfg).unwrap();
        assert_eq!(plan.alphabet.len(), 20);
        assert!(plan.alphabet.contains(&CanonicalResidue::Pro));
        assert_eq!(plan.anchors, vec![(Chirality::L, 9), (Chirality::D, 8)]);
        assert_eq!(plan.base_sequence.n_cap, NCap::Acetyl);
        assert_eq!(plan.base_sequence.c_cap, CCap::Amide);
        assert_eq!(
            plan.base_sequence.to_daspr_string(),
            "AaAaAaAaAaAaAaAaAaAaAa"
        );

        let no_pro = ProbeConfig {
            include_proline: false,
            ..cfg
        };
        let al = no_pro.effective_residues().unwrap();
        assert_eq!(al.len(), 19);
        assert!(!al.contains(&CanonicalResidue::Pro));
    }

    #[test]
    fn rejects_invalid_host_length_or_zero_offset() {
        let too_short = ProbeConfig {
            host_length: 10,
            ..ProbeConfig::default()
        };
        assert!(validate_and_plan(&too_short).is_err());

        let zero_off = ProbeConfig {
            offsets: vec![0, 6],
            ..ProbeConfig::default()
        };
        assert!(validate_and_plan(&zero_off).is_err());
    }

    #[test]
    fn parity_comparison_flags_significant_asymmetry_and_accepts_identical() {
        let make_row = |anchor: Chirality, ri: char, rj: char, dde: f64| PairProbeRow {
            anchor_chirality: anchor,
            anchor_seq_id: if anchor == Chirality::L { 9 } else { 10 },
            offset: 6,
            partner_seq_id: if anchor == Chirality::L { 15 } else { 16 },
            res_i_code: ri,
            res_j_code: rj,
            res_i_case: ri,
            res_j_case: rj,
            res_i_pdb: "ALA".to_string(),
            res_j_pdb: "ALA".to_string(),
            dde_dmc: dde,
            de_vs_ala: dde,
            e_xy: dde,
            e_xa: 0.0,
            e_ay: 0.0,
            e_aa: 0.0,
            direct_total: 0.0,
            direct_steric: 0.0,
            direct_desolvation: 0.0,
            direct_charge: 0.0,
            direct_sc_total: 0.0,
            direct_sc_steric: 0.0,
            direct_sc_desolvation: 0.0,
            direct_sc_charge: 0.0,
        };

        let identical = vec![
            make_row(Chirality::L, 'A', 'A', 0.0),
            make_row(Chirality::D, 'A', 'A', 0.0),
            make_row(Chirality::L, 'K', 'D', -1.8),
            make_row(Chirality::D, 'K', 'D', -1.8),
        ];
        let cmp_ok = compute_parity_comparisons(&identical, &[6], 0.5, 0.15);
        assert_eq!(cmp_ok.len(), 1);
        assert!(!cmp_ok[0].flagged_significant);
        assert!((cmp_ok[0].pearson_r - 1.0).abs() < 1e-12);
        assert!(cmp_ok[0].rmsd < 1e-12);

        let skewed = vec![
            make_row(Chirality::L, 'A', 'A', 0.0),
            make_row(Chirality::D, 'A', 'A', 0.0),
            make_row(Chirality::L, 'W', 'P', -0.4),
            make_row(Chirality::D, 'W', 'P', 1.2),
        ];
        let cmp_bad = compute_parity_comparisons(&skewed, &[6], 0.5, 0.15);
        assert!(cmp_bad[0].flagged_significant);
        assert_eq!(cmp_bad[0].max_diff_pair, "W-P");
        assert!((cmp_bad[0].max_abs_diff - 1.6).abs() < 1e-12);
    }

    #[test]
    fn report_csv_and_matrix_csv_format_correctly() {
        let row_aa = PairProbeRow {
            anchor_chirality: Chirality::L,
            anchor_seq_id: 9,
            offset: 6,
            partner_seq_id: 15,
            res_i_code: 'A',
            res_j_code: 'A',
            res_i_case: 'A',
            res_j_case: 'A',
            res_i_pdb: "ALA".to_string(),
            res_j_pdb: "ALA".to_string(),
            dde_dmc: 0.0,
            de_vs_ala: 0.0,
            e_xy: -10.0,
            e_xa: -10.0,
            e_ay: -10.0,
            e_aa: -10.0,
            direct_total: -0.2,
            direct_steric: 0.0,
            direct_desolvation: -0.2,
            direct_charge: 0.0,
            direct_sc_total: -0.1,
            direct_sc_steric: 0.0,
            direct_sc_desolvation: -0.1,
            direct_sc_charge: 0.0,
        };
        let report = ProbeReport {
            base_sequence: "Ac-AaAaAaAaAaAaAaAaAaAaAa-NH2".to_string(),
            e_aa: -10.0,
            offsets: vec![6],
            anchor_mode: AnchorParity::L,
            rank_by: RankBy::Dmc,
            parity_max_diff_threshold: 0.5,
            parity_rmsd_threshold: 0.15,
            alphabet_codes: vec!['A'],
            rows: vec![row_aa],
            parity_comparisons: Vec::new(),
            skipped_pairs: Vec::new(),
        };
        let csv = report.to_csv();
        assert!(csv.starts_with("anchor,anchor_seq_id,offset,partner_seq_id,res_i,res_j,"));
        assert!(csv.contains("L,9,6,15,A,A,A,A,ALA,ALA,0.000000,0.000000,"));
        let mat = report.matrix_csv(Chirality::L, 6, RankBy::Dmc).unwrap();
        assert_eq!(mat, "res_i,A\nA,0.000000\n");
    }
}
