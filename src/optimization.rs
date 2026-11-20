//! Population-based Differential Evolution (DE) bundle parameter optimisation.
//!
//! Modeled after ISAMBARD's `BaseOptimizer` and `DE` algorithms using normalized `[-1, 1]`
//! parameter encoding, lognormal-scaled differential mutation, optional ring neighbourhood
//! topology, deterministic PRNG seeding, `rayon` parallelism, implicit `dASPR` packing,
//! and native Rust BUDEFF scoring.

use std::fmt;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::LogNormal;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::bundle::{build_bundle, bundle_target_sequences};
use crate::energy::{
    BudeForceField, BundleEnergyReport, EnergyOptions, ScoredStructure, evaluate_bundle,
};
use crate::input::{
    BundleSpec, HelixParams, HelixPlacementParams, InputError, Orientation, Symmetry, TwistMode,
};
use crate::sidechain::{DasprConfig, SidechainError, pack_bundle, resolve_daspr_binary};

/// Objective energy term minimised during Differential Evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptObjective {
    /// Total assembly BUDEFF internal energy (default).
    #[default]
    Internal,
    /// Inter-chain BUDEFF interaction energy only.
    Interaction,
}

/// Bounded symmetric search range `[mean - half_range, mean + half_range]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    #[serde(alias = "value")]
    pub mean: f64,
    #[serde(alias = "range")]
    pub half_range: f64,
}

/// Optimisable parameter: either fixed (`Static`) or searched over `[mean ± half_range]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ParamValue {
    Static(f64),
    Dynamic(ParamRange),
}

impl ParamValue {
    /// Creates a fixed parameter.
    pub fn fixed(val: f64) -> Self {
        Self::Static(val)
    }

    /// Creates a dynamic parameter centered at `mean` with `±half_range`.
    pub fn dynamic(mean: f64, half_range: f64) -> Self {
        if half_range == 0.0 {
            Self::Static(mean)
        } else {
            Self::Dynamic(ParamRange { mean, half_range })
        }
    }

    /// Returns the baseline/mean value.
    pub fn mean(self) -> f64 {
        match self {
            Self::Static(v) => v,
            Self::Dynamic(r) => r.mean,
        }
    }

    fn validate(self, label: &str) -> Result<(), OptError> {
        match self {
            Self::Static(v) => {
                if !v.is_finite() {
                    return Err(OptError::InvalidSpec(format!(
                        "parameter '{label}' must be finite"
                    )));
                }
            }
            Self::Dynamic(r) => {
                if !r.mean.is_finite() || !r.half_range.is_finite() || r.half_range < 0.0 {
                    return Err(OptError::InvalidSpec(format!(
                        "parameter '{label}' must have finite mean and non-negative range"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for ParamValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Scalar(f64),
            Tuple(f64, f64),
            Map(ParamRange),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Scalar(v) => Ok(Self::Static(v)),
            Repr::Tuple(mean, half_range) => Ok(Self::dynamic(mean, half_range)),
            Repr::Map(r) => Ok(Self::dynamic(r.mean, r.half_range)),
        }
    }
}

/// Optimisation specification for a single helix (or symmetry template helix).
///
/// Minor-helix internal fold parameters (`radius_l`, `radius_d`, `tilt_ld_deg`, `tilt_dl_deg`)
/// remain fixed on `minor_params` to preserve a regular underlying beta-helix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptHelixSpec {
    pub sequence: String,
    pub bundle_radius: ParamValue,
    pub pitch: Option<ParamValue>,
    pub phi_c_alpha_deg: ParamValue,
    pub z_offset: ParamValue,
    pub azimuth_deg: Option<ParamValue>,
    pub residues_per_turn: Option<ParamValue>,
    pub orientation: Orientation,
    pub twist_mode: TwistMode,
    pub minor_params: HelixParams,
    pub chain_id: Option<char>,
}

impl Default for OptHelixSpec {
    fn default() -> Self {
        Self {
            sequence: "AaAaAaAaAaAaAaA".to_string(),
            bundle_radius: ParamValue::dynamic(10.5, 2.5),
            pitch: Some(ParamValue::dynamic(140.0, 50.0)),
            phi_c_alpha_deg: ParamValue::dynamic(0.0, 180.0),
            z_offset: ParamValue::fixed(0.0),
            azimuth_deg: None,
            residues_per_turn: None,
            orientation: Orientation::Parallel,
            twist_mode: TwistMode::CoupledDarboux,
            minor_params: HelixParams::beta_6_3(),
            chain_id: None,
        }
    }
}

/// Hyperparameters for Differential Evolution (`DE`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeConfig {
    /// Population size per generation.
    pub pop_size: usize,
    /// Number of evolutionary generations.
    pub generations: usize,
    /// Crossover probability in `[0.0, 1.0]` (ISAMBARD default: `0.75`).
    pub cxpb: f64,
    /// Differential scaling factor (ISAMBARD default: `1.0`).
    pub diff_weight: f64,
    /// Optional ring neighbourhood radius (samples `a, b, c` from `±neighbours`).
    pub neighbours: Option<usize>,
    /// Deterministic PRNG seed (`ChaCha8Rng`).
    pub seed: u64,
    /// Optional `rayon` worker thread count (defaults to global pool).
    pub threads: Option<usize>,
    /// Record every evaluated individual for funnel analysis.
    pub store_history: bool,
}

impl Default for DeConfig {
    fn default() -> Self {
        Self {
            pop_size: 20,
            generations: 20,
            cxpb: 0.75,
            diff_weight: 1.0,
            neighbours: None,
            seed: 42,
            threads: None,
            store_history: true,
        }
    }
}

impl DeConfig {
    /// Validates population size, crossover probability, differential weight, and neighbourhood bounds.
    pub fn validate(&self) -> Result<(), OptError> {
        if self.pop_size < 4 {
            return Err(OptError::InvalidSpec(
                "pop_size must be at least 4 for DE sampling".to_string(),
            ));
        }
        if !self.cxpb.is_finite() || !(0.0..=1.0).contains(&self.cxpb) {
            return Err(OptError::InvalidSpec(
                "cxpb must be within [0.0, 1.0]".to_string(),
            ));
        }
        if !self.diff_weight.is_finite() || self.diff_weight <= 0.0 {
            return Err(OptError::InvalidSpec(
                "diff_weight must be positive and finite".to_string(),
            ));
        }
        if let Some(k) = self.neighbours
            && (k < 2 || 2 * k + 1 > self.pop_size)
        {
            return Err(OptError::InvalidSpec(format!(
                "neighbours={k} requires 2 <= neighbours and 2*neighbours + 1 <= pop_size ({})",
                self.pop_size
            )));
        }
        Ok(())
    }
}

/// Complete YAML/API specification for bundle Differential Evolution optimisation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizeSpec {
    pub symmetry: Symmetry,
    pub keep_gly: bool,
    pub objective: OptObjective,
    pub de: DeConfig,
    pub helices: Vec<OptHelixSpec>,
}

impl Default for OptimizeSpec {
    fn default() -> Self {
        Self {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            objective: OptObjective::Internal,
            de: DeConfig::default(),
            helices: vec![OptHelixSpec::default()],
        }
    }
}

/// Descriptor for a single active dynamic parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDescriptor {
    pub label: String,
    pub mean: f64,
    pub half_range: f64,
}

impl ParamDescriptor {
    fn decode(&self, norm: f64) -> f64 {
        self.mean + norm * self.half_range
    }
}

impl OptimizeSpec {
    /// Extracts active dynamic parameters and validates the specification.
    pub fn validate_and_descriptors(&self) -> Result<Vec<ParamDescriptor>, OptError> {
        self.de.validate()?;
        if self.helices.is_empty() {
            return Err(OptError::InvalidSpec(
                "optimize spec must define at least one helix".to_string(),
            ));
        }
        match self.symmetry {
            Symmetry::Cn(n) | Symmetry::Dn(n) => {
                if n == 0 {
                    return Err(OptError::InvalidSpec(
                        "symmetry order n must be >= 1".to_string(),
                    ));
                }
                if self.helices.len() != 1 {
                    return Err(OptError::InvalidSpec(
                        "symmetric Cn/Dn optimization must specify exactly one template helix"
                            .to_string(),
                    ));
                }
            }
            Symmetry::None => {}
        }

        let mut descriptors = Vec::new();
        for (idx, h) in self.helices.iter().enumerate() {
            let prefix = if self.helices.len() == 1 {
                String::new()
            } else {
                format!("helix{idx}.")
            };
            let mut check = |name: &str, val: ParamValue| -> Result<(), OptError> {
                let full_label = format!("{prefix}{name}");
                val.validate(&full_label)?;
                if let ParamValue::Dynamic(r) = val
                    && r.half_range > 0.0
                {
                    descriptors.push(ParamDescriptor {
                        label: full_label,
                        mean: r.mean,
                        half_range: r.half_range,
                    });
                }
                Ok(())
            };

            check("bundle_radius", h.bundle_radius)?;
            if let Some(p) = h.pitch {
                check("pitch", p)?;
            }
            check("phi_c_alpha_deg", h.phi_c_alpha_deg)?;
            check("z_offset", h.z_offset)?;
            if let Some(az) = h.azimuth_deg {
                check("azimuth_deg", az)?;
            }
            if let Some(rpt) = h.residues_per_turn {
                check("residues_per_turn", rpt)?;
            }
        }

        if descriptors.is_empty() {
            return Err(OptError::InvalidSpec(
                "at least one dynamic parameter with range > 0 is required for optimisation"
                    .to_string(),
            ));
        }

        // Validate the baseline (normalized = 0) BundleSpec catches invalid sequences/minor params early.
        let baseline = self.decode_bundle_spec(&vec![0.0; descriptors.len()])?;
        baseline.validate()?;
        Ok(descriptors)
    }

    /// Decodes a normalized vector in `[-1, 1]^dim` into a concrete [`BundleSpec`].
    pub fn decode_bundle_spec(&self, norm: &[f64]) -> Result<BundleSpec, OptError> {
        let mut cursor = 0usize;
        let mut next_val = |val: ParamValue| -> Result<f64, OptError> {
            match val {
                ParamValue::Static(v) => Ok(v),
                ParamValue::Dynamic(r) => {
                    if r.half_range == 0.0 {
                        Ok(r.mean)
                    } else {
                        let x = *norm.get(cursor).ok_or_else(|| {
                            OptError::InvalidSpec("normalized vector length mismatch".to_string())
                        })?;
                        cursor += 1;
                        Ok(r.mean + x * r.half_range)
                    }
                }
            }
        };

        let mut helices = Vec::with_capacity(self.helices.len());
        for h in &self.helices {
            let bundle_radius = next_val(h.bundle_radius)?;
            let pitch = match h.pitch {
                Some(p) => Some(next_val(p)?),
                None => None,
            };
            let phi_c_alpha_deg = next_val(h.phi_c_alpha_deg)?;
            let z_offset = next_val(h.z_offset)?;
            let azimuth_deg = match h.azimuth_deg {
                Some(az) => Some(next_val(az)?),
                None => None,
            };
            let mut minor_params = h.minor_params.clone();
            if let Some(rpt) = h.residues_per_turn {
                minor_params.residues_per_turn = next_val(rpt)?;
            }

            helices.push(HelixPlacementParams {
                sequence: h.sequence.clone(),
                bundle_radius,
                pitch,
                phi_c_alpha_deg,
                z_offset,
                azimuth_deg,
                orientation: h.orientation,
                twist_mode: h.twist_mode,
                minor_params,
                chain_id: h.chain_id,
            });
        }

        Ok(BundleSpec {
            symmetry: self.symmetry,
            keep_gly: self.keep_gly,
            pack: true,
            helices,
        })
    }
}

/// Summary statistics for a single DE generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub evals: usize,
    pub min: f64,
    pub avg: f64,
    pub std: f64,
    pub max: f64,
}

/// Record of one evaluated candidate for energy-funnel analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecord {
    pub generation: usize,
    pub eval_id: usize,
    pub energy: f64,
    pub norm_params: Vec<f64>,
    pub decoded_params: Vec<f64>,
}

/// Output returned by the generic [`differential_evolution`] solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeSolverResult {
    pub best_norm: Vec<f64>,
    pub best_energy: f64,
    pub total_evals: usize,
    pub logbook: Vec<GenerationStats>,
    pub history: Vec<EvalRecord>,
}

fn sample_three_distinct(rng: &mut ChaCha8Rng, pool: &[usize]) -> (usize, usize, usize) {
    let n = pool.len();
    let i0 = rng.random_range(0..n);
    let mut i1 = rng.random_range(0..n);
    while i1 == i0 {
        i1 = rng.random_range(0..n);
    }
    let mut i2 = rng.random_range(0..n);
    while i2 == i0 || i2 == i1 {
        i2 = rng.random_range(0..n);
    }
    (pool[i0], pool[i1], pool[i2])
}

fn build_neighbours_table(pop_size: usize, k: usize) -> Vec<Vec<usize>> {
    (0..pop_size)
        .map(|i| {
            let mut ring = Vec::with_capacity(2 * k);
            for offset in 1..=k {
                ring.push((i + pop_size - offset) % pop_size);
                ring.push((i + offset) % pop_size);
            }
            ring
        })
        .collect()
}

fn compute_gen_stats(generation: usize, evals: usize, fitnesses: &[f64]) -> GenerationStats {
    let finite: Vec<f64> = fitnesses.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return GenerationStats {
            generation,
            evals,
            min: f64::INFINITY,
            avg: f64::INFINITY,
            std: 0.0,
            max: f64::INFINITY,
        };
    }
    let count = finite.len() as f64;
    let avg = finite.iter().sum::<f64>() / count;
    let var = finite.iter().map(|&v| (v - avg).powi(2)).sum::<f64>() / count;
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    GenerationStats {
        generation,
        evals,
        min,
        avg,
        std: var.sqrt(),
        max,
    }
}

fn evaluate_batch_parallel<F>(
    candidates: &[Vec<f64>],
    pool: Option<&rayon::ThreadPool>,
    eval_fn: &F,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    match pool {
        Some(p) => p.install(|| candidates.par_iter().map(|c| eval_fn(c)).collect()),
        None => candidates.par_iter().map(|c| eval_fn(c)).collect(),
    }
}

/// Runs deterministic Differential Evolution over `[-1.0, 1.0]^dim` to minimise `eval_fn`.
pub fn differential_evolution<F>(
    dim: usize,
    descriptors: &[ParamDescriptor],
    config: &DeConfig,
    eval_fn: F,
) -> Result<DeSolverResult, OptError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    config.validate()?;
    if dim == 0 || descriptors.len() != dim {
        return Err(OptError::InvalidSpec(
            "descriptor count must match optimisation dimension > 0".to_string(),
        ));
    }

    let pool = match config.threads {
        Some(n) if n > 0 => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| OptError::ThreadPool(e.to_string()))?,
        ),
        _ => None,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let lognormal = LogNormal::new(-1.2, 0.5).expect("valid fixed lognormal parameters");
    let all_indices: Vec<usize> = (0..config.pop_size).collect();
    let neighbour_table = config
        .neighbours
        .map(|k| build_neighbours_table(config.pop_size, k));

    let mut population: Vec<Vec<f64>> = (0..config.pop_size)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..=1.0)).collect())
        .collect();
    let mut fitnesses = evaluate_batch_parallel(&population, pool.as_ref(), &eval_fn);
    let mut total_evals = config.pop_size;

    let mut history = Vec::new();
    let mut record_evals = |generation: usize, inds: &[Vec<f64>], fits: &[f64], start_id: usize| {
        if !config.store_history {
            return;
        }
        for (offset, (norm_v, &fit)) in inds.iter().zip(fits.iter()).enumerate() {
            let decoded_params = descriptors
                .iter()
                .zip(norm_v.iter())
                .map(|(d, &x)| d.decode(x))
                .collect();
            history.push(EvalRecord {
                generation,
                eval_id: start_id + offset,
                energy: fit,
                norm_params: norm_v.clone(),
                decoded_params,
            });
        }
    };

    record_evals(0, &population, &fitnesses, 0);

    let (best_idx, mut best_energy) = fitnesses
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, &f)| (i, f))
        .unwrap();
    let mut best_norm = population[best_idx].clone();

    let mut logbook = Vec::with_capacity(config.generations + 1);
    logbook.push(compute_gen_stats(0, config.pop_size, &fitnesses));

    for generation in 1..=config.generations {
        let mut trials = Vec::with_capacity(config.pop_size);
        for i in 0..config.pop_size {
            let sample_pool = neighbour_table
                .as_ref()
                .map_or(all_indices.as_slice(), |t| t[i].as_slice());
            let (ia, ib, ic) = sample_three_distinct(&mut rng, sample_pool);
            let a = &population[ia];
            let b = &population[ib];
            let c = &population[ic];

            let mut trial = a.clone();
            let forced_dim = rng.random_range(0..dim);
            for d in 0..dim {
                if d == forced_dim || rng.random::<f64>() < config.cxpb {
                    let mut entry = f64::NAN;
                    let mut tries = 0usize;
                    while !entry.is_finite() || entry.abs() > 1.0 {
                        if tries > 10_000 {
                            entry = a[d];
                            break;
                        }
                        let step = rng.sample(lognormal) * config.diff_weight;
                        entry = a[d] + step * (b[d] - c[d]);
                        tries += 1;
                    }
                    trial[d] = entry;
                }
            }
            trials.push(trial);
        }

        let trial_fits = evaluate_batch_parallel(&trials, pool.as_ref(), &eval_fn);
        record_evals(generation, &trials, &trial_fits, total_evals);
        total_evals += config.pop_size;

        for i in 0..config.pop_size {
            if trial_fits[i] < fitnesses[i] {
                fitnesses[i] = trial_fits[i];
                population[i] = std::mem::take(&mut trials[i]);
                if trial_fits[i] < best_energy {
                    best_energy = trial_fits[i];
                    best_norm = population[i].clone();
                }
            }
        }
        logbook.push(compute_gen_stats(generation, config.pop_size, &fitnesses));
    }

    Ok(DeSolverResult {
        best_norm,
        best_energy,
        total_evals,
        logbook,
        history,
    })
}

/// Named optimised parameter value for reporting and PDB `REMARK` headers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedParamValue {
    pub label: String,
    pub value: f64,
    pub normalized: f64,
}

/// Diagnostic summary checking whether Differential Evolution had converged by termination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceStats {
    pub tail_drop_per_chain: f64,
    pub last_gen_drop_per_chain: f64,
    pub final_pop_std_per_chain: f64,
    pub still_minimizing: bool,
    pub boundary_hits: usize,
}

/// Complete result of `optimize_bundle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeResult {
    pub best_spec: BundleSpec,
    pub best_parameters: Vec<NamedParamValue>,
    pub best_energy: BundleEnergyReport,
    pub best_pdb: String,
    pub total_evals: usize,
    pub logbook: Vec<GenerationStats>,
    pub history: Vec<EvalRecord>,
    pub boundary_warnings: Vec<String>,
}

impl OptimizeResult {
    /// Checks whether the optimization trajectory was still noticeably decreasing at termination.
    pub fn convergence_diagnostics(
        &self,
        num_chains: usize,
        tail_gens: usize,
        tail_tol_per_chain: f64,
        last_tol_per_chain: f64,
    ) -> ConvergenceStats {
        let m = num_chains.max(1) as f64;
        let len = self.logbook.len();
        if len < 2 {
            return ConvergenceStats {
                tail_drop_per_chain: 0.0,
                last_gen_drop_per_chain: 0.0,
                final_pop_std_per_chain: self.logbook.last().map_or(0.0, |g| g.std / m),
                still_minimizing: false,
                boundary_hits: self.boundary_warnings.len(),
            };
        }
        let last = &self.logbook[len - 1];
        let prev = &self.logbook[len - 2];
        let tail_idx = len.saturating_sub(tail_gens.max(1) + 1);
        let tail_start = &self.logbook[tail_idx];

        let tail_drop_per_chain = (tail_start.min - last.min).max(0.0) / m;
        let last_gen_drop_per_chain = (prev.min - last.min).max(0.0) / m;
        let final_pop_std_per_chain = last.std / m;
        let still_minimizing =
            tail_drop_per_chain > tail_tol_per_chain || last_gen_drop_per_chain > last_tol_per_chain;

        ConvergenceStats {
            tail_drop_per_chain,
            last_gen_drop_per_chain,
            final_pop_std_per_chain,
            still_minimizing,
            boundary_hits: self.boundary_warnings.len(),
        }
    }

    /// Formats a concise generation logbook, final parameter summary, and boundary warnings.
    pub fn format_log(&self) -> String {
        let mut out = String::new();
        out.push_str("gen\tevals\tmin\t\tavg\t\tstd\t\tmax\n");
        for row in &self.logbook {
            out.push_str(&format!(
                "{}\t{}\t{:.3}\t\t{:.3}\t\t{:.3}\t\t{:.3}\n",
                row.generation, row.evals, row.min, row.avg, row.std, row.max
            ));
        }
        out.push_str(&format!("Evaluated {} models in total\n", self.total_evals));
        out.push_str(&format!("Best {}\n", self.best_energy));
        out.push_str("Best parameters:\n");
        for p in &self.best_parameters {
            out.push_str(&format!("  {} = {:.4} (norm={:+.3})\n", p.label, p.value, p.normalized));
        }
        for warn in &self.boundary_warnings {
            out.push_str(&format!("Warning: {warn}\n"));
        }
        out
    }

    /// Formats the full evaluation history as CSV for energy-funnel analysis.
    pub fn history_to_csv(&self) -> String {
        let mut out = String::from("gen,eval_id,energy");
        for p in &self.best_parameters {
            out.push(',');
            out.push_str(&p.label);
        }
        out.push('\n');
        for rec in &self.history {
            out.push_str(&format!("{},{},{:.6}", rec.generation, rec.eval_id, rec.energy));
            for v in &rec.decoded_params {
                out.push_str(&format!(",{v:.6}"));
            }
            out.push('\n');
        }
        out
    }
}

fn prepend_pdb_remarks(
    spec: &OptimizeSpec,
    params: &[NamedParamValue],
    energy: &BundleEnergyReport,
    packed_pdb: &str,
) -> String {
    let mut header = String::new();
    header.push_str("REMARK Optimized by betahelixbuilder Differential Evolution\n");
    header.push_str(&format!(
        "REMARK symmetry={:?} objective={:?} seed={} pop_size={} generations={}\n",
        spec.symmetry, spec.objective, spec.de.seed, spec.de.pop_size, spec.de.generations
    ));
    header.push_str(&format!("REMARK internal_energy {}\n", energy.internal));
    header.push_str(&format!("REMARK interaction_energy {}\n", energy.interaction));
    for p in params {
        header.push_str(&format!("REMARK param {} = {:.4}\n", p.label, p.value));
    }
    for (idx, h) in energy_bundle_summary_lines(&params_spec_or_helices(spec, params)) {
        header.push_str(&format!("REMARK helix_{idx} {h}\n"));
    }
    header.push_str(packed_pdb);
    header
}

fn params_spec_or_helices(spec: &OptimizeSpec, params: &[NamedParamValue]) -> BundleSpec {
    let norm: Vec<f64> = params.iter().map(|p| p.normalized).collect();
    spec.decode_bundle_spec(&norm)
        .expect("decoded parameters come from valid vector")
}

fn energy_bundle_summary_lines(bundle_spec: &BundleSpec) -> Vec<(usize, String)> {
    bundle_spec
        .helices
        .iter()
        .enumerate()
        .map(|(i, h)| {
            (
                i,
                format!(
                    "seq={} R={:.3} pitch={:?} phi={:.3} z_off={:.3} rpt={:.3} orient={:?} twist={:?}",
                    h.sequence,
                    h.bundle_radius,
                    h.pitch,
                    h.phi_c_alpha_deg,
                    h.z_offset,
                    h.minor_params.residues_per_turn,
                    h.orientation,
                    h.twist_mode
                ),
            )
        })
        .collect()
}

fn evaluate_candidate(
    spec: &OptimizeSpec,
    norm_vec: &[f64],
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<(BundleSpec, BundleEnergyReport, String), OptError> {
    let bundle_spec = spec.decode_bundle_spec(norm_vec)?;
    let bundle = build_bundle(&bundle_spec).map_err(|e| OptError::InvalidSpec(e.to_string()))?;
    let target_seqs =
        bundle_target_sequences(&bundle_spec).map_err(|e| OptError::InvalidSpec(e.to_string()))?;
    let packed_pdb = pack_bundle(&bundle, &target_seqs, daspr_cfg)?;
    let structure =
        ScoredStructure::from_pdb_str(&packed_pdb).map_err(|e| OptError::InvalidSpec(e.to_string()))?;
    let report = evaluate_bundle(&structure, ff, energy_opts);
    Ok((bundle_spec, report, packed_pdb))
}

/// Runs Differential Evolution bundle optimisation with implicit `dASPR` side-chain packing.
pub fn optimize_bundle(
    spec: &OptimizeSpec,
    ff: &BudeForceField,
    energy_opts: &EnergyOptions,
    daspr_cfg: &DasprConfig,
) -> Result<OptimizeResult, OptError> {
    resolve_daspr_binary(daspr_cfg)?;
    let descriptors = spec.validate_and_descriptors()?;

    let de_result = differential_evolution(descriptors.len(), &descriptors, &spec.de, |norm_vec| {
        match evaluate_candidate(spec, norm_vec, ff, energy_opts, daspr_cfg) {
            Ok((_bs, report, _pdb)) => match spec.objective {
                OptObjective::Internal => report.internal.total,
                OptObjective::Interaction => report.interaction.total,
            },
            Err(_) => f64::INFINITY,
        }
    })?;

    if !de_result.best_energy.is_finite() {
        return Err(OptError::NoValidCandidate);
    }

    let (best_spec, best_energy, packed_pdb) =
        evaluate_candidate(spec, &de_result.best_norm, ff, energy_opts, daspr_cfg)?;

    let best_parameters: Vec<NamedParamValue> = descriptors
        .iter()
        .zip(de_result.best_norm.iter())
        .map(|(d, &x)| NamedParamValue {
            label: d.label.clone(),
            value: d.decode(x),
            normalized: x,
        })
        .collect();

    let mut boundary_warnings = Vec::new();
    for p in &best_parameters {
        if p.normalized > 0.95 {
            boundary_warnings.push(format!(
                "Parameter '{}' ({:.3}) is at or near maximum allowed value",
                p.label, p.value
            ));
        } else if p.normalized < -0.95 {
            boundary_warnings.push(format!(
                "Parameter '{}' ({:.3}) is at or near minimum allowed value",
                p.label, p.value
            ));
        }
    }

    let best_pdb = prepend_pdb_remarks(spec, &best_parameters, &best_energy, &packed_pdb);

    Ok(OptimizeResult {
        best_spec,
        best_parameters,
        best_energy,
        best_pdb,
        total_evals: de_result.total_evals,
        logbook: de_result.logbook,
        history: de_result.history,
        boundary_warnings,
    })
}

/// Errors produced during optimisation setup or execution.
#[derive(Debug)]
pub enum OptError {
    Input(InputError),
    Sidechain(SidechainError),
    InvalidSpec(String),
    ThreadPool(String),
    NoValidCandidate,
}

impl From<InputError> for OptError {
    fn from(err: InputError) -> Self {
        Self::Input(err)
    }
}

impl From<SidechainError> for OptError {
    fn from(err: SidechainError) -> Self {
        Self::Sidechain(err)
    }
}

impl fmt::Display for OptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(err) => write!(f, "{err}"),
            Self::Sidechain(err) => write!(f, "{err}"),
            Self::InvalidSpec(msg) => write!(f, "invalid optimisation specification: {msg}"),
            Self::ThreadPool(msg) => write!(f, "failed to create rayon thread pool: {msg}"),
            Self::NoValidCandidate => {
                write!(f, "no valid finite-energy bundle candidate found during optimisation")
            }
        }
    }
}

impl std::error::Error for OptError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn differential_evolution_converges_deterministically_on_analytical_bowl() {
        let descriptors = vec![
            ParamDescriptor {
                label: "x0".to_string(),
                mean: 10.0,
                half_range: 4.0,
            },
            ParamDescriptor {
                label: "x1".to_string(),
                mean: 140.0,
                half_range: 60.0,
            },
            ParamDescriptor {
                label: "x2".to_string(),
                mean: 0.0,
                half_range: 180.0,
            },
        ];
        let cfg = DeConfig {
            pop_size: 20,
            generations: 35,
            cxpb: 0.75,
            diff_weight: 1.0,
            neighbours: Some(3),
            seed: 42,
            threads: Some(2),
            store_history: true,
        };

        let eval = |norm: &[f64]| -> f64 {
            let x0 = descriptors[0].decode(norm[0]);
            let x1 = descriptors[1].decode(norm[1]);
            let x2 = descriptors[2].decode(norm[2]);
            (x0 - 11.2).powi(2) + ((x1 - 125.0) / 20.0).powi(2) + ((x2 - 45.0) / 45.0).powi(2)
        };

        let r1 = differential_evolution(3, &descriptors, &cfg, eval).unwrap();
        let r2 = differential_evolution(3, &descriptors, &cfg, eval).unwrap();

        assert_eq!(r1.best_norm, r2.best_norm);
        assert_eq!(r1.logbook, r2.logbook);
        assert!(r1.best_energy < 1e-3, "failed to converge: {}", r1.best_energy);
        assert!((descriptors[0].decode(r1.best_norm[0]) - 11.2).abs() < 0.05);
        assert!((descriptors[1].decode(r1.best_norm[1]) - 125.0).abs() < 1.0);
        assert!((descriptors[2].decode(r1.best_norm[2]) - 45.0).abs() < 2.0);
    }

    #[test]
    fn yaml_parses_static_and_range_parameters() {
        let yaml = r#"
symmetry: d2
objective: internal
de:
  pop_size: 16
  generations: 10
  seed: 7
helices:
  - sequence: "VGAlAvVvWlWlWlW"
    bundle_radius: { mean: 11.0, range: 2.0 }
    pitch: [140.0, 40.0]
    phi_c_alpha_deg: 0.0
    z_offset: 0.0
"#;
        let spec: OptimizeSpec = serde_yml::from_str(yaml).unwrap();
        let desc = spec.validate_and_descriptors().unwrap();
        assert_eq!(desc.len(), 2);
        assert_eq!(desc[0].label, "bundle_radius");
        assert_eq!(desc[1].label, "pitch");
        let decoded = spec.decode_bundle_spec(&[0.5, -0.5]).unwrap();
        assert!((decoded.helices[0].bundle_radius - 12.0).abs() < 1e-12);
        assert!((decoded.helices[0].pitch.unwrap() - 120.0).abs() < 1e-12);
    }

    #[test]
    fn optimize_bundle_rejects_missing_daspr_upfront() {
        let spec = OptimizeSpec::default();
        let ff = BudeForceField::bude_2016v1().unwrap();
        let daspr = DasprConfig {
            daspr_bin: Some(PathBuf::from("/nonexistent/dASPR")),
            rotlib_path: None,
        };
        let err = optimize_bundle(&spec, &ff, &EnergyOptions::standard(), &daspr).unwrap_err();
        assert!(matches!(err, OptError::Sidechain(SidechainError::DasprNotFound(_))));
    }

    #[test]
    fn convergence_diagnostics_flags_dropping_tail_vs_flat_tail() {
        let make_res = |mins: &[f64]| OptimizeResult {
            best_spec: BundleSpec::default(),
            best_parameters: Vec::new(),
            best_energy: BundleEnergyReport {
                internal: crate::energy::EnergyScore::default(),
                interaction: crate::energy::EnergyScore::default(),
            },
            best_pdb: String::new(),
            total_evals: 0,
            logbook: mins
                .iter()
                .enumerate()
                .map(|(g, &m)| GenerationStats {
                    generation: g,
                    evals: 16,
                    min: m,
                    avg: m + 5.0,
                    std: 2.0,
                    max: m + 10.0,
                })
                .collect(),
            history: Vec::new(),
            boundary_warnings: Vec::new(),
        };

        let dropping = make_res(&[-100.0, -104.0, -108.0, -112.0]);
        let st_drop = dropping.convergence_diagnostics(2, 3, 0.25, 0.05);
        assert!(st_drop.still_minimizing);
        assert!((st_drop.tail_drop_per_chain - 6.0).abs() < 1e-12);

        let plateaued = make_res(&[-100.0, -112.0, -112.01, -112.02]);
        let st_flat = plateaued.convergence_diagnostics(2, 2, 0.25, 0.05);
        assert!(!st_flat.still_minimizing);
    }
}
