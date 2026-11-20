//! CLI entrypoint for generating and scoring monomeric and bundled beta-helix PDB structures.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command as ProcessCommand};

use betahelixbuilder::{
    AnchorParity, BudeForceField, BundleEnergyReport, BundleSpec, CCap, CanonicalResidue,
    Chirality, DasprConfig, DesignConfig, DesignConstraints, EdgePattern, EnergyOptions,
    EnergyScore, FrameMode, Handedness, HelixParams, HelixPlacementParams, HexOligomerTarget,
    HexamerWorkflowConfig, LumenMode, MonomerSpec, NCap, OligomerState, OligomerWorkflowConfig,
    OptObjective, OptimizeSpec, Orientation, PairSurrogateModel, ParamValue, ProbeConfig, RankBy,
    ScoredStructure, SeamComposition, SeamPattern, SeamWidth, Sequence, Symmetry, TwistMode,
    build_beta_helix, build_bundle, bundle_target_sequences, bundle_to_pdb_string,
    design_hexamers, design_oligomers, design_sequences, evaluate_bundle, interaction_energy,
    internal_energy, optimize_bundle, pack_bundle, pack_chain, probe_pair_interactions,
    to_pdb_string,
};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Preset {
    Beta63,
    Gramicidin,
    Polytheonamide,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliHandedness {
    Right,
    Left,
}

impl From<CliHandedness> for Handedness {
    fn from(h: CliHandedness) -> Self {
        match h {
            CliHandedness::Right => Self::Right,
            CliHandedness::Left => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliTwistMode {
    CoupledDarboux,
    UncoupledLabFrame,
}

impl From<CliTwistMode> for TwistMode {
    fn from(m: CliTwistMode) -> Self {
        match m {
            CliTwistMode::CoupledDarboux => Self::CoupledDarboux,
            CliTwistMode::UncoupledLabFrame => Self::UncoupledLabFrame,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOrientation {
    Parallel,
    Antiparallel,
}

impl From<CliOrientation> for Orientation {
    fn from(o: CliOrientation) -> Self {
        match o {
            CliOrientation::Parallel => Self::Parallel,
            CliOrientation::Antiparallel => Self::Antiparallel,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScoreMode {
    Internal,
    Interaction,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOptObjective {
    Internal,
    Interaction,
}

impl From<CliOptObjective> for OptObjective {
    fn from(o: CliOptObjective) -> Self {
        match o {
            CliOptObjective::Internal => Self::Internal,
            CliOptObjective::Interaction => Self::Interaction,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAnchorParity {
    L,
    D,
    Both,
}

impl From<CliAnchorParity> for AnchorParity {
    fn from(a: CliAnchorParity) -> Self {
        match a {
            CliAnchorParity::L => Self::L,
            CliAnchorParity::D => Self::D,
            CliAnchorParity::Both => Self::Both,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliRankBy {
    Dmc,
    VsAla,
    DirectPair,
}

impl From<CliRankBy> for RankBy {
    fn from(r: CliRankBy) -> Self {
        match r {
            CliRankBy::Dmc => Self::Dmc,
            CliRankBy::VsAla => Self::VsAla,
            CliRankBy::DirectPair => Self::DirectPair,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliFrameMode {
    Alternating,
    Strict,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSeamPattern {
    Strict6,
    Staggered6676,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSeamWidth {
    #[value(name = "1")]
    One,
    #[value(name = "2")]
    Two,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSeamComp {
    Aliphatic,
    Mixed,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliEdgePattern {
    Parallel,
    Alternating,
    Neutral,
    All,
}

#[derive(Debug, Parser)]
#[command(name = "betahelixbuilder")]
#[command(about = "Parametric Crick-style beta-helix monomer, bundle, and BUDE energy tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a single monomeric beta-helix PDB structure.
    Monomer(MonomerArgs),
    /// Generate a symmetric (Cn/Dn) or heteromeric supercoiled beta-helix bundle PDB structure.
    Bundle(BundleArgs),
    /// Score an existing PDB file with the native BUDE force field.
    Score(ScoreArgs),
    /// Optimise bundle parameters via Differential Evolution with implicit dASPR packing.
    Optimize(Box<OptimizeArgs>),
    /// Systematically probe pairwise L/D side-chain couplings across beta-helix offsets.
    ProbePairs(ProbePairsArgs),
    /// Design stable, handedness-specific repetitive beta-helix sequences (~30 residues).
    DesignSeq(Box<DesignSeqArgs>),
    /// Design water-soluble seam-bearing beta-helices and assess oligomerization via multi-state DE.
    DesignOligomers(Box<DesignOligomersArgs>),
    /// Design two-interface (AB >> AA/BB) asymmetric parallel hexamers (C6) and scale across pools.
    DesignHexamers(Box<DesignHexamersArgs>),
}

#[derive(Debug, Args, Default)]
struct EnergyFlags {
    /// Evaluate BUDE force-field energy.
    #[arg(long)]
    energy: bool,

    /// Include full pairwise interaction breakdown in energy output.
    #[arg(long)]
    energy_details: bool,

    /// Emit full energy report as JSON.
    #[arg(long)]
    energy_json: bool,

    /// Override BUDE force-field JSON path (defaults to BUDEFF_FF_PATH or bundled bude_2016v1.json).
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior for exact comparison.
    #[arg(long)]
    legacy_electrostatics: bool,
}

impl EnergyFlags {
    fn enabled(&self) -> bool {
        self.energy || self.energy_details || self.energy_json
    }

    fn load_ff(&self) -> Result<BudeForceField, Box<dyn std::error::Error>> {
        match &self.force_field {
            Some(p) => Ok(BudeForceField::from_file(p)?),
            None => Ok(BudeForceField::load_default()?),
        }
    }

    fn options(&self) -> EnergyOptions {
        EnergyOptions {
            include_interactions: self.energy_details || self.energy_json,
            no_neighbour_backbone: true,
            legacy_unassigned_electrostatic_type: self.legacy_electrostatics,
        }
    }
}

#[derive(Debug, Args)]
struct MonomerArgs {
    /// Optional YAML monomer configuration file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Case-encoded sequence (uppercase = L, lowercase = D, uppercase G = Gly).
    #[arg(short, long)]
    sequence: Option<String>,

    /// Parameter preset.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Cylinder radius (Å) for L-parity Cα atoms.
    #[arg(long)]
    radius_l: Option<f64>,

    /// Cylinder radius (Å) for D-parity Cα atoms.
    #[arg(long)]
    radius_d: Option<f64>,

    /// Base residues per helical turn.
    #[arg(long)]
    res_per_turn: Option<f64>,

    /// Comma-separated per-step residues-per-turn profile (`len == n_res - 1`).
    #[arg(long, value_delimiter = ',')]
    res_per_turn_profile: Option<Vec<f64>>,

    /// Helical handedness.
    #[arg(long, value_enum)]
    handedness: Option<CliHandedness>,

    /// PDB chain identifier.
    #[arg(long)]
    chain: Option<char>,

    /// Keep glycine residues as GLY instead of converting to poly-D/L-Ala.
    #[arg(long)]
    keep_gly: bool,

    /// Pack side chains for the full target sequence using dASPR.
    #[arg(long)]
    pack: bool,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    #[command(flatten)]
    energy: EnergyFlags,

    /// Output PDB file path (defaults to stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct BundleArgs {
    /// Optional YAML bundle configuration file (supports Dn/Cn and heteromeric assemblies).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Bundle symmetry (`none`, `c2`, `d2`, `d3`, ...).
    #[arg(long)]
    symmetry: Option<String>,

    /// Case-encoded sequence for template helix (uppercase = L, lowercase = D, uppercase G = Gly).
    #[arg(short, long)]
    sequence: Option<String>,

    /// Bundle radius (Å) from bundle Z-axis to helix centerline.
    #[arg(long)]
    bundle_radius: Option<f64>,

    /// Signed supercoil pitch (Å): >0 right-handed, <0 left-handed, 0 = straight.
    #[arg(long, allow_hyphen_values = true)]
    pitch: Option<f64>,

    /// Helix rotation φ_Cα (degrees; 0° points inward toward bundle axis).
    #[arg(long, allow_hyphen_values = true)]
    phi_c_alpha: Option<f64>,

    /// Longitudinal offset (Å) relative to helix 0.
    #[arg(long, allow_hyphen_values = true)]
    z_offset: Option<f64>,

    /// Orientation for single/template helix.
    #[arg(long, value_enum)]
    orientation: Option<CliOrientation>,

    /// Supercoil twist coupling mode.
    #[arg(long, value_enum)]
    twist_mode: Option<CliTwistMode>,

    /// Minor helix residues per turn.
    #[arg(long)]
    res_per_turn: Option<f64>,

    /// Minor helix handedness.
    #[arg(long, value_enum)]
    handedness: Option<CliHandedness>,

    /// Keep glycine residues as GLY instead of converting to poly-D/L-Ala.
    #[arg(long)]
    keep_gly: bool,

    /// Pack side chains for the full target sequence(s) using dASPR.
    #[arg(long)]
    pack: bool,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    #[command(flatten)]
    energy: EnergyFlags,

    /// Output PDB file path (defaults to stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ScoreArgs {
    /// Input PDB file to evaluate.
    pdb: PathBuf,

    /// Scoring mode (defaults to internal for 1 chain, both for multi-chain).
    #[arg(long, value_enum)]
    mode: Option<ScoreMode>,

    #[command(flatten)]
    energy: EnergyFlags,
}

#[derive(Debug, Args)]
struct OptimizeArgs {
    /// Optional YAML optimisation configuration file (supports per-helix heteromeric ranges).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Bundle symmetry (`none`, `c2`, `d2`, `d3`, ...).
    #[arg(long)]
    symmetry: Option<String>,

    /// Case-encoded sequence for template helix (uppercase = L, lowercase = D, uppercase G = Gly).
    #[arg(short, long)]
    sequence: Option<String>,

    /// Mean bundle radius (Å).
    #[arg(long)]
    bundle_radius: Option<f64>,
    /// Half-range (±Å) for bundle radius search.
    #[arg(long)]
    bundle_radius_range: Option<f64>,

    /// Mean signed supercoil pitch (Å).
    #[arg(long, allow_hyphen_values = true)]
    pitch: Option<f64>,
    /// Half-range (±Å) for supercoil pitch search.
    #[arg(long)]
    pitch_range: Option<f64>,

    /// Mean helix rotation φ_Cα (degrees).
    #[arg(long, allow_hyphen_values = true)]
    phi_c_alpha: Option<f64>,
    /// Half-range (±°) for φ_Cα search.
    #[arg(long)]
    phi_c_alpha_range: Option<f64>,

    /// Mean longitudinal offset (Å) relative to helix 0.
    #[arg(long, allow_hyphen_values = true)]
    z_offset: Option<f64>,
    /// Half-range (±Å) for longitudinal offset search.
    #[arg(long)]
    z_offset_range: Option<f64>,

    /// Mean minor helix residues per turn.
    #[arg(long)]
    res_per_turn: Option<f64>,
    /// Half-range for minor residues per turn search.
    #[arg(long)]
    res_per_turn_range: Option<f64>,

    /// Supercoil twist coupling mode.
    #[arg(long, value_enum)]
    twist_mode: Option<CliTwistMode>,

    /// Keep glycine residues as GLY instead of converting to poly-D/L-Ala.
    #[arg(long)]
    keep_gly: bool,

    /// Energy objective to minimise (`internal` default or `interaction`).
    #[arg(long, value_enum)]
    objective: Option<CliOptObjective>,

    /// Population size per DE generation.
    #[arg(long)]
    pop_size: Option<usize>,

    /// Number of DE generations.
    #[arg(long)]
    generations: Option<usize>,

    /// DE crossover probability `[0, 1]`.
    #[arg(long)]
    cxpb: Option<f64>,

    /// DE differential weight (> 0).
    #[arg(long)]
    diff_weight: Option<f64>,

    /// Optional ring neighbourhood radius.
    #[arg(long)]
    neighbours: Option<usize>,

    /// Deterministic PRNG seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Parallel rayon worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    /// Override BUDE force-field JSON path.
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior.
    #[arg(long)]
    legacy_electrostatics: bool,

    /// Output PDB path for best packed model.
    #[arg(short, long, default_value = "optimized_best.pdb")]
    output: PathBuf,

    /// Optional file path to save the text optimisation log.
    #[arg(long)]
    log_file: Option<PathBuf>,

    /// Optional CSV file path for full evaluation funnel history.
    #[arg(long)]
    history_csv: Option<PathBuf>,

    /// Optional JSON file path for full evaluation history.
    #[arg(long)]
    history_json: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ProbePairsArgs {
    /// Optional YAML monomer configuration file supplying host `HelixParams`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Backbone parameter preset for the host beta-helix.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Host poly-(L/D-Ala) peptide length.
    #[arg(long, default_value_t = 22)]
    host_length: usize,

    /// Comma-separated relative partner offsets `j - i` to scan.
    #[arg(long, default_value = "-1,1,5,6,7", allow_hyphen_values = true)]
    offsets: String,

    /// Which anchor chirality position(s) to scan.
    #[arg(long, value_enum, default_value = "both")]
    anchor: CliAnchorParity,

    /// Exclude Proline (`P`/`p`) from the scanned residue alphabet.
    #[arg(long)]
    exclude_proline: bool,

    /// Optional comma-separated uppercase one-letter residue subset (e.g. `A,W,D,K,P`).
    #[arg(long)]
    residues: Option<String>,

    /// Primary ranking metric for sorting CSV and summary tables.
    #[arg(long, value_enum, default_value = "dmc")]
    rank_by: CliRankBy,

    /// Number of top/bottom pairs printed per `(anchor, offset)` summary table.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Threshold on `max |ddE_L - ddE_D|` for flagging L-vs-D asymmetry.
    #[arg(long, default_value_t = 0.5)]
    parity_max_diff: f64,

    /// Threshold on RMSD(ddE_L, ddE_D) for flagging L-vs-D asymmetry.
    #[arg(long, default_value_t = 0.15)]
    parity_rmsd: f64,

    /// Fail immediately instead of recording NaN if an external dASPR call errors on an isolated pair.
    #[arg(long)]
    strict: bool,

    /// Parallel rayon worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    /// Override BUDE force-field JSON path.
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior.
    #[arg(long)]
    legacy_electrostatics: bool,

    /// Output CSV file path containing all ranked pairings and decomposed metrics.
    #[arg(short, long, default_value = "pair_scan_ranked.csv")]
    output: PathBuf,

    /// Optional directory to write per-(anchor, offset) 20x20 matrix CSVs.
    #[arg(long)]
    matrix_dir: Option<PathBuf>,

    /// Optional directory to render heatmap & L-vs-D parity PNG plots via Python/Matplotlib.
    #[arg(long)]
    plot_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DesignSeqArgs {
    /// Precomputed `probe-pairs` CSV file to build Stage-1 surrogate without re-running `probe-pairs`.
    #[arg(long)]
    pair_csv: Option<PathBuf>,

    /// Handedness under which `--pair-csv` was recorded (default: right).
    #[arg(long, value_enum, default_value = "right")]
    pair_handedness: CliHandedness,

    /// Optional YAML monomer configuration supplying target `HelixParams`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Target backbone parameter preset.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Target helical handedness to stabilize relative to its mirror enantiomer decoy.
    #[arg(long, value_enum)]
    target_handedness: Option<CliHandedness>,

    /// Designed peptide length in residues (default 30 = 5 hexapeptide turns).
    #[arg(long, default_value_t = 30)]
    length: usize,

    /// Disable N-terminal acetyl cap (`Ac-`).
    #[arg(long)]
    no_n_cap: bool,

    /// Disable C-terminal primary amide cap (`-NH2`).
    #[arg(long)]
    no_c_cap: bool,

    /// Repeat template mode (`alternating` 12-mer super-repeat or `strict` 6-mer repeat).
    #[arg(long, value_enum, default_value = "alternating")]
    frame_mode: CliFrameMode,

    /// Maximum differing columns between alternating 6-mer halves (`A` and `B`).
    #[arg(long, default_value_t = 2)]
    max_alternating_cols: usize,

    /// Allow Cysteine (`C`/`c`) in designed sequences.
    #[arg(long)]
    allow_cys: bool,

    /// Allow Methionine (`M`/`m`) in designed sequences.
    #[arg(long)]
    allow_met: bool,

    /// Exclude Arginine (`R`/`r`), leaving Lysine as the positive charge option.
    #[arg(long)]
    exclude_arg: bool,

    /// Allow Proline (`P`/`p`) in the repeating core frame.
    #[arg(long)]
    allow_pro: bool,

    /// Allow Proline (`P`/`p`) in the first/last 4 terminal residues during `--refine`.
    #[arg(long)]
    allow_terminal_pro: bool,

    /// Minimum formal net charge (default +1).
    #[arg(long, default_value_t = 1)]
    min_charge: i32,

    /// Maximum formal net charge (default +4).
    #[arg(long, default_value_t = 4)]
    max_charge: i32,

    /// Minimum total Trp + Tyr residue count (default 1).
    #[arg(long, default_value_t = 1)]
    min_trp_tyr: usize,

    /// Optional maximum total aromatic (`Phe` + `Tyr` + `Trp`) count across the sequence.
    #[arg(long)]
    max_aromatic: Option<usize>,

    /// Optional maximum Histidine (`His`) count across the sequence.
    #[arg(long)]
    max_his: Option<usize>,

    /// Optional maximum bulky hydrophobic aliphatic (`Val` + `Leu` + `Ile`) count across the sequence.
    #[arg(long)]
    max_aliphatic: Option<usize>,

    /// Forbid continuous vertical `Val`/`Leu`/`Ile` ladders (`col_a[k]` and `col_b[k]` both in `{V,L,I}`).
    #[arg(long)]
    no_aliphatic_ladders: bool,

    /// Minimum count of soluble polar/charged residues (`S,T,N,Q,H,D,E,K,R`).
    #[arg(long, default_value_t = 0)]
    min_polar: usize,

    /// Handedness specificity weight (`score = E_target - weight * (E_decoy - E_target)`).
    #[arg(long, default_value_t = 0.5)]
    specificity_weight: f64,

    /// Number of Stage-1 Simulated Annealing restarts.
    #[arg(long, default_value_t = 32)]
    sa_restarts: usize,

    /// Steps per Stage-1 Simulated Annealing trajectory.
    #[arg(long, default_value_t = 6000)]
    sa_steps: usize,

    /// Number of Stage-1 Pareto-diverse candidates verified in full 3D (`dASPR` + `BUDEFF`).
    #[arg(long, default_value_t = 25)]
    top_k_3d: usize,

    /// Polish first/last 4 residues of top candidates directly in 3D (`dASPR` + `BUDEFF`).
    #[arg(long)]
    refine: bool,

    /// Number of top 3D candidates refined when `--refine` is active.
    #[arg(long, default_value_t = 4)]
    refine_top_n: usize,

    /// Number of top candidates displayed to stdout.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Deterministic Stage-1 PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Parallel rayon worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    /// Override BUDE force-field JSON path.
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior.
    #[arg(long)]
    legacy_electrostatics: bool,

    /// Ranked CSV output path.
    #[arg(short, long, default_value = "designed_sequences.csv")]
    output: PathBuf,

    /// Optional directory to write packed 3D target & mirror-decoy PDBs for top candidates.
    #[arg(long)]
    pdb_dir: Option<PathBuf>,

    /// Optional output PNG path for Stability vs Handedness Pareto plot.
    #[arg(long)]
    plot: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DesignOligomersArgs {
    /// Precomputed `probe-pairs` CSV file to build Stage-1 surrogate without re-running `probe-pairs`.
    #[arg(long)]
    pair_csv: Option<PathBuf>,

    /// Handedness under which `--pair-csv` was recorded (default: right).
    #[arg(long, value_enum, default_value = "right")]
    pair_handedness: CliHandedness,

    /// Optional YAML monomer configuration supplying target `HelixParams`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Target backbone parameter preset.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Designed peptide length (recommended 28-30; default 30).
    #[arg(long, default_value_t = 30)]
    length: usize,

    /// Disable N-terminal acetyl cap (`Ac-`).
    #[arg(long)]
    no_n_cap: bool,

    /// Disable C-terminal primary amide cap (`-NH2`).
    #[arg(long)]
    no_c_cap: bool,

    /// Hydrophobic seam repeating step pattern.
    #[arg(long, value_enum, default_value = "both")]
    seam_pattern: CliSeamPattern,

    /// Lateral seam width (1 residue/turn vs 2-residue L+D pair/turn).
    #[arg(long, value_enum, default_value = "both")]
    seam_width: CliSeamWidth,

    /// Seam residue composition (`aliphatic` = I/L/V, `mixed` = I/L/V + W/Y/F).
    #[arg(long, value_enum, default_value = "both")]
    seam_comp: CliSeamComp,

    /// Seam-flanking charge edge grammar (`parallel` default).
    #[arg(long, value_enum, default_value = "parallel")]
    edge_pattern: CliEdgePattern,

    /// Exclude Arginine (`R`/`r`), leaving Lysine as the positive charge option.
    #[arg(long)]
    exclude_arg: bool,

    /// Skip adding terminal Trp on pure aliphatic seams.
    #[arg(long)]
    no_terminal_wy: bool,

    /// Minimum right-handed monomer gap `E_left - E_right` for gating (default 5.0).
    #[arg(long, default_value_t = 5.0)]
    min_hand_gap: f64,

    /// Stage-1 SA restarts per `(pattern, width, comp, edge)` class.
    #[arg(long, default_value_t = 6)]
    sa_restarts: usize,

    /// Stage-1 SA steps per restart.
    #[arg(long, default_value_t = 2500)]
    sa_steps: usize,

    /// Top diverse monomers carried into multi-state 3D oligomer optimization.
    #[arg(long, default_value_t = 6)]
    top_monomers: usize,

    /// Comma-separated oligomer symmetry states to optimize (supports `c2,d1,c3,d2,c4,c6,d3`).
    #[arg(long, default_value = "c2,d1,c3,d2,c4")]
    states: String,

    /// Skip the coarse 2D `(bundle_radius, pitch)` pre-screen before DE.
    #[arg(long)]
    skip_coarse: bool,

    /// DE population size per oligomer state optimization.
    #[arg(long, default_value_t = 14)]
    de_pop: usize,

    /// DE generations per oligomer state optimization.
    #[arg(long, default_value_t = 12)]
    de_gens: usize,

    /// Per-chain tail energy drop threshold flagging `still_minimizing` (default 0.25).
    #[arg(long, default_value_t = 0.25)]
    tail_drop_tol: f64,

    /// Deterministic PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Parallel rayon worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    /// Override BUDE force-field JSON path.
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior.
    #[arg(long)]
    legacy_electrostatics: bool,

    /// Output directory for `monomer_screen.csv`, `oligomer_screen.csv`, PDBs, and optional plot.
    #[arg(short, long, default_value = "oligomer_workflow_out")]
    output_dir: PathBuf,

    /// Generate `oligomer_landscape.png` heatmap and selectivity plot in `--output-dir`.
    #[arg(long)]
    plot: bool,
}

#[derive(Debug, Args)]
struct DesignHexamersArgs {
    /// Precomputed `probe-pairs` CSV file to build Stage-1 surrogate without re-running `probe-pairs`.
    #[arg(long)]
    pair_csv: Option<PathBuf>,

    /// Handedness under which `--pair-csv` was recorded (default: right).
    #[arg(long, value_enum, default_value = "right")]
    pair_handedness: CliHandedness,

    /// Optional YAML monomer configuration supplying target `HelixParams`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Target backbone parameter preset.
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Designed peptide length (recommended 28-30; default 30).
    #[arg(long, default_value_t = 30)]
    length: usize,

    /// Target parallel cyclic oligomer stoichiometry (default `6` for C6).
    #[arg(long, default_value_t = 6)]
    target_n: usize,

    /// Disable N-terminal acetyl cap (`Ac-`).
    #[arg(long)]
    no_n_cap: bool,

    /// Disable C-terminal primary amide cap (`-NH2`).
    #[arg(long)]
    no_c_cap: bool,

    /// Seam repeating step schedule (`strict6`, `staggered6676`, or `both`).
    #[arg(long, value_enum, default_value = "both")]
    seam_pattern: CliSeamPattern,

    /// Comma-separated central lumen chemistries (`aromatic`, `salt-bridge`, `polar`).
    #[arg(long, default_value = "aromatic,salt-bridge")]
    lumen_mode: String,

    /// Exclude Arginine (`R`/`r`), leaving Lysine as positive option.
    #[arg(long)]
    exclude_arg: bool,

    /// Minimum right-handed monomer gap `E_left - E_right` (default 5.0).
    #[arg(long, default_value_t = 5.0)]
    min_hand_gap: f64,

    /// Stage-1 SA restarts per `(seam_pattern, lumen_mode)` class.
    #[arg(long, default_value_t = 10)]
    sa_restarts: usize,

    /// Stage-1 SA steps per restart.
    #[arg(long, default_value_t = 2500)]
    sa_steps: usize,

    /// Monomer-gated sequences pushed through flat-parallel Stage 3A coarse multi-state screen.
    #[arg(long, default_value_t = 20)]
    coarse_pool: usize,

    /// Top coarse-screen winners promoted to full Stage 3B DE optimization.
    #[arg(long, default_value_t = 8)]
    top_de: usize,

    /// Comma-separated targets (supports `c2_aa,c2_bb,c2,c3,c4,c5,c6,c7,d2,d3`).
    #[arg(long, default_value = "c2_aa,c2_bb,c2,c3,c4,c5,c6,c7,d2,d3")]
    states: String,

    /// DE population size per promoted `(candidate, state)` refinement.
    #[arg(long, default_value_t = 16)]
    de_pop: usize,

    /// DE generations per promoted `(candidate, state)` refinement.
    #[arg(long, default_value_t = 12)]
    de_gens: usize,

    /// Per-chain tail energy drop threshold flagging `still_minimizing` (default 0.25).
    #[arg(long, default_value_t = 0.25)]
    tail_drop_tol: f64,

    /// Deterministic PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Parallel rayon worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Path to dASPR binary (defaults to DASPR_BIN, PATH, or ~/dASPR/dASPR).
    #[arg(long)]
    daspr_bin: Option<PathBuf>,

    /// Path to dASPR rotamer library (`dun2010_mirror.bin`).
    #[arg(long)]
    rotlib: Option<PathBuf>,

    /// Override BUDE force-field JSON path.
    #[arg(long)]
    force_field: Option<PathBuf>,

    /// Mirror budeff C++ unassigned electrostaticType behavior.
    #[arg(long)]
    legacy_electrostatics: bool,

    /// Output directory for `monomer_screen.csv`, `coarse_screen.csv`, `hexamer_screen.csv`, PDBs, and optional plot.
    #[arg(short, long, default_value = "hexamer_workflow_out")]
    output_dir: PathBuf,

    /// Generate `hexamer_landscape.png` heatmap and pore/selectivity plot in `--output-dir`.
    #[arg(long)]
    plot: bool,
}

fn default_sequence(preset: Preset) -> &'static str {
    match preset {
        Preset::Beta63 | Preset::Gramicidin => "AaAaAaAaAaAaAaA",
        Preset::Polytheonamide => "AAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaA",
    }
}

fn open_output(path: Option<PathBuf>) -> io::Result<Box<dyn Write>> {
    match path {
        Some(p) => Ok(Box::new(BufWriter::new(File::create(p)?))),
        None => Ok(Box::new(BufWriter::new(io::stdout().lock()))),
    }
}

fn write_score_text(
    writer: &mut impl Write,
    label: &str,
    score: &EnergyScore,
    show_details: bool,
) -> io::Result<()> {
    writeln!(writer, "{label}: {score}")?;
    if !score.unparameterized_atoms.is_empty() {
        writeln!(
            writer,
            "  unparameterized_atoms ({}): {}",
            score.unparameterized_atoms.len(),
            score.unparameterized_atoms.join(", ")
        )?;
    }
    if show_details {
        for inter in &score.interactions {
            writeln!(
                writer,
                "  {}:{} {} {} <-> {}:{} {} {} d={:.3} st={:.3} de={:.3} ch={:.3}",
                inter.chain_a,
                inter.seq_id_a,
                inter.res_name_a,
                inter.atom_name_a,
                inter.chain_b,
                inter.seq_id_b,
                inter.res_name_b,
                inter.atom_name_b,
                inter.distance,
                inter.steric,
                inter.desolvation,
                inter.charge
            )?;
        }
    }
    Ok(())
}

fn emit_monomer_energy(
    writer: &mut impl Write,
    flags: &EnergyFlags,
    score: &EnergyScore,
) -> Result<(), Box<dyn std::error::Error>> {
    if flags.energy_json {
        writeln!(writer, "{}", serde_json::to_string_pretty(score)?)?;
    } else {
        write_score_text(writer, "internal", score, flags.energy_details)?;
    }
    writer.flush()?;
    Ok(())
}

fn emit_bundle_energy(
    writer: &mut impl Write,
    flags: &EnergyFlags,
    report: &BundleEnergyReport,
) -> Result<(), Box<dyn std::error::Error>> {
    if flags.energy_json {
        writeln!(writer, "{}", serde_json::to_string_pretty(report)?)?;
    } else {
        write_score_text(writer, "internal", &report.internal, flags.energy_details)?;
        write_score_text(
            writer,
            "interaction",
            &report.interaction,
            flags.energy_details,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn run_monomer(args: MonomerArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut spec = match args.config {
        Some(path) => serde_yml::from_str::<MonomerSpec>(&fs::read_to_string(path)?)?,
        None => MonomerSpec::default(),
    };

    if let Some(preset) = args.preset {
        spec.params = match preset {
            Preset::Beta63 => HelixParams::beta_6_3(),
            Preset::Gramicidin => HelixParams::beta_6_3_gramicidin(),
            Preset::Polytheonamide => HelixParams::beta_6_3_polytheonamide(),
        };
        if args.sequence.is_none() {
            spec.sequence = default_sequence(preset).to_string();
        }
    }
    if let Some(s) = args.sequence {
        spec.sequence = s;
    }
    if let Some(r) = args.radius_l {
        spec.params.radius_l = r;
    }
    if let Some(r) = args.radius_d {
        spec.params.radius_d = r;
    }
    if let Some(n) = args.res_per_turn {
        spec.params.residues_per_turn = n;
    }
    if let Some(prof) = args.res_per_turn_profile {
        spec.params.residues_per_turn_profile = Some(prof);
    }
    if let Some(h) = args.handedness {
        spec.params.handedness = h.into();
    }
    if let Some(ch) = args.chain {
        spec.chain_id = ch;
    }
    if args.keep_gly {
        spec.keep_gly = true;
    }
    if args.pack {
        spec.pack = true;
    }

    let target_seq = Sequence::parse(&spec.sequence)?;
    let backbone_seq = target_seq.as_poly_ala(spec.keep_gly || spec.pack);
    let chain = build_beta_helix(&backbone_seq, &spec.params, spec.chain_id)?;
    let pdb_text = if spec.pack {
        let daspr_cfg = DasprConfig {
            daspr_bin: args.daspr_bin,
            rotlib_path: args.rotlib,
        };
        pack_chain(&chain, &target_seq, &daspr_cfg)?
    } else {
        to_pdb_string(&chain)
    };

    let mut writer = open_output(args.output)?;
    writer.write_all(pdb_text.as_bytes())?;
    writer.flush()?;

    if args.energy.enabled() {
        let ff = args.energy.load_ff()?;
        let structure = if spec.pack {
            ScoredStructure::from_pdb_str(&pdb_text)?
        } else {
            ScoredStructure::from_chain(&chain)
        };
        let score = internal_energy(&structure, &ff, &args.energy.options());
        let mut stderr = io::stderr().lock();
        emit_monomer_energy(&mut stderr, &args.energy, &score)?;
    }

    Ok(())
}

fn run_bundle(args: BundleArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut spec = match args.config {
        Some(path) => serde_yml::from_str::<BundleSpec>(&fs::read_to_string(path)?)?,
        None => BundleSpec {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams::default()],
        },
    };

    if let Some(sym) = args.symmetry {
        spec.symmetry = sym.parse()?;
    }
    if args.keep_gly {
        spec.keep_gly = true;
    }
    if args.pack {
        spec.pack = true;
    }
    if let Some(tpl) = spec.helices.first_mut() {
        if let Some(s) = args.sequence {
            tpl.sequence = s;
        }
        if let Some(r) = args.bundle_radius {
            tpl.bundle_radius = r;
        }
        if let Some(p) = args.pitch {
            tpl.pitch = Some(p);
        }
        if let Some(phi) = args.phi_c_alpha {
            tpl.phi_c_alpha_deg = phi;
        }
        if let Some(z) = args.z_offset {
            tpl.z_offset = z;
        }
        if let Some(o) = args.orientation {
            tpl.orientation = o.into();
        }
        if let Some(tm) = args.twist_mode {
            tpl.twist_mode = tm.into();
        }
        if let Some(n) = args.res_per_turn {
            tpl.minor_params.residues_per_turn = n;
        }
        if let Some(h) = args.handedness {
            tpl.minor_params.handedness = h.into();
        }
    }

    let bundle = build_bundle(&spec)?;
    let pdb_text = if spec.pack {
        let daspr_cfg = DasprConfig {
            daspr_bin: args.daspr_bin,
            rotlib_path: args.rotlib,
        };
        let target_seqs = bundle_target_sequences(&spec)?;
        pack_bundle(&bundle, &target_seqs, &daspr_cfg)?
    } else {
        bundle_to_pdb_string(&bundle)
    };

    let mut writer = open_output(args.output)?;
    writer.write_all(pdb_text.as_bytes())?;
    writer.flush()?;

    if args.energy.enabled() {
        let ff = args.energy.load_ff()?;
        let structure = if spec.pack {
            ScoredStructure::from_pdb_str(&pdb_text)?
        } else {
            ScoredStructure::from_bundle(&bundle)
        };
        let report = evaluate_bundle(&structure, &ff, &args.energy.options());
        let mut stderr = io::stderr().lock();
        emit_bundle_energy(&mut stderr, &args.energy, &report)?;
    }

    Ok(())
}

fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ff = args.energy.load_ff()?;
    let pdb_text = fs::read_to_string(&args.pdb)?;
    let structure = ScoredStructure::from_pdb_str(&pdb_text)?;
    let opts = args.energy.options();

    let mode = args.mode.unwrap_or({
        if structure.chains.len() > 1 {
            ScoreMode::Both
        } else {
            ScoreMode::Internal
        }
    });

    let mut stdout = io::stdout().lock();
    match mode {
        ScoreMode::Internal => {
            let score = internal_energy(&structure, &ff, &opts);
            emit_monomer_energy(&mut stdout, &args.energy, &score)?;
        }
        ScoreMode::Interaction => {
            let score = interaction_energy(&structure, &ff, &opts);
            if args.energy.energy_json {
                writeln!(stdout, "{}", serde_json::to_string_pretty(&score)?)?;
            } else {
                write_score_text(&mut stdout, "interaction", &score, args.energy.energy_details)?;
            }
            stdout.flush()?;
        }
        ScoreMode::Both => {
            let report = evaluate_bundle(&structure, &ff, &opts);
            emit_bundle_energy(&mut stdout, &args.energy, &report)?;
        }
    }
    Ok(())
}

fn apply_param_override(
    target: &mut ParamValue,
    mean_opt: Option<f64>,
    range_opt: Option<f64>,
) {
    if mean_opt.is_none() && range_opt.is_none() {
        return;
    }
    let mean = mean_opt.unwrap_or_else(|| target.mean());
    let range = match range_opt {
        Some(r) => r,
        None => match *target {
            ParamValue::Static(_) => 0.0,
            ParamValue::Dynamic(pr) => pr.half_range,
        },
    };
    *target = ParamValue::dynamic(mean, range);
}

fn run_optimize(args: Box<OptimizeArgs>) -> Result<(), Box<dyn std::error::Error>> {
    let args = *args;
    let mut spec = match args.config {
        Some(path) => serde_yml::from_str::<OptimizeSpec>(&fs::read_to_string(path)?)?,
        None => OptimizeSpec::default(),
    };

    if let Some(sym) = args.symmetry {
        spec.symmetry = sym.parse()?;
    }
    if args.keep_gly {
        spec.keep_gly = true;
    }
    if let Some(obj) = args.objective {
        spec.objective = obj.into();
    }
    if let Some(p) = args.pop_size {
        spec.de.pop_size = p;
    }
    if let Some(g) = args.generations {
        spec.de.generations = g;
    }
    if let Some(c) = args.cxpb {
        spec.de.cxpb = c;
    }
    if let Some(w) = args.diff_weight {
        spec.de.diff_weight = w;
    }
    if let Some(n) = args.neighbours {
        spec.de.neighbours = Some(n);
    }
    if let Some(s) = args.seed {
        spec.de.seed = s;
    }
    if let Some(t) = args.threads {
        spec.de.threads = Some(t);
    }

    if let Some(tpl) = spec.helices.first_mut() {
        if let Some(seq) = args.sequence {
            tpl.sequence = seq;
        }
        apply_param_override(
            &mut tpl.bundle_radius,
            args.bundle_radius,
            args.bundle_radius_range,
        );
        if args.pitch.is_some() || args.pitch_range.is_some() {
            let mut p = tpl.pitch.unwrap_or_else(|| ParamValue::fixed(140.0));
            apply_param_override(&mut p, args.pitch, args.pitch_range);
            tpl.pitch = Some(p);
        }
        apply_param_override(
            &mut tpl.phi_c_alpha_deg,
            args.phi_c_alpha,
            args.phi_c_alpha_range,
        );
        apply_param_override(&mut tpl.z_offset, args.z_offset, args.z_offset_range);
        if args.res_per_turn.is_some() || args.res_per_turn_range.is_some() {
            let mut rpt = tpl
                .residues_per_turn
                .unwrap_or_else(|| ParamValue::fixed(tpl.minor_params.residues_per_turn));
            apply_param_override(&mut rpt, args.res_per_turn, args.res_per_turn_range);
            tpl.residues_per_turn = Some(rpt);
        }
        if let Some(tm) = args.twist_mode {
            tpl.twist_mode = tm.into();
        }
    }

    let ff = match &args.force_field {
        Some(p) => BudeForceField::from_file(p)?,
        None => BudeForceField::load_default()?,
    };
    let energy_opts = EnergyOptions {
        include_interactions: false,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: args.legacy_electrostatics,
    };
    let daspr_cfg = DasprConfig {
        daspr_bin: args.daspr_bin,
        rotlib_path: args.rotlib,
    };

    let result = optimize_bundle(&spec, &ff, &energy_opts, &daspr_cfg)?;
    fs::write(&args.output, &result.best_pdb)?;

    let log_text = result.format_log();
    print!("{log_text}");
    if let Some(log_path) = args.log_file {
        fs::write(log_path, &log_text)?;
    }
    if let Some(csv_path) = args.history_csv {
        fs::write(csv_path, result.history_to_csv())?;
    }
    if let Some(json_path) = args.history_json {
        fs::write(json_path, serde_json::to_string_pretty(&result.history)?)?;
    }
    Ok(())
}

fn parse_probe_offsets(raw: &str) -> Result<Vec<isize>, Box<dyn std::error::Error>> {
    let mut offsets = Vec::new();
    for tok in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        offsets.push(tok.parse::<isize>()?);
    }
    Ok(offsets)
}

fn parse_probe_residues(raw: &str) -> Result<Vec<CanonicalResidue>, Box<dyn std::error::Error>> {
    let mut residues = Vec::new();
    for tok in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let mut chars = tok.chars();
        let ch = chars
            .next()
            .ok_or_else(|| format!("empty residue token in '{raw}'"))?;
        if chars.next().is_some() {
            return Err(format!("expected single-letter residue code, got '{tok}'").into());
        }
        let (res, _, _) = CanonicalResidue::from_case_char(ch.to_ascii_uppercase())?;
        residues.push(res);
    }
    Ok(residues)
}

fn find_repo_script(filename: &str) -> Option<PathBuf> {
    let rel = PathBuf::from("scripts").join(filename);
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(filename);
    [rel, manifest].into_iter().find(|p| p.is_file())
}

fn preset_params(preset: Preset) -> HelixParams {
    match preset {
        Preset::Beta63 => HelixParams::beta_6_3(),
        Preset::Gramicidin => HelixParams::beta_6_3_gramicidin(),
        Preset::Polytheonamide => HelixParams::beta_6_3_polytheonamide(),
    }
}

fn anchor_slug(chirality: Chirality) -> &'static str {
    match chirality {
        Chirality::L => "L",
        Chirality::D => "D",
        Chirality::Achiral => "Achiral",
    }
}

fn offset_slug(offset: isize) -> String {
    if offset >= 0 {
        format!("p{offset}")
    } else {
        format!("m{}", offset.unsigned_abs())
    }
}

fn run_probe_pairs(args: ProbePairsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = ProbeConfig {
        host_length: args.host_length,
        offsets: parse_probe_offsets(&args.offsets)?,
        anchor: args.anchor.into(),
        include_proline: !args.exclude_proline,
        rank_by: args.rank_by.into(),
        parity_max_diff_threshold: args.parity_max_diff,
        parity_rmsd_threshold: args.parity_rmsd,
        skip_pack_errors: !args.strict,
        threads: args.threads,
        ..ProbeConfig::default()
    };

    if let Some(config_path) = &args.config {
        let mono = serde_yml::from_str::<MonomerSpec>(&fs::read_to_string(config_path)?)?;
        cfg.minor_params = mono.params;
    }
    if let Some(preset) = args.preset {
        cfg.minor_params = preset_params(preset);
    }
    if let Some(res_str) = &args.residues {
        cfg.residues = Some(parse_probe_residues(res_str)?);
    }

    let ff = match &args.force_field {
        Some(p) => BudeForceField::from_file(p)?,
        None => BudeForceField::load_default()?,
    };
    let energy_opts = EnergyOptions {
        include_interactions: false,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: args.legacy_electrostatics,
    };
    let daspr_cfg = DasprConfig {
        daspr_bin: args.daspr_bin,
        rotlib_path: args.rotlib,
    };

    let report = probe_pair_interactions(&cfg, &ff, &energy_opts, &daspr_cfg)?;
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, report.to_csv())?;

    if let Some(matrix_dir) = &args.matrix_dir {
        fs::create_dir_all(matrix_dir)?;
        for chirality in cfg.anchor.active_chirality_slots() {
            for &offset in &report.offsets {
                for &metric in &[RankBy::Dmc, RankBy::VsAla, RankBy::DirectPair] {
                    let filename = format!(
                        "{}_{}_{}.csv",
                        metric.slug(),
                        anchor_slug(chirality),
                        offset_slug(offset)
                    );
                    let path = Path::new(matrix_dir).join(filename);
                    fs::write(path, report.matrix_csv(chirality, offset, metric)?)?;
                }
            }
        }
    }

    print!("{}", report.format_summary(args.top));

    if let Some(plot_dir) = &args.plot_dir {
        let script = find_repo_script("plot_pair_heatmaps.py").ok_or(
            "could not locate scripts/plot_pair_heatmaps.py relative to cwd or CARGO_MANIFEST_DIR",
        )?;
        let status = ProcessCommand::new("python3")
            .arg(&script)
            .arg(&args.output)
            .arg("--output-dir")
            .arg(plot_dir)
            .arg("--metric")
            .arg(match cfg.rank_by {
                RankBy::Dmc => "ddE_dmc",
                RankBy::VsAla => "dE_vs_ala",
                RankBy::DirectPair => "direct_total",
            })
            .status()?;
        if !status.success() {
            return Err(format!("heatmap plotting script exited with {status}").into());
        }
    }
    Ok(())
}

fn run_design_seq(args: DesignSeqArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut target_params = HelixParams::beta_6_3();
    if let Some(config_path) = &args.config {
        let mono = serde_yml::from_str::<MonomerSpec>(&fs::read_to_string(config_path)?)?;
        target_params = mono.params;
    }
    if let Some(preset) = args.preset {
        target_params = preset_params(preset);
    }
    if let Some(handedness) = args.target_handedness {
        let desired: Handedness = handedness.into();
        if target_params.handedness != desired {
            target_params = target_params.mirror_enantiomer();
        }
    }

    let frame_mode = match args.frame_mode {
        CliFrameMode::Alternating => FrameMode::Alternating {
            max_alternating_cols: args.max_alternating_cols,
        },
        CliFrameMode::Strict => FrameMode::Strict,
    };

    let cfg = DesignConfig {
        length: args.length,
        n_cap: if args.no_n_cap { NCap::None } else { NCap::Acetyl },
        c_cap: if args.no_c_cap { CCap::Hydroxyl } else { CCap::Amide },
        target_params,
        frame_mode,
        constraints: DesignConstraints {
            allow_cys: args.allow_cys,
            allow_met: args.allow_met,
            allow_arg: !args.exclude_arg,
            allow_pro: args.allow_pro,
            allow_terminal_pro: args.allow_terminal_pro,
            min_net_charge: args.min_charge,
            max_net_charge: args.max_charge,
            min_trp_tyr: args.min_trp_tyr,
            max_aromatic: args.max_aromatic,
            max_his: args.max_his,
            max_aliphatic: args.max_aliphatic,
            no_aliphatic_ladders: args.no_aliphatic_ladders,
            min_polar: args.min_polar,
            ..DesignConstraints::default()
        },
        specificity_weight: args.specificity_weight,
        sa_restarts: args.sa_restarts,
        sa_steps: args.sa_steps,
        top_k_3d: args.top_k_3d,
        refine_termini: args.refine,
        refine_top_n: args.refine_top_n,
        seed: args.seed,
        threads: args.threads,
    };

    let ff = match &args.force_field {
        Some(p) => BudeForceField::from_file(p)?,
        None => BudeForceField::load_default()?,
    };
    let energy_opts = EnergyOptions {
        include_interactions: false,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: args.legacy_electrostatics,
    };
    let daspr_cfg = DasprConfig {
        daspr_bin: args.daspr_bin,
        rotlib_path: args.rotlib,
    };

    let surrogate = match &args.pair_csv {
        Some(path) => Some(PairSurrogateModel::from_csv(path, args.pair_handedness.into())?),
        None => None,
    };

    let report = design_sequences(&cfg, surrogate.as_ref(), &ff, &energy_opts, &daspr_cfg)?;

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, report.to_csv())?;

    if let Some(pdb_dir) = &args.pdb_dir {
        fs::create_dir_all(pdb_dir)?;
        for cand in report.candidates.iter().take(args.top) {
            let tgt_path = Path::new(pdb_dir).join(format!("design_{:02}_target.pdb", cand.rank));
            let dec_path = Path::new(pdb_dir).join(format!("design_{:02}_decoy.pdb", cand.rank));
            fs::write(tgt_path, &cand.target_pdb)?;
            fs::write(dec_path, &cand.decoy_pdb)?;
        }
    }

    print!("{}", report.format_summary(args.top));

    if let Some(plot_path) = &args.plot {
        let script = find_repo_script("plot_design_pareto.py").ok_or(
            "could not locate scripts/plot_design_pareto.py relative to cwd or CARGO_MANIFEST_DIR",
        )?;
        let status = ProcessCommand::new("python3")
            .arg(&script)
            .arg(&args.output)
            .arg("--output")
            .arg(plot_path)
            .arg("--host-energy")
            .arg(format!("{:.4}", report.host_3d_target_energy))
            .status()?;
        if !status.success() {
            return Err(format!("design Pareto plotting script exited with {status}").into());
        }
    }
    Ok(())
}

fn resolve_target_helix_params(
    config: Option<&PathBuf>,
    preset: Option<Preset>,
) -> Result<HelixParams, Box<dyn std::error::Error>> {
    let mut target_params = HelixParams::beta_6_3();
    if let Some(config_path) = config {
        let mono = serde_yml::from_str::<MonomerSpec>(&fs::read_to_string(config_path)?)?;
        target_params = mono.params;
    }
    if let Some(p) = preset {
        target_params = preset_params(p);
    }
    Ok(target_params)
}

fn parse_seam_patterns(cli: CliSeamPattern) -> Vec<SeamPattern> {
    match cli {
        CliSeamPattern::Strict6 => vec![SeamPattern::Strict6],
        CliSeamPattern::Staggered6676 => vec![SeamPattern::Staggered6676],
        CliSeamPattern::Both => vec![SeamPattern::Strict6, SeamPattern::Staggered6676],
    }
}

fn run_design_oligomers(args: DesignOligomersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let target_params = resolve_target_helix_params(args.config.as_ref(), args.preset)?;
    let seam_patterns = parse_seam_patterns(args.seam_pattern);
    let seam_widths = match args.seam_width {
        CliSeamWidth::One => vec![SeamWidth::Single],
        CliSeamWidth::Two => vec![SeamWidth::PairedLD],
        CliSeamWidth::Both => vec![SeamWidth::Single, SeamWidth::PairedLD],
    };
    let seam_compositions = match args.seam_comp {
        CliSeamComp::Aliphatic => vec![SeamComposition::AliphaticOnly],
        CliSeamComp::Mixed => vec![SeamComposition::MixedAliphaticAromatic],
        CliSeamComp::Both => vec![
            SeamComposition::AliphaticOnly,
            SeamComposition::MixedAliphaticAromatic,
        ],
    };
    let edge_patterns = match args.edge_pattern {
        CliEdgePattern::Parallel => vec![EdgePattern::ParallelBias],
        CliEdgePattern::Alternating => vec![EdgePattern::Alternating],
        CliEdgePattern::Neutral => vec![EdgePattern::NeutralPolar],
        CliEdgePattern::All => vec![
            EdgePattern::ParallelBias,
            EdgePattern::Alternating,
            EdgePattern::NeutralPolar,
        ],
    };

    let mut states = Vec::new();
    for tok in args.states.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        states.push(OligomerState::parse(tok)?);
    }

    let cfg = OligomerWorkflowConfig {
        length: args.length,
        n_cap: if args.no_n_cap { NCap::None } else { NCap::Acetyl },
        c_cap: if args.no_c_cap { CCap::Hydroxyl } else { CCap::Amide },
        target_params,
        seam_patterns,
        seam_widths,
        seam_compositions,
        edge_patterns,
        exclude_arg: args.exclude_arg,
        allow_terminal_wy: !args.no_terminal_wy,
        min_monomer_hand_gap: args.min_hand_gap,
        sa_restarts_per_class: args.sa_restarts,
        sa_steps: args.sa_steps,
        top_monomers: args.top_monomers,
        states,
        run_coarse_screen: !args.skip_coarse,
        de_pop_size: args.de_pop,
        de_generations: args.de_gens,
        tail_drop_tol_per_chain: args.tail_drop_tol,
        seed: args.seed,
        threads: args.threads,
        ..OligomerWorkflowConfig::default()
    };

    let ff = match &args.force_field {
        Some(p) => BudeForceField::from_file(p)?,
        None => BudeForceField::load_default()?,
    };
    let energy_opts = EnergyOptions {
        include_interactions: false,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: args.legacy_electrostatics,
    };
    let daspr_cfg = DasprConfig {
        daspr_bin: args.daspr_bin,
        rotlib_path: args.rotlib,
    };
    let surrogate = match &args.pair_csv {
        Some(path) => Some(PairSurrogateModel::from_csv(path, args.pair_handedness.into())?),
        None => None,
    };

    let report = design_oligomers(&cfg, surrogate.as_ref(), &ff, &energy_opts, &daspr_cfg)?;

    fs::create_dir_all(&args.output_dir)?;
    let mono_csv = args.output_dir.join("monomer_screen.csv");
    let oligo_csv = args.output_dir.join("oligomer_screen.csv");
    let pdb_dir = args.output_dir.join("pdbs");
    fs::write(&mono_csv, report.monomer_csv())?;
    fs::write(&oligo_csv, report.oligomer_csv())?;
    fs::create_dir_all(&pdb_dir)?;
    for ev in &report.evaluations {
        let pdb_path = pdb_dir.join(format!("cand_{:02}_{}.pdb", ev.candidate_id, ev.state_label));
        fs::write(pdb_path, &ev.best_pdb)?;
    }

    print!("{}", report.format_summary());

    if args.plot {
        let script = find_repo_script("plot_oligomer_landscape.py").ok_or(
            "could not locate scripts/plot_oligomer_landscape.py relative to cwd or CARGO_MANIFEST_DIR",
        )?;
        let plot_path = args.output_dir.join("oligomer_landscape.png");
        let status = ProcessCommand::new("python3")
            .arg(&script)
            .arg(&mono_csv)
            .arg(&oligo_csv)
            .arg("--output")
            .arg(&plot_path)
            .status()?;
        if !status.success() {
            return Err(format!("oligomer landscape script exited with {status}").into());
        }
    }
    Ok(())
}

fn run_design_hexamers(args: DesignHexamersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let target_params = resolve_target_helix_params(args.config.as_ref(), args.preset)?;
    let seam_patterns = parse_seam_patterns(args.seam_pattern);

    let mut lumen_modes = Vec::new();
    for tok in args.lumen_mode.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        lumen_modes.push(LumenMode::parse(tok)?);
    }
    let mut states = Vec::new();
    for tok in args.states.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        states.push(HexOligomerTarget::parse(tok)?);
    }

    let cfg = HexamerWorkflowConfig {
        length: args.length,
        n_cap: if args.no_n_cap { NCap::None } else { NCap::Acetyl },
        c_cap: if args.no_c_cap { CCap::Hydroxyl } else { CCap::Amide },
        target_params,
        target_oligomer_n: args.target_n,
        seam_patterns,
        lumen_modes,
        exclude_arg: args.exclude_arg,
        min_monomer_hand_gap: args.min_hand_gap,
        sa_restarts_per_class: args.sa_restarts,
        sa_steps: args.sa_steps,
        coarse_pool_size: args.coarse_pool,
        top_de: args.top_de,
        states,
        de_pop_size: args.de_pop,
        de_generations: args.de_gens,
        tail_drop_tol_per_chain: args.tail_drop_tol,
        seed: args.seed,
        threads: args.threads,
        ..HexamerWorkflowConfig::default()
    };

    let ff = match &args.force_field {
        Some(p) => BudeForceField::from_file(p)?,
        None => BudeForceField::load_default()?,
    };
    let energy_opts = EnergyOptions {
        include_interactions: false,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: args.legacy_electrostatics,
    };
    let daspr_cfg = DasprConfig {
        daspr_bin: args.daspr_bin,
        rotlib_path: args.rotlib,
    };
    let surrogate = match &args.pair_csv {
        Some(path) => Some(PairSurrogateModel::from_csv(path, args.pair_handedness.into())?),
        None => None,
    };

    let report = design_hexamers(&cfg, surrogate.as_ref(), &ff, &energy_opts, &daspr_cfg)?;

    fs::create_dir_all(&args.output_dir)?;
    let mono_csv = args.output_dir.join("monomer_screen.csv");
    let coarse_csv = args.output_dir.join("coarse_screen.csv");
    let hex_csv = args.output_dir.join("hexamer_screen.csv");
    let pdb_dir = args.output_dir.join("pdbs");
    fs::write(&mono_csv, report.monomer_csv())?;
    fs::write(&coarse_csv, report.coarse_csv())?;
    fs::write(&hex_csv, report.hexamer_csv())?;
    fs::create_dir_all(&pdb_dir)?;
    for ev in &report.de_evaluations {
        let pdb_path = pdb_dir.join(format!("cand_{:02}_{}.pdb", ev.candidate_id, ev.state_label));
        fs::write(pdb_path, &ev.best_pdb)?;
    }

    print!("{}", report.format_summary());

    if args.plot {
        let script = find_repo_script("plot_hexamer_landscape.py").ok_or(
            "could not locate scripts/plot_hexamer_landscape.py relative to cwd or CARGO_MANIFEST_DIR",
        )?;
        let plot_path = args.output_dir.join("hexamer_landscape.png");
        let status = ProcessCommand::new("python3")
            .arg(&script)
            .arg(&mono_csv)
            .arg(&hex_csv)
            .arg("--target-state")
            .arg(&report.target_state_label)
            .arg("--output")
            .arg(&plot_path)
            .status()?;
        if !status.success() {
            return Err(format!("hexamer landscape script exited with {status}").into());
        }
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.command {
        Command::Monomer(args) => run_monomer(args),
        Command::Bundle(args) => run_bundle(args),
        Command::Score(args) => run_score(args),
        Command::Optimize(args) => run_optimize(args),
        Command::ProbePairs(args) => run_probe_pairs(args),
        Command::DesignSeq(args) => run_design_seq(*args),
        Command::DesignOligomers(args) => run_design_oligomers(*args),
        Command::DesignHexamers(args) => run_design_hexamers(*args),
    };
    if let Err(err) = res {
        eprintln!("error: {err}");
        process::exit(1);
    }
}
