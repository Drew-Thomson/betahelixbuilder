//! CLI entrypoint for generating monomeric and bundled parametric beta-helix PDB structures.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process;

use betahelixbuilder::{
    BundleSpec, DasprConfig, Handedness, HelixParams, HelixPlacementParams, MonomerSpec,
    Orientation, Sequence, Symmetry, TwistMode, build_beta_helix, build_bundle,
    bundle_target_sequences, pack_bundle, pack_chain, write_bundle_pdb, write_pdb,
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

#[derive(Debug, Parser)]
#[command(name = "betahelixbuilder")]
#[command(about = "Parametric Crick-style beta-helix monomer and bundle generator")]
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

    /// Output PDB file path (defaults to stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
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
    let mut writer = open_output(args.output)?;

    if spec.pack {
        let daspr_cfg = DasprConfig {
            daspr_bin: args.daspr_bin,
            rotlib_path: args.rotlib,
        };
        let packed_pdb = pack_chain(&chain, &target_seq, &daspr_cfg)?;
        writer.write_all(packed_pdb.as_bytes())?;
    } else {
        write_pdb(&chain, &mut writer)?;
    }
    writer.flush()?;
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
    let mut writer = open_output(args.output)?;
    if spec.pack {
        let daspr_cfg = DasprConfig {
            daspr_bin: args.daspr_bin,
            rotlib_path: args.rotlib,
        };
        let target_seqs = bundle_target_sequences(&spec)?;
        let packed_pdb = pack_bundle(&bundle, &target_seqs, &daspr_cfg)?;
        writer.write_all(packed_pdb.as_bytes())?;
    } else {
        write_bundle_pdb(&bundle, &mut writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.command {
        Command::Monomer(args) => run_monomer(args),
        Command::Bundle(args) => run_bundle(args),
    };
    if let Err(err) = res {
        eprintln!("error: {err}");
        process::exit(1);
    }
}
