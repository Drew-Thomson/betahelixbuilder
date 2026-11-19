//! Parametric beta-helix peptide modelling library.
//!
//! Phase 1 implements Crick-style monomeric poly-D/L-alanine backbone generation,
//! case-based stereochemical sequence parsing, and standard PDB output.
//! Phase 2 adds supercoiled oligomeric bundles with Cn/Dn symmetry and heteromeric specs.
//! Phase 3 adds side-chain packing via `dASPR` for mixed L/D canonical sequences.
//! Phase 4 adds native Rust BUDEFF energy scoring for L- and D-amino acid structures.
//! Phase 5 adds Differential Evolution bundle parameter optimisation.

pub mod bundle;
pub mod energy;
pub mod geometry;
pub mod input;
pub mod optimization;
pub mod pdb;
pub mod sidechain;

pub use bundle::{Bundle, build_bundle, bundle_target_sequences, supercoil_samples};
pub use energy::{
    AtomFfParams, BudeForceField, BundleEnergyReport, ElectrostaticType, EnergyError,
    EnergyOptions, EnergyScore, PairInteraction, ScoredAtom, ScoredChain, ScoredResidue,
    ScoredStructure, evaluate_bundle, interaction_energy, internal_energy, pair_energy,
};
pub use geometry::{
    BuildError, Chain, CurveSample, ResidueAtoms, backbone_dihedrals, bond_angle_deg, bond_length,
    build_beta_helix, build_helix_along_curve, dihedral_deg, improper_chirality_deg, kabsch_rmsd,
};
pub use input::{
    BundleSpec, CanonicalResidue, Chirality, Handedness, HelixParams, HelixPlacementParams,
    InputError, MonomerSpec, Orientation, ResidueSpec, Sequence, Symmetry, TwistMode,
};
pub use optimization::{
    DeConfig, DeSolverResult, EvalRecord, GenerationStats, NamedParamValue, OptError,
    OptHelixSpec, OptObjective, OptimizeResult, OptimizeSpec, ParamDescriptor, ParamRange,
    ParamValue, differential_evolution, optimize_bundle,
};
pub use pdb::{bundle_to_pdb_string, to_pdb_string, write_bundle_pdb, write_pdb};
pub use sidechain::{
    DasprConfig, SidechainError, normalize_daspr_pdb, pack_bundle, pack_chain, pack_pdb_with_daspr,
    resolve_daspr_binary,
};
