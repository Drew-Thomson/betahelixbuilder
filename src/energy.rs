//! Native Rust implementation of the BUDE all-atom force field (BUDEFF).
//!
//! Adapted from the MIT-licensed BUDEFF library (Copyright (c) 2018 ISAMBARD / Sessions group)
//! and its D-amino acid force-field parameterization (`bude_2016v1.json`).

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;

use nalgebra::Point3;
use serde::{Deserialize, Serialize};

use crate::bundle::Bundle;
use crate::geometry::Chain;

const EMBEDDED_BUDE_2016V1: &str = include_str!("../resources/bude_2016v1.json");
const COULOMB_CONSTANT: f64 = 22.5;
const BACKBONE_ATOMS: [&str; 4] = ["N", "CA", "C", "O"];

/// Electrostatic interaction classification from BUDEFF parameter tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElectrostaticType {
    /// Non-electrostatic (`"-"`).
    None,
    /// Dipolar partial charge (`"D"`).
    Dipole,
    /// Formal charge (`"F"`, uses 4.0 Å cutoff between formal charges).
    Formal,
    /// Hydrogen-bond donor/acceptor (`"E"`, flips repulsive sign).
    DonorAcceptor,
}

impl ElectrostaticType {
    fn parse(code: &str) -> Result<Self, EnergyError> {
        match code.trim() {
            "-" => Ok(Self::None),
            "D" => Ok(Self::Dipole),
            "F" => Ok(Self::Formal),
            "E" => Ok(Self::DonorAcceptor),
            other => Err(EnergyError::InvalidForceField(format!(
                "unknown electrostatic type '{other}'"
            ))),
        }
    }
}

/// Per-atom BUDE force-field parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomFfParams {
    pub electrostatic_type: ElectrostaticType,
    pub radius: f64,
    pub hydrophobic_potential: f64,
    pub hardness: f64,
    pub dist_np_np: f64,
    pub dist_np_p: f64,
    pub radius_scale: f64,
    pub electrostatic_potential: f64,
}

/// Loaded BUDE force field parameter table (`residue -> atom -> AtomFfParams`).
#[derive(Debug, Clone)]
pub struct BudeForceField {
    residues: HashMap<String, HashMap<String, AtomFfParams>>,
    distance_cutoff: f64,
}

impl BudeForceField {
    /// Loads the bundled `bude_2016v1` force field (or `BUDEFF_FF_PATH` if set).
    pub fn load_default() -> Result<Self, EnergyError> {
        if let Ok(path) = env::var("BUDEFF_FF_PATH") {
            return Self::from_file(Path::new(&path));
        }
        Self::bude_2016v1()
    }

    /// Loads the embedded MIT-licensed `bude_2016v1` force field with L and D amino acids.
    pub fn bude_2016v1() -> Result<Self, EnergyError> {
        Self::from_json_str(EMBEDDED_BUDE_2016V1)
    }

    /// Loads a BUDE force-field JSON file from disk.
    pub fn from_file(path: &Path) -> Result<Self, EnergyError> {
        let raw = fs::read_to_string(path).map_err(|e| {
            EnergyError::Io(format!("failed to read force field '{}': {e}", path.display()))
        })?;
        Self::from_json_str(&raw)
    }

    /// Parses a BUDE force-field JSON string.
    pub fn from_json_str(raw: &str) -> Result<Self, EnergyError> {
        #[derive(Deserialize)]
        struct JsonAtomRow(String, f64, f64, f64, f64, f64, f64, f64);

        let parsed: HashMap<String, serde_json::Value> = serde_json::from_str(raw)
            .map_err(|e| EnergyError::InvalidForceField(format!("JSON parse error: {e}")))?;

        let mut residues = HashMap::new();
        let mut max_rad = 0.0_f64;
        let mut max_npnp = 0.0_f64;

        for (res_name, entry) in parsed {
            if res_name == "KEY" {
                continue;
            }
            let atoms_map: HashMap<String, JsonAtomRow> = serde_json::from_value(entry)
                .map_err(|e| {
                    EnergyError::InvalidForceField(format!(
                        "invalid parameter table under residue '{res_name}': {e}"
                    ))
                })?;

            let mut parsed_atoms = HashMap::with_capacity(atoms_map.len());
            for (atom_name, vals) in atoms_map {
                let electrostatic_type = ElectrostaticType::parse(&vals.0)?;
                max_rad = max_rad.max(vals.1);
                max_npnp = max_npnp.max(vals.4);
                parsed_atoms.insert(
                    atom_name.to_ascii_uppercase(),
                    AtomFfParams {
                        electrostatic_type,
                        radius: vals.1,
                        hydrophobic_potential: vals.2,
                        hardness: vals.3,
                        dist_np_np: vals.4,
                        dist_np_p: vals.5,
                        radius_scale: vals.6,
                        electrostatic_potential: vals.7,
                    },
                );
            }
            residues.insert(res_name.to_ascii_uppercase(), parsed_atoms);
        }

        if residues.is_empty() {
            return Err(EnergyError::InvalidForceField(
                "force field contains no residue entries".to_string(),
            ));
        }

        Ok(Self {
            residues,
            distance_cutoff: 2.0 * max_rad + max_npnp,
        })
    }

    /// Distance cutoff (Å) used for both residue-pair and atom-pair pruning.
    pub fn distance_cutoff(&self) -> f64 {
        self.distance_cutoff
    }

    /// Resolves parameters for `(res_name, atom_name)`, falling back to `"WLD"` wildcard.
    pub fn resolve_atom_params(&self, res_name: &str, atom_name: &str) -> Option<&AtomFfParams> {
        let res_upper = res_name.trim().to_ascii_uppercase();
        let atom_upper = atom_name.trim().to_ascii_uppercase();
        if let Some(res_map) = self.residues.get(&res_upper)
            && let Some(params) = res_map.get(&atom_upper)
        {
            return Some(params);
        }
        self.residues
            .get("WLD")
            .and_then(|wld| wld.get(&atom_upper))
    }
}

/// Configuration flags controlling BUDE energy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnergyOptions {
    /// Include individual non-zero pairwise interactions in [`EnergyScore::interactions`].
    pub include_interactions: bool,
    /// Ignore backbone–backbone pairs (`N, CA, C, O`) between consecutive residues (`seq_id + 1`).
    pub no_neighbour_backbone: bool,
    /// Mirror `budeff`'s `calculateEnergies.cpp` unassigned `electrostaticType` behavior for parity checks.
    pub legacy_unassigned_electrostatic_type: bool,
}

impl EnergyOptions {
    /// Standard BUDEFF defaults (`no_neighbour_backbone = true`, fixed electrostatics).
    pub fn standard() -> Self {
        Self {
            include_interactions: false,
            no_neighbour_backbone: true,
            legacy_unassigned_electrostatic_type: false,
        }
    }
}

/// Individual non-zero pairwise atom interaction score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairInteraction {
    pub chain_a: char,
    pub seq_id_a: i32,
    pub res_name_a: String,
    pub atom_name_a: String,
    pub chain_b: char,
    pub seq_id_b: i32,
    pub res_name_b: String,
    pub atom_name_b: String,
    pub distance: f64,
    pub steric: f64,
    pub desolvation: f64,
    pub charge: f64,
}

/// Aggregate BUDE energy score and optional pairwise interaction breakdown.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EnergyScore {
    pub total: f64,
    pub steric: f64,
    pub desolvation: f64,
    pub charge: f64,
    pub unparameterized_atoms: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub interactions: Vec<PairInteraction>,
}

impl fmt::Display for EnergyScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "total={:.3} (steric={:.3}, desolvation={:.3}, charge={:.3})",
            self.total, self.steric, self.desolvation, self.charge
        )
    }
}

/// Combined energy report for oligomeric bundles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleEnergyReport {
    pub internal: EnergyScore,
    pub interaction: EnergyScore,
}

impl fmt::Display for BundleEnergyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "internal: {}\ninteraction: {}",
            self.internal, self.interaction
        )
    }
}

/// Heavy atom prepared for BUDE scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredAtom {
    pub atom_name: String,
    pub element: String,
    pub pos: Point3<f64>,
}

/// Residue prepared for BUDE scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredResidue {
    pub seq_id: i32,
    pub res_name: String,
    pub atoms: Vec<ScoredAtom>,
    pub reference_pos: Point3<f64>,
}

/// Chain prepared for BUDE scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredChain {
    pub chain_id: char,
    pub residues: Vec<ScoredResidue>,
}

/// Full structure prepared for BUDE scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredStructure {
    pub chains: Vec<ScoredChain>,
}

impl ScoredStructure {
    /// Converts a generated monomeric [`Chain`] into a [`ScoredStructure`].
    pub fn from_chain(chain: &Chain) -> Self {
        Self {
            chains: vec![convert_chain(chain)],
        }
    }

    /// Converts a generated [`Bundle`] into a [`ScoredStructure`].
    pub fn from_bundle(bundle: &Bundle) -> Self {
        Self {
            chains: bundle.chains.iter().map(convert_chain).collect(),
        }
    }

    /// Parses heavy atoms from a PDB string (ignoring hydrogen atoms and secondary altLocs).
    pub fn from_pdb_str(pdb: &str) -> Result<Self, EnergyError> {
        let mut chains: Vec<ScoredChain> = Vec::new();
        let mut current_chain_idx: Option<usize> = None;
        let mut current_res_key: Option<(char, i32, char, String)> = None;
        let mut current_atoms: Vec<ScoredAtom> = Vec::new();

        let flush_residue = |chains: &mut Vec<ScoredChain>,
                             chain_idx: &mut Option<usize>,
                             res_key: &mut Option<(char, i32, char, String)>,
                             atoms: &mut Vec<ScoredAtom>| {
            if let Some((ch, seq_id, _icode, res_name)) = res_key.take()
                && !atoms.is_empty()
            {
                let reference_pos = pick_reference_pos(atoms);
                let res = ScoredResidue {
                    seq_id,
                    res_name,
                    atoms: std::mem::take(atoms),
                    reference_pos,
                };
                let idx = match *chain_idx {
                    Some(i) if chains[i].chain_id == ch => i,
                    _ => {
                        if let Some(pos) = chains.iter().position(|c| c.chain_id == ch) {
                            *chain_idx = Some(pos);
                            pos
                        } else {
                            chains.push(ScoredChain {
                                chain_id: ch,
                                residues: Vec::new(),
                            });
                            let idx = chains.len() - 1;
                            *chain_idx = Some(idx);
                            idx
                        }
                    }
                };
                chains[idx].residues.push(res);
            }
        };

        for line in pdb.lines() {
            if line.starts_with("END") {
                break;
            }
            if !line.starts_with("ATOM  ") || line.len() < 54 {
                continue;
            }
            let alt_loc = line.as_bytes()[16] as char;
            if alt_loc != ' ' && alt_loc != 'A' {
                continue;
            }
            let atom_name = line[12..16].trim().to_string();
            let res_name = line[17..20].trim().to_string();
            let chain_id = line.as_bytes()[21] as char;
            let seq_id: i32 = line[22..26].trim().parse().map_err(|_| {
                EnergyError::InvalidPdb(format!("invalid residue sequence number in line: {line}"))
            })?;
            let icode = line.as_bytes()[26] as char;
            let x: f64 = line[30..38].trim().parse().map_err(|_| {
                EnergyError::InvalidPdb(format!("invalid X coordinate in line: {line}"))
            })?;
            let y: f64 = line[38..46].trim().parse().map_err(|_| {
                EnergyError::InvalidPdb(format!("invalid Y coordinate in line: {line}"))
            })?;
            let z: f64 = line[46..54].trim().parse().map_err(|_| {
                EnergyError::InvalidPdb(format!("invalid Z coordinate in line: {line}"))
            })?;
            let element = if line.len() >= 78 {
                line[76..78].trim().to_string()
            } else {
                atom_name.chars().next().unwrap_or(' ').to_string()
            };

            if element.eq_ignore_ascii_case("H")
                || (element.is_empty() && atom_name.starts_with('H'))
            {
                continue;
            }

            let key = (chain_id, seq_id, icode, res_name);
            if current_res_key.as_ref().map(|k| (k.0, k.1, k.2)) != Some((key.0, key.1, key.2)) {
                flush_residue(
                    &mut chains,
                    &mut current_chain_idx,
                    &mut current_res_key,
                    &mut current_atoms,
                );
                current_res_key = Some(key);
            }
            current_atoms.push(ScoredAtom {
                atom_name,
                element,
                pos: Point3::new(x, y, z),
            });
        }

        flush_residue(
            &mut chains,
            &mut current_chain_idx,
            &mut current_res_key,
            &mut current_atoms,
        );

        if chains.is_empty() {
            return Err(EnergyError::EmptyStructure);
        }
        Ok(Self { chains })
    }
}

fn pick_reference_pos(atoms: &[ScoredAtom]) -> Point3<f64> {
    atoms
        .iter()
        .find(|a| a.atom_name == "CA")
        .unwrap_or(&atoms[0])
        .pos
}

fn convert_chain(chain: &Chain) -> ScoredChain {
    let mut residues = Vec::with_capacity(chain.residues.len() + 2);
    if let Some(ace) = &chain.n_cap {
        residues.push(ScoredResidue {
            seq_id: ace.seq_id as i32,
            res_name: "ACE".to_string(),
            atoms: vec![
                ScoredAtom {
                    atom_name: "CH3".to_string(),
                    element: "C".to_string(),
                    pos: ace.ch3,
                },
                ScoredAtom {
                    atom_name: "C".to_string(),
                    element: "C".to_string(),
                    pos: ace.c,
                },
                ScoredAtom {
                    atom_name: "O".to_string(),
                    element: "O".to_string(),
                    pos: ace.o,
                },
            ],
            reference_pos: ace.c,
        });
    }
    for r in &chain.residues {
        let mut atoms = Vec::with_capacity(6);
        atoms.push(ScoredAtom {
            atom_name: "N".to_string(),
            element: "N".to_string(),
            pos: r.n,
        });
        atoms.push(ScoredAtom {
            atom_name: "CA".to_string(),
            element: "C".to_string(),
            pos: r.ca,
        });
        atoms.push(ScoredAtom {
            atom_name: "C".to_string(),
            element: "C".to_string(),
            pos: r.c,
        });
        atoms.push(ScoredAtom {
            atom_name: "O".to_string(),
            element: "O".to_string(),
            pos: r.o,
        });
        if let Some(cb) = r.cb {
            atoms.push(ScoredAtom {
                atom_name: "CB".to_string(),
                element: "C".to_string(),
                pos: cb,
            });
        }
        if let Some(oxt) = r.oxt {
            atoms.push(ScoredAtom {
                atom_name: "OXT".to_string(),
                element: "O".to_string(),
                pos: oxt,
            });
        }
        residues.push(ScoredResidue {
            seq_id: r.seq_id as i32,
            res_name: r.res_name.to_string(),
            atoms,
            reference_pos: r.ca,
        });
    }
    if let Some(nh2) = &chain.c_cap {
        residues.push(ScoredResidue {
            seq_id: nh2.seq_id as i32,
            res_name: "NH2".to_string(),
            atoms: vec![ScoredAtom {
                atom_name: "N".to_string(),
                element: "N".to_string(),
                pos: nh2.n,
            }],
            reference_pos: nh2.n,
        });
    }
    ScoredChain {
        chain_id: chain.chain_id,
        residues,
    }
}

/// Evaluates the pairwise BUDEFF energy `(steric, desolvation, charge)` at distance `dist` (Å).
pub fn pair_energy(
    a: &AtomFfParams,
    b: &AtomFfParams,
    dist: f64,
    legacy_unassigned_electrostatic_type: bool,
) -> (f64, f64, f64) {
    let atoms_radius = a.radius + b.radius;

    // 1. Steric repulsion.
    let steric = if dist < atoms_radius {
        0.5 * (a.hardness + b.hardness) * (1.0 - dist / atoms_radius)
    } else {
        0.0
    };

    // 2. Electrostatics.
    let mut charge = 0.0;
    if legacy_unassigned_electrostatic_type {
        let elec_dist = 2.0;
        if dist < atoms_radius {
            charge = COULOMB_CONSTANT * a.electrostatic_potential * b.electrostatic_potential;
        } else if dist < atoms_radius + elec_dist {
            let base = COULOMB_CONSTANT * a.electrostatic_potential * b.electrostatic_potential;
            charge = base - base * (dist - atoms_radius) / elec_dist;
        }
    } else if a.electrostatic_type != ElectrostaticType::None
        && b.electrostatic_type != ElectrostaticType::None
    {
        let elec_dist = if a.electrostatic_type == ElectrostaticType::Formal
            && b.electrostatic_type == ElectrostaticType::Formal
        {
            4.0
        } else {
            2.0
        };
        if dist < atoms_radius {
            charge = COULOMB_CONSTANT * a.electrostatic_potential * b.electrostatic_potential;
        } else if dist < atoms_radius + elec_dist {
            let base = COULOMB_CONSTANT * a.electrostatic_potential * b.electrostatic_potential;
            charge = base - base * (dist - atoms_radius) / elec_dist;
        }
        if (a.electrostatic_type == ElectrostaticType::DonorAcceptor
            || b.electrostatic_type == ElectrostaticType::DonorAcceptor)
            && charge > 0.0
        {
            charge = -charge;
        }
    }

    // 3. Desolvation (NP-P and NP-NP).
    let mut desolvation = 0.0;
    if a.hydrophobic_potential > 0.0 && b.hydrophobic_potential < 0.0 {
        let base = 0.5 * (a.hydrophobic_potential - b.hydrophobic_potential);
        if dist < atoms_radius {
            desolvation = base;
        } else if dist < atoms_radius + a.dist_np_p {
            desolvation = base - base * (dist - atoms_radius) / a.dist_np_p;
        }
    } else if a.hydrophobic_potential < 0.0 && b.hydrophobic_potential > 0.0 {
        let base = 0.5 * (-a.hydrophobic_potential + b.hydrophobic_potential);
        if dist < atoms_radius {
            desolvation = base;
        } else if dist < atoms_radius + b.dist_np_p {
            desolvation = base - base * (dist - atoms_radius) / b.dist_np_p;
        }
    } else if a.hydrophobic_potential < 0.0 && b.hydrophobic_potential < 0.0 {
        let base = 0.5 * (a.hydrophobic_potential + b.hydrophobic_potential);
        if dist < atoms_radius {
            desolvation = base;
        } else if dist < atoms_radius + a.dist_np_np {
            desolvation = base - base * (dist - atoms_radius) / a.dist_np_np;
        }
    }

    (steric, desolvation, charge)
}

struct IndexedResidue<'a> {
    chain_id: char,
    seq_id: i32,
    res_name: &'a str,
    reference_pos: Point3<f64>,
    atoms: Vec<(&'a ScoredAtom, &'a AtomFfParams)>,
}

fn index_residue<'a>(
    chain_id: char,
    res: &'a ScoredResidue,
    ff: &'a BudeForceField,
    sidechain_only: bool,
    unparameterized: &mut Vec<String>,
) -> IndexedResidue<'a> {
    let mut atoms = Vec::with_capacity(res.atoms.len());
    for atom in &res.atoms {
        if sidechain_only && BACKBONE_ATOMS.contains(&atom.atom_name.as_str()) {
            continue;
        }
        if let Some(params) = ff.resolve_atom_params(&res.res_name, &atom.atom_name) {
            atoms.push((atom, params));
        } else {
            unparameterized.push(format!(
                "{chain_id}:{} {} {}",
                res.seq_id, res.res_name, atom.atom_name
            ));
        }
    }
    IndexedResidue {
        chain_id,
        seq_id: res.seq_id,
        res_name: &res.res_name,
        reference_pos: res.reference_pos,
        atoms,
    }
}

fn index_chain<'a>(
    chain: &'a ScoredChain,
    ff: &'a BudeForceField,
    unparameterized: &mut Vec<String>,
) -> Vec<IndexedResidue<'a>> {
    chain
        .residues
        .iter()
        .map(|res| index_residue(chain.chain_id, res, ff, false, unparameterized))
        .collect()
}

fn is_adjacent_backbone_atom(res_name: &str, atom_name: &str) -> bool {
    match res_name {
        "ACE" => matches!(atom_name, "CH3" | "C" | "O"),
        "NH2" => atom_name == "N",
        _ => BACKBONE_ATOMS.contains(&atom_name),
    }
}

fn score_residue_pairs(
    residues_a: &[IndexedResidue<'_>],
    residues_b: Option<&[IndexedResidue<'_>]>,
    ff: &BudeForceField,
    options: &EnergyOptions,
    unparameterized_atoms: Vec<String>,
) -> EnergyScore {
    let cutoff = ff.distance_cutoff();
    let cutoff_sq = cutoff * cutoff;
    let mut score = EnergyScore {
        unparameterized_atoms,
        ..Default::default()
    };

    let same_slice = residues_b.is_none();
    let rhs = residues_b.unwrap_or(residues_a);

    for (i, ra) in residues_a.iter().enumerate() {
        let j_start = if same_slice { i + 1 } else { 0 };
        for rb in &rhs[j_start..] {
            if (ra.reference_pos - rb.reference_pos).norm_squared() > cutoff_sq {
                continue;
            }
            let adjacent_same_chain = same_slice
                && options.no_neighbour_backbone
                && ra.chain_id == rb.chain_id
                && rb.seq_id == ra.seq_id + 1;

            for &(atom_a, params_a) in &ra.atoms {
                let a_is_bb = adjacent_same_chain
                    && is_adjacent_backbone_atom(ra.res_name, &atom_a.atom_name);
                for &(atom_b, params_b) in &rb.atoms {
                    if a_is_bb && is_adjacent_backbone_atom(rb.res_name, &atom_b.atom_name) {
                        continue;
                    }
                    let dist_sq = (atom_a.pos - atom_b.pos).norm_squared();
                    if dist_sq > cutoff_sq {
                        continue;
                    }
                    let dist = dist_sq.sqrt();
                    let (st, de, ch) = pair_energy(
                        params_a,
                        params_b,
                        dist,
                        options.legacy_unassigned_electrostatic_type,
                    );
                    score.steric += st;
                    score.desolvation += de;
                    score.charge += ch;
                    if options.include_interactions && (st + de + ch) != 0.0 {
                        score.interactions.push(PairInteraction {
                            chain_a: ra.chain_id,
                            seq_id_a: ra.seq_id,
                            res_name_a: ra.res_name.to_string(),
                            atom_name_a: atom_a.atom_name.clone(),
                            chain_b: rb.chain_id,
                            seq_id_b: rb.seq_id,
                            res_name_b: rb.res_name.to_string(),
                            atom_name_b: atom_b.atom_name.clone(),
                            distance: dist,
                            steric: st,
                            desolvation: de,
                            charge: ch,
                        });
                    }
                }
            }
        }
    }

    score.total = score.steric + score.desolvation + score.charge;
    score
}

/// Computes the BUDEFF internal energy of all residue pairs in `structure`.
pub fn internal_energy(
    structure: &ScoredStructure,
    ff: &BudeForceField,
    options: &EnergyOptions,
) -> EnergyScore {
    let mut unparameterized = Vec::new();
    let mut residues = Vec::new();
    for chain in &structure.chains {
        residues.extend(index_chain(chain, ff, &mut unparameterized));
    }
    score_residue_pairs(&residues, None, ff, options, unparameterized)
}

/// Computes the direct BUDEFF interaction energy between residues `seq_a` and `seq_b` in `chain_id`.
///
/// Residues are ordered ascending by `seq_id` so adjacent-backbone filtering matches
/// [`internal_energy`] for `|seq_a - seq_b| == 1`. Set `sidechain_only` to exclude
/// backbone `N`, `CA`, `C`, and `O` atoms.
pub fn direct_residue_pair_energy(
    structure: &ScoredStructure,
    ff: &BudeForceField,
    chain_id: char,
    seq_a: i32,
    seq_b: i32,
    sidechain_only: bool,
    options: &EnergyOptions,
) -> EnergyScore {
    if seq_a == seq_b {
        return EnergyScore::default();
    }
    let Some(chain) = structure.chains.iter().find(|c| c.chain_id == chain_id) else {
        return EnergyScore::default();
    };
    let (lo, hi) = if seq_a < seq_b {
        (seq_a, seq_b)
    } else {
        (seq_b, seq_a)
    };
    let (Some(res_lo), Some(res_hi)) = (
        chain.residues.iter().find(|r| r.seq_id == lo),
        chain.residues.iter().find(|r| r.seq_id == hi),
    ) else {
        return EnergyScore::default();
    };
    let mut unparameterized = Vec::new();
    let indexed = [
        index_residue(chain_id, res_lo, ff, sidechain_only, &mut unparameterized),
        index_residue(chain_id, res_hi, ff, sidechain_only, &mut unparameterized),
    ];
    score_residue_pairs(&indexed, None, ff, options, unparameterized)
}

/// Computes the BUDEFF inter-chain interaction energy across all distinct chain pairs.
pub fn interaction_energy(
    structure: &ScoredStructure,
    ff: &BudeForceField,
    options: &EnergyOptions,
) -> EnergyScore {
    let mut unparameterized = Vec::new();
    let per_chain_residues: Vec<Vec<IndexedResidue<'_>>> = structure
        .chains
        .iter()
        .map(|chain| index_chain(chain, ff, &mut unparameterized))
        .collect();

    let mut total_score = EnergyScore {
        unparameterized_atoms: unparameterized,
        ..Default::default()
    };

    for i in 0..per_chain_residues.len() {
        for j in (i + 1)..per_chain_residues.len() {
            let pair_score = score_residue_pairs(
                &per_chain_residues[i],
                Some(&per_chain_residues[j]),
                ff,
                options,
                Vec::new(),
            );
            total_score.steric += pair_score.steric;
            total_score.desolvation += pair_score.desolvation;
            total_score.charge += pair_score.charge;
            total_score.interactions.extend(pair_score.interactions);
        }
    }
    total_score.total = total_score.steric + total_score.desolvation + total_score.charge;
    total_score
}

/// Computes both `internal` and inter-chain `interaction` BUDEFF energies for a bundle.
pub fn evaluate_bundle(
    structure: &ScoredStructure,
    ff: &BudeForceField,
    options: &EnergyOptions,
) -> BundleEnergyReport {
    BundleEnergyReport {
        internal: internal_energy(structure, ff, options),
        interaction: interaction_energy(structure, ff, options),
    }
}

/// Errors produced during BUDEFF force-field loading or structure parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyError {
    Io(String),
    InvalidForceField(String),
    InvalidPdb(String),
    EmptyStructure,
}

impl fmt::Display for EnergyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "energy I/O error: {msg}"),
            Self::InvalidForceField(msg) => write!(f, "invalid BUDE force field: {msg}"),
            Self::InvalidPdb(msg) => write!(f, "invalid PDB input for scoring: {msg}"),
            Self::EmptyStructure => write!(f, "cannot score an empty PDB structure"),
        }
    }
}

impl std::error::Error for EnergyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::build_beta_helix;
    use crate::input::{HelixParams, Sequence};
    use nalgebra::Vector3;

    #[test]
    fn resolves_all_canonical_l_and_d_residues_in_bude_2016v1() {
        let ff = BudeForceField::bude_2016v1().expect("embedded JSON parses");
        assert!(ff.distance_cutoff() > 10.0);

        let all_seq = Sequence::parse("ACDEFGHIKLMNPQRSTVWYacdefhiklmnpqrstvwy").unwrap();
        for r in &all_seq.residues {
            let code = r.pdb_name();
            assert!(ff.resolve_atom_params(code, "N").is_some(), "missing N for {code}");
            assert!(ff.resolve_atom_params(code, "CA").is_some(), "missing CA for {code}");
            assert!(ff.resolve_atom_params(code, "C").is_some(), "missing C for {code}");
            assert!(ff.resolve_atom_params(code, "O").is_some(), "missing O for {code}");
            if code != "GLY" {
                assert!(ff.resolve_atom_params(code, "CB").is_some(), "missing CB for {code}");
            }
        }
    }

    #[test]
    fn analytical_pair_energy_terms_behave_as_expected() {
        let ff = BudeForceField::bude_2016v1().unwrap();
        let cb_ala = ff.resolve_atom_params("ALA", "CB").unwrap();
        let od1_das = ff.resolve_atom_params("DAS", "OD1").unwrap();
        let nz_lys = ff.resolve_atom_params("LYS", "NZ").unwrap();

        // Steric clash at short distance (< radius sum).
        let (st, _, _) = pair_energy(cb_ala, cb_ala, 1.5, false);
        assert!(st > 5.0, "expected steric penalty, got {st}");

        // Hydrophobic-hydrophobic attraction (both negative hydrophobic potential).
        let (st_np, de_np, _) = pair_energy(cb_ala, cb_ala, 3.5, false);
        assert_eq!(st_np, 0.0);
        assert!(de_np < 0.0, "expected favorable hydrophobic desolvation, got {de_np}");

        // Opposite formal charges attract within 4.0 Å beyond contact.
        let (_, _, ch_attr) = pair_energy(nz_lys, od1_das, 4.5, false);
        assert!(ch_attr < 0.0, "expected attractive formal charge, got {ch_attr}");

        // Donor-acceptor sign flip vs legacy unassigned mode.
        let og_ser = ff.resolve_atom_params("SER", "OG").unwrap();
        let (_, _, ch_fixed) = pair_energy(og_ser, og_ser, 2.8, false);
        let (_, _, ch_legacy) = pair_energy(og_ser, og_ser, 2.8, true);
        assert!(ch_fixed < 0.0, "expected DonorAcceptor sign flip, got {ch_fixed}");
        assert!(ch_legacy > 0.0, "expected positive legacy charge, got {ch_legacy}");
    }

    #[test]
    fn internal_energy_is_rigid_motion_invariant() {
        let seq = Sequence::parse("AaAaAaAaAaAa").unwrap();
        let chain = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();
        let ff = BudeForceField::bude_2016v1().unwrap();
        let opts = EnergyOptions {
            include_interactions: true,
            ..EnergyOptions::standard()
        };

        let s1 = ScoredStructure::from_chain(&chain);
        let e1 = internal_energy(&s1, &ff, &opts);
        assert!(!e1.interactions.is_empty());

        let rot = nalgebra::Rotation3::from_euler_angles(0.7, -1.2, 0.4);
        let shift = Vector3::new(15.0, -8.0, 22.0);
        let mut moved_chain = chain.clone();
        for res in &mut moved_chain.residues {
            res.n = rot * res.n + shift;
            res.ca = rot * res.ca + shift;
            res.c = rot * res.c + shift;
            res.o = rot * res.o + shift;
            if let Some(cb) = &mut res.cb {
                *cb = rot * (*cb) + shift;
            }
            if let Some(oxt) = &mut res.oxt {
                *oxt = rot * (*oxt) + shift;
            }
        }
        let s2 = ScoredStructure::from_chain(&moved_chain);
        let e2 = internal_energy(&s2, &ff, &opts);
        assert!((e1.total - e2.total).abs() < 1e-10);
        assert_eq!(e1.interactions.len(), e2.interactions.len());
    }

    #[test]
    fn direct_residue_pair_matches_internal_interactions_and_respects_order() {
        let seq = Sequence::parse("AaAaAaAaAaAa").unwrap();
        let chain = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();
        let ff = BudeForceField::bude_2016v1().unwrap();
        let opts = EnergyOptions {
            include_interactions: true,
            ..EnergyOptions::standard()
        };
        let structure = ScoredStructure::from_chain(&chain);
        let full = internal_energy(&structure, &ff, &opts);

        for &(a, b) in &[(3_i32, 9_i32), (3, 4), (4, 3)] {
            let lo = a.min(b);
            let hi = a.max(b);
            let expected: f64 = full
                .interactions
                .iter()
                .filter(|x| x.seq_id_a == lo && x.seq_id_b == hi)
                .map(|x| x.steric + x.desolvation + x.charge)
                .sum();
            let direct = direct_residue_pair_energy(&structure, &ff, 'A', a, b, false, &opts);
            assert!(
                (direct.total - expected).abs() < 1e-12,
                "pair ({a},{b}) mismatch: {} vs {expected}",
                direct.total
            );
        }

        let sc_only = direct_residue_pair_energy(&structure, &ff, 'A', 3, 9, true, &opts);
        for inter in &sc_only.interactions {
            assert!(!BACKBONE_ATOMS.contains(&inter.atom_name_a.as_str()));
            assert!(!BACKBONE_ATOMS.contains(&inter.atom_name_b.as_str()));
        }
    }

    #[test]
    fn mirror_enantiomer_has_identical_internal_energy_for_inverted_chirality_sequence() {
        let ff = BudeForceField::bude_2016v1().unwrap();
        let opts = EnergyOptions::standard();

        let seq_r = Sequence::parse("Ac-AaAaAaAaAaAaAaAaAaAaAaAaAaAaAa-NH2").unwrap();
        let chain_r = build_beta_helix(&seq_r, &HelixParams::beta_6_3(), 'A').unwrap();
        let e_r = internal_energy(&ScoredStructure::from_chain(&chain_r), &ff, &opts);

        let seq_l = Sequence::parse("Ac-aAaAaAaAaAaAaAaAaAaAaAaAaAaAaA-NH2").unwrap();
        let mirror_params = HelixParams::beta_6_3().mirror_enantiomer();
        let chain_l = build_beta_helix(&seq_l, &mirror_params, 'A').unwrap();
        let e_l = internal_energy(&ScoredStructure::from_chain(&chain_l), &ff, &opts);

        assert!(
            (e_r.total - e_l.total).abs() < 1e-10,
            "mirror internal energy mismatch: right={} vs left={}",
            e_r.total,
            e_l.total
        );
    }
}
