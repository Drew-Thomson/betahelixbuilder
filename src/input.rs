//! Sequence parsing, stereochemistry conventions, helix parameters, and validation.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

/// Stereochemical configuration of an amino acid residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Chirality {
    /// Standard L-amino acid (uppercase one-letter code).
    L,
    /// D-amino acid (lowercase one-letter code).
    D,
    /// Achiral glycine (`G` or `g`).
    Achiral,
}

/// Helix handedness around the N-to-C axial direction (+Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handedness {
    /// Right-handed helix: azimuth increases (+Δθ) from N- to C-terminus.
    Right,
    /// Left-handed helix: azimuth decreases (-Δθ) from N- to C-terminus.
    Left,
}

impl Handedness {
    /// Returns +1.0 for right-handed and -1.0 for left-handed helices.
    pub fn sign(self) -> f64 {
        match self {
            Self::Right => 1.0,
            Self::Left => -1.0,
        }
    }
}

/// Canonical 20 amino acid types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalResidue {
    Ala,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

struct ResidueTableEntry {
    one_letter: char,
    residue: CanonicalResidue,
    l_code: &'static str,
    d_code: &'static str,
}

const RESIDUE_TABLE: &[ResidueTableEntry] = &[
    ResidueTableEntry { one_letter: 'A', residue: CanonicalResidue::Ala, l_code: "ALA", d_code: "DAL" },
    ResidueTableEntry { one_letter: 'R', residue: CanonicalResidue::Arg, l_code: "ARG", d_code: "DAR" },
    ResidueTableEntry { one_letter: 'N', residue: CanonicalResidue::Asn, l_code: "ASN", d_code: "DSG" },
    ResidueTableEntry { one_letter: 'D', residue: CanonicalResidue::Asp, l_code: "ASP", d_code: "DAS" },
    ResidueTableEntry { one_letter: 'C', residue: CanonicalResidue::Cys, l_code: "CYS", d_code: "DCY" },
    ResidueTableEntry { one_letter: 'Q', residue: CanonicalResidue::Gln, l_code: "GLN", d_code: "DGN" },
    ResidueTableEntry { one_letter: 'E', residue: CanonicalResidue::Glu, l_code: "GLU", d_code: "DGL" },
    ResidueTableEntry { one_letter: 'G', residue: CanonicalResidue::Gly, l_code: "GLY", d_code: "GLY" },
    ResidueTableEntry { one_letter: 'H', residue: CanonicalResidue::His, l_code: "HIS", d_code: "DHI" },
    ResidueTableEntry { one_letter: 'I', residue: CanonicalResidue::Ile, l_code: "ILE", d_code: "DIL" },
    ResidueTableEntry { one_letter: 'L', residue: CanonicalResidue::Leu, l_code: "LEU", d_code: "DLE" },
    ResidueTableEntry { one_letter: 'K', residue: CanonicalResidue::Lys, l_code: "LYS", d_code: "DLY" },
    ResidueTableEntry { one_letter: 'M', residue: CanonicalResidue::Met, l_code: "MET", d_code: "MED" },
    ResidueTableEntry { one_letter: 'F', residue: CanonicalResidue::Phe, l_code: "PHE", d_code: "DPN" },
    ResidueTableEntry { one_letter: 'P', residue: CanonicalResidue::Pro, l_code: "PRO", d_code: "DPR" },
    ResidueTableEntry { one_letter: 'S', residue: CanonicalResidue::Ser, l_code: "SER", d_code: "DSN" },
    ResidueTableEntry { one_letter: 'T', residue: CanonicalResidue::Thr, l_code: "THR", d_code: "DTH" },
    ResidueTableEntry { one_letter: 'W', residue: CanonicalResidue::Trp, l_code: "TRP", d_code: "DTR" },
    ResidueTableEntry { one_letter: 'Y', residue: CanonicalResidue::Tyr, l_code: "TYR", d_code: "DTY" },
    ResidueTableEntry { one_letter: 'V', residue: CanonicalResidue::Val, l_code: "VAL", d_code: "DVA" },
];

impl CanonicalResidue {
    /// Parses a case-encoded amino acid character (uppercase = L, lowercase = D; glycine must be uppercase `G`).
    pub fn from_case_char(ch: char) -> Result<(Self, Chirality, Chirality), InputError> {
        if ch == 'g' {
            return Err(InputError::LowercaseGlycine);
        }
        let upper = ch.to_ascii_uppercase();
        let entry = RESIDUE_TABLE
            .iter()
            .find(|e| e.one_letter == upper)
            .ok_or(InputError::UnsupportedResidueChar(ch))?;

        let parity = if ch.is_ascii_uppercase() {
            Chirality::L
        } else {
            Chirality::D
        };
        let chirality = if entry.residue == CanonicalResidue::Gly {
            Chirality::Achiral
        } else {
            parity
        };
        Ok((entry.residue, chirality, parity))
    }

    /// Returns the dASPR-compatible case-encoded one-letter code (`G` for Gly).
    pub fn case_char(self, chirality: Chirality) -> char {
        let entry = RESIDUE_TABLE
            .iter()
            .find(|e| e.residue == self)
            .expect("all CanonicalResidue variants exist in RESIDUE_TABLE");
        match chirality {
            Chirality::D => entry.one_letter.to_ascii_lowercase(),
            Chirality::L | Chirality::Achiral => entry.one_letter,
        }
    }

    /// Returns the standard wwPDB three-letter component ID for the given chirality.
    pub fn pdb_code(self, chirality: Chirality) -> &'static str {
        let entry = RESIDUE_TABLE
            .iter()
            .find(|e| e.residue == self)
            .expect("all CanonicalResidue variants exist in RESIDUE_TABLE");
        match chirality {
            Chirality::D => entry.d_code,
            Chirality::L | Chirality::Achiral => entry.l_code,
        }
    }
}

/// Parsed residue specification in a peptide sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidueSpec {
    /// 1-based residue sequence number.
    pub seq_id: usize,
    /// Amino acid identity.
    pub residue: CanonicalResidue,
    /// True chemical stereochemistry (`Achiral` for Gly).
    pub chirality: Chirality,
    /// Backbone parity slot (`L` or `D`), preserving case-based parity even for Gly.
    pub backbone_parity: Chirality,
}

impl ResidueSpec {
    /// Returns the standard wwPDB 3-letter residue name.
    pub fn pdb_name(&self) -> &'static str {
        self.residue.pdb_code(self.chirality)
    }

    /// Converts any residue to poly-D/L-alanine (retaining `GLY` only if `keep_gly` is true).
    pub fn to_poly_ala(self, keep_gly: bool) -> Self {
        if keep_gly && self.residue == CanonicalResidue::Gly {
            return self;
        }
        Self {
            seq_id: self.seq_id,
            residue: CanonicalResidue::Ala,
            chirality: self.backbone_parity,
            backbone_parity: self.backbone_parity,
        }
    }
}

/// Validated peptide sequence with per-residue stereochemistry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub residues: Vec<ResidueSpec>,
}

fn flip_parity(parity: Chirality, distance: usize) -> Chirality {
    if distance.is_multiple_of(2) {
        parity
    } else {
        match parity {
            Chirality::L | Chirality::Achiral => Chirality::D,
            Chirality::D => Chirality::L,
        }
    }
}

impl Sequence {
    /// Parses a case-based sequence string (uppercase = L, lowercase = D, uppercase `G` = Gly).
    ///
    /// Glycine backbone parity (L-slot vs D-slot) is inferred from the alternating chirality
    /// of neighboring non-glycine residues.
    pub fn parse(raw: &str) -> Result<Self, InputError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(InputError::EmptySequence);
        }
        let mut residues = Vec::with_capacity(trimmed.len());
        for (idx, ch) in trimmed.chars().enumerate() {
            let (residue, chirality, backbone_parity) = CanonicalResidue::from_case_char(ch)?;
            residues.push(ResidueSpec {
                seq_id: idx + 1,
                residue,
                chirality,
                backbone_parity,
            });
        }

        // Assign Gly backbone parity by alternating from nearest preceding (or following) chiral residue.
        let mut last_chiral: Option<(usize, Chirality)> = None;
        for (i, r) in residues.iter_mut().enumerate() {
            if r.residue != CanonicalResidue::Gly {
                last_chiral = Some((i, r.backbone_parity));
            } else if let Some((anchor_idx, anchor_parity)) = last_chiral {
                r.backbone_parity = flip_parity(anchor_parity, i - anchor_idx);
            }
        }
        let mut next_chiral: Option<(usize, Chirality)> = None;
        for i in (0..residues.len()).rev() {
            if residues[i].residue != CanonicalResidue::Gly {
                next_chiral = Some((i, residues[i].backbone_parity));
            } else if i < next_chiral.map_or(0, |(idx, _)| idx) {
                // Only needed for leading glycines before the first chiral residue.
                let has_prev = residues[..i]
                    .iter()
                    .any(|r| r.residue != CanonicalResidue::Gly);
                if !has_prev && let Some((anchor_idx, anchor_parity)) = next_chiral {
                    residues[i].backbone_parity = flip_parity(anchor_parity, anchor_idx - i);
                }
            }
        }
        if residues.iter().all(|r| r.residue == CanonicalResidue::Gly) {
            for (i, r) in residues.iter_mut().enumerate() {
                r.backbone_parity = if i % 2 == 0 {
                    Chirality::L
                } else {
                    Chirality::D
                };
            }
        }

        Ok(Self { residues })
    }

    /// Validates that consecutive residues strictly alternate L/D backbone parity.
    pub fn validate_alternating(&self) -> Result<(), InputError> {
        for w in self.residues.windows(2) {
            if w[0].backbone_parity == w[1].backbone_parity {
                return Err(InputError::NonAlternatingChirality {
                    seq_id: w[1].seq_id,
                });
            }
        }
        Ok(())
    }

    /// Returns the dASPR-compatible case-encoded sequence (`G` for Gly).
    pub fn to_daspr_string(&self) -> String {
        self.residues
            .iter()
            .map(|r| r.residue.case_char(r.chirality))
            .collect()
    }

    /// Converts all residues to poly-D/L-alanine for Phase 1 backbone modelling.
    pub fn as_poly_ala(&self, keep_gly: bool) -> Self {
        Self {
            residues: self.residues.iter().map(|r| r.to_poly_ala(keep_gly)).collect(),
        }
    }

    /// Validates that the sequence only contains Phase-1 backbone-supported residues (`Ala`/`Gly`).
    pub fn validate_phase1_backbone(&self) -> Result<(), InputError> {
        for r in &self.residues {
            if !matches!(r.residue, CanonicalResidue::Ala | CanonicalResidue::Gly) {
                return Err(InputError::UnsupportedPhase1Residue {
                    seq_id: r.seq_id,
                    code: r.pdb_name(),
                });
            }
        }
        Ok(())
    }

    /// Number of residues in the sequence.
    pub fn len(&self) -> usize {
        self.residues.len()
    }

    /// Returns `true` if the sequence has no residues.
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }
}

/// Crick-style parametric parameters for a single monomeric beta-helix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HelixParams {
    /// Cylinder radius (in Å) for L-parity Cα atoms.
    pub radius_l: f64,
    /// Cylinder radius (in Å) for D-parity Cα atoms.
    pub radius_d: f64,
    /// Base number of residues per full helical turn (e.g. ~6.2–6.4 for β6.3 helices).
    pub residues_per_turn: f64,
    /// Optional per-step variation in residues per turn (`len == n_res - 1`).
    pub residues_per_turn_profile: Option<Vec<f64>>,
    /// Axial rise (in Å) for steps starting at an L-parity residue.
    pub rise_ld: f64,
    /// Axial rise (in Å) for steps starting at a D-parity residue.
    pub rise_dl: f64,
    /// Peptide-plane C=O tilt angle (degrees) in the `(e_rad, e_perp)` frame for L-originating steps.
    pub tilt_ld_deg: f64,
    /// Peptide-plane C=O tilt angle (degrees) in the `(e_rad, e_perp)` frame for D-originating steps.
    pub tilt_dl_deg: f64,
    /// Helical handedness.
    pub handedness: Handedness,
    /// Initial azimuth offset (radians) at residue 0.
    pub phase_offset_rad: f64,
}

impl Default for HelixParams {
    fn default() -> Self {
        Self::beta_6_3()
    }
}

impl HelixParams {
    /// Generic right-handed β6.3-helix parameters.
    pub fn beta_6_3() -> Self {
        Self {
            radius_l: 3.85,
            radius_d: 4.03,
            residues_per_turn: 6.20,
            residues_per_turn_profile: None,
            rise_ld: 0.45,
            rise_dl: 1.02,
            tilt_ld_deg: 97.5,
            tilt_dl_deg: -84.5,
            handedness: Handedness::Right,
            phase_offset_rad: 0.0,
        }
    }

    /// Reference β6.3-helix parameters calibrated to Gramicidin A (`1NRM`, chain A).
    pub fn beta_6_3_gramicidin() -> Self {
        Self {
            radius_l: 3.77,
            radius_d: 4.11,
            residues_per_turn: 6.11,
            residues_per_turn_profile: None,
            rise_ld: 0.530,
            rise_dl: 0.889,
            tilt_ld_deg: 101.7,
            tilt_dl_deg: -82.6,
            handedness: Handedness::Right,
            phase_offset_rad: 0.0,
        }
    }

    /// Reference β6.3-helix parameters calibrated to Polytheonamide B (`2RQO`, chain A).
    pub fn beta_6_3_polytheonamide() -> Self {
        Self {
            radius_l: 3.93,
            radius_d: 3.96,
            residues_per_turn: 6.24,
            residues_per_turn_profile: None,
            rise_ld: 0.357,
            rise_dl: 1.152,
            tilt_ld_deg: 93.7,
            tilt_dl_deg: -86.5,
            handedness: Handedness::Right,
            phase_offset_rad: 0.0,
        }
    }

    /// Validates parameter ranges and optional per-step `residues_per_turn_profile` length.
    pub fn validate(&self, n_residues: usize) -> Result<(), InputError> {
        if n_residues < 2 {
            return Err(InputError::SequenceTooShort(n_residues));
        }
        if !self.radius_l.is_finite() || self.radius_l <= 0.0 {
            return Err(InputError::InvalidParameter("radius_l must be positive"));
        }
        if !self.radius_d.is_finite() || self.radius_d <= 0.0 {
            return Err(InputError::InvalidParameter("radius_d must be positive"));
        }
        if !self.residues_per_turn.is_finite() || self.residues_per_turn <= 2.0 {
            return Err(InputError::InvalidParameter(
                "residues_per_turn must be > 2.0",
            ));
        }
        if !self.rise_ld.is_finite() || !self.rise_dl.is_finite() {
            return Err(InputError::InvalidParameter("rise values must be finite"));
        }
        if let Some(profile) = &self.residues_per_turn_profile {
            if profile.len() != n_residues - 1 {
                return Err(InputError::InvalidProfileLength {
                    expected: n_residues - 1,
                    actual: profile.len(),
                });
            }
            if profile.iter().any(|&n| !n.is_finite() || n <= 2.0) {
                return Err(InputError::InvalidParameter(
                    "all residues_per_turn_profile entries must be > 2.0",
                ));
            }
        }
        Ok(())
    }

    /// Returns the number of residues per turn for step `step_idx` (`i -> i + 1`).
    pub fn step_residues_per_turn(&self, step_idx: usize) -> f64 {
        self.residues_per_turn_profile
            .as_ref()
            .and_then(|p| p.get(step_idx).copied())
            .unwrap_or(self.residues_per_turn)
    }
}

/// Axial direction of a helix within a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    /// N-to-C runs along +Z.
    #[default]
    Parallel,
    /// N-to-C runs along -Z.
    Antiparallel,
}

/// Controls whether minor-helix twist co-rotates with the supercoil Darboux frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwistMode {
    /// Minor twist increments in the local Darboux frame, keeping φ_Cα constant along the bundle.
    #[default]
    CoupledDarboux,
    /// Centerline follows the supercoil path while minor twist advances in a twist-free Bishop frame.
    UncoupledLabFrame,
}

/// Oligomeric symmetry mode for bundle generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
pub enum Symmetry {
    /// Explicit heteromeric list of helices with independent parameters.
    #[default]
    None,
    /// Cyclic C_n symmetry generating `n` parallel helices.
    Cn(usize),
    /// Dihedral D_n symmetry generating `2n` alternating parallel/antiparallel helices.
    Dn(usize),
}

impl FromStr for Symmetry {
    type Err = InputError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let s = raw.trim().to_ascii_lowercase();
        if s == "none" {
            return Ok(Self::None);
        }
        if let Some(rest) = s.strip_prefix('c') {
            let n: usize = rest
                .parse()
                .map_err(|_| InputError::InvalidParameter("invalid Cn symmetry order"))?;
            return Ok(Self::Cn(n));
        }
        if let Some(rest) = s.strip_prefix('d') {
            let n: usize = rest
                .parse()
                .map_err(|_| InputError::InvalidParameter("invalid Dn symmetry order"))?;
            return Ok(Self::Dn(n));
        }
        Err(InputError::InvalidParameter(
            "symmetry must be 'none', 'c<n>', or 'd<n>'",
        ))
    }
}

impl<'de> Deserialize<'de> for Symmetry {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum SymRepr {
            Str(String),
            Map {
                #[serde(alias = "cn")]
                #[serde(rename = "Cn")]
                cn: Option<usize>,
                #[serde(alias = "dn")]
                #[serde(rename = "Dn")]
                dn: Option<usize>,
            },
        }

        match SymRepr::deserialize(deserializer)? {
            SymRepr::Str(s) => s.parse().map_err(serde::de::Error::custom),
            SymRepr::Map {
                cn: Some(n),
                dn: None,
            } => Ok(Symmetry::Cn(n)),
            SymRepr::Map {
                cn: None,
                dn: Some(n),
            } => Ok(Symmetry::Dn(n)),
            SymRepr::Map { .. } => Err(serde::de::Error::custom(
                "symmetry map must specify exactly one of Cn or Dn",
            )),
        }
    }
}

/// Parametric placement specification for a single helix inside a bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HelixPlacementParams {
    /// Case-encoded sequence (uppercase = L, lowercase = D).
    pub sequence: String,
    /// Distance (in Å) from the bundle Z-axis to the helix centerline.
    pub bundle_radius: f64,
    /// Signed supercoil pitch (Å): >0 right-handed, <0 left-handed, 0 or None = straight.
    pub pitch: Option<f64>,
    /// Rotation around minor helix axis (degrees), with 0° pointing inward toward the bundle axis.
    pub phi_c_alpha_deg: f64,
    /// Longitudinal offset (Å) relative to helix 0 (antiparallel offset in Dn, per-helix in heteromers).
    pub z_offset: f64,
    /// Explicit azimuth (degrees) around bundle Z-axis (used in heteromers; ignored in Cn/Dn).
    pub azimuth_deg: Option<f64>,
    /// Axial orientation (parallel or antiparallel).
    pub orientation: Orientation,
    /// Coupling mode between supercoiling and minor residues per turn.
    pub twist_mode: TwistMode,
    /// Minor beta-helix geometry parameters.
    pub minor_params: HelixParams,
    /// Optional explicit PDB chain identifier.
    pub chain_id: Option<char>,
}

impl Default for HelixPlacementParams {
    fn default() -> Self {
        Self {
            sequence: "AaAaAaAaAaAaAaA".to_string(),
            bundle_radius: 7.5,
            pitch: None,
            phi_c_alpha_deg: 0.0,
            z_offset: 0.0,
            azimuth_deg: None,
            orientation: Orientation::Parallel,
            twist_mode: TwistMode::CoupledDarboux,
            minor_params: HelixParams::beta_6_3(),
            chain_id: None,
        }
    }
}

impl HelixPlacementParams {
    /// Validates finite placement values, sequence syntax, and minor helix parameters.
    pub fn validate(&self) -> Result<(), InputError> {
        let seq = Sequence::parse(&self.sequence)?;
        self.minor_params.validate(seq.len())?;
        if !self.bundle_radius.is_finite() || self.bundle_radius < 0.0 {
            return Err(InputError::InvalidParameter(
                "bundle_radius must be finite and non-negative",
            ));
        }
        if let Some(p) = self.pitch
            && !p.is_finite()
        {
            return Err(InputError::InvalidParameter("pitch must be finite"));
        }
        if !self.phi_c_alpha_deg.is_finite() || !self.z_offset.is_finite() {
            return Err(InputError::InvalidParameter(
                "phi_c_alpha_deg and z_offset must be finite",
            ));
        }
        if let Some(az) = self.azimuth_deg
            && !az.is_finite()
        {
            return Err(InputError::InvalidParameter("azimuth_deg must be finite"));
        }
        Ok(())
    }
}

/// Complete YAML/API specification for an oligomeric beta-helix bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BundleSpec {
    /// Symmetry group (`None` for heteromers, `Cn(n)`, or `Dn(n)`).
    pub symmetry: Symmetry,
    /// Keep glycine residues as GLY rather than replacing with D/L-Ala.
    pub keep_gly: bool,
    /// Run dASPR side-chain packing on the generated backbone.
    pub pack: bool,
    /// Template helix (for Cn/Dn) or explicit list of helices (for `Symmetry::None`).
    pub helices: Vec<HelixPlacementParams>,
}

impl Default for BundleSpec {
    fn default() -> Self {
        Self {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams::default()],
        }
    }
}

impl BundleSpec {
    /// Validates bundle symmetry order, helix count, and per-helix parameters.
    pub fn validate(&self) -> Result<(), InputError> {
        if self.helices.is_empty() {
            return Err(InputError::InvalidParameter(
                "bundle must specify at least one helix",
            ));
        }
        match self.symmetry {
            Symmetry::Cn(n) | Symmetry::Dn(n) => {
                if n == 0 {
                    return Err(InputError::InvalidParameter(
                        "symmetry order n must be >= 1",
                    ));
                }
                if self.helices.len() != 1 {
                    return Err(InputError::InvalidParameter(
                        "symmetric Cn/Dn bundle must specify exactly one template helix",
                    ));
                }
            }
            Symmetry::None => {}
        }
        for helix in &self.helices {
            helix.validate()?;
        }
        Ok(())
    }
}

fn default_chain_a() -> char {
    'A'
}

/// YAML-serializable monomer specification supporting variable twist profiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MonomerSpec {
    pub sequence: String,
    pub keep_gly: bool,
    pub pack: bool,
    #[serde(default = "default_chain_a")]
    pub chain_id: char,
    pub params: HelixParams,
}

impl Default for MonomerSpec {
    fn default() -> Self {
        Self {
            sequence: "AaAaAaAaAaAaAaA".to_string(),
            keep_gly: false,
            pack: false,
            chain_id: 'A',
            params: HelixParams::beta_6_3(),
        }
    }
}

/// Errors produced during input parsing or parameter validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    EmptySequence,
    SequenceTooShort(usize),
    LowercaseGlycine,
    NonAlternatingChirality { seq_id: usize },
    UnsupportedResidueChar(char),
    UnsupportedPhase1Residue { seq_id: usize, code: &'static str },
    InvalidParameter(&'static str),
    InvalidProfileLength { expected: usize, actual: usize },
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "sequence cannot be empty"),
            Self::SequenceTooShort(n) => {
                write!(f, "sequence must contain at least 2 residues, got {n}")
            }
            Self::LowercaseGlycine => {
                write!(f, "glycine is achiral and must be specified as uppercase 'G'")
            }
            Self::NonAlternatingChirality { seq_id } => {
                write!(f, "non-alternating chirality at residue {seq_id}")
            }
            Self::UnsupportedResidueChar(c) => {
                write!(f, "unsupported amino acid character '{c}'")
            }
            Self::UnsupportedPhase1Residue { seq_id, code } => write!(
                f,
                "residue {seq_id} ({code}) is not supported without --pack / poly-Ala conversion"
            ),
            Self::InvalidParameter(msg) => write!(f, "invalid helix parameter: {msg}"),
            Self::InvalidProfileLength { expected, actual } => write!(
                f,
                "residues_per_turn_profile length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for InputError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_case_chirality_and_pdb_codes() {
        let seq = Sequence::parse("VGAlAv").expect("valid sequence");
        assert_eq!(seq.len(), 6);
        assert_eq!(seq.residues[0].chirality, Chirality::L);
        assert_eq!(seq.residues[0].pdb_name(), "VAL");
        assert_eq!(seq.residues[1].chirality, Chirality::Achiral);
        assert_eq!(seq.residues[1].backbone_parity, Chirality::D);
        assert_eq!(seq.residues[1].pdb_name(), "GLY");
        assert_eq!(seq.residues[2].chirality, Chirality::L);
        assert_eq!(seq.residues[2].pdb_name(), "ALA");
        assert_eq!(seq.residues[3].chirality, Chirality::D);
        assert_eq!(seq.residues[3].pdb_name(), "DLE");
        assert_eq!(seq.to_daspr_string(), "VGAlAv");
        seq.validate_alternating().unwrap();

        let lead_gly = Sequence::parse("GlA").expect("leading Gly valid");
        assert_eq!(lead_gly.residues[0].backbone_parity, Chirality::L);
    }

    #[test]
    fn poly_ala_conversion_replaces_gly_by_parity() {
        let seq = Sequence::parse("AGA").unwrap().as_poly_ala(false);
        assert_eq!(seq.residues[1].residue, CanonicalResidue::Ala);
        assert_eq!(seq.residues[1].chirality, Chirality::D);
        assert_eq!(seq.residues[1].pdb_name(), "DAL");

        let keep_gly = Sequence::parse("AGA").unwrap().as_poly_ala(true);
        assert_eq!(keep_gly.residues[1].residue, CanonicalResidue::Gly);
        assert_eq!(keep_gly.residues[1].backbone_parity, Chirality::D);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(Sequence::parse(""), Err(InputError::EmptySequence));
        assert_eq!(Sequence::parse("AgA"), Err(InputError::LowercaseGlycine));
        assert_eq!(
            Sequence::parse("AXA"),
            Err(InputError::UnsupportedResidueChar('X'))
        );
        let non_alt = Sequence::parse("AAa").unwrap();
        assert_eq!(
            non_alt.validate_alternating(),
            Err(InputError::NonAlternatingChirality { seq_id: 2 })
        );
        let params = HelixParams::beta_6_3();
        assert_eq!(params.validate(1), Err(InputError::SequenceTooShort(1)));
    }

    #[test]
    fn parses_bundle_yaml_with_dn_and_heteromer() {
        let dn_yaml = r#"
symmetry: d2
helices:
  - sequence: "AaAaAaAa"
    bundle_radius: 7.5
    pitch: -120.0
    phi_c_alpha_deg: 15.0
    z_offset: 1.5
    twist_mode: uncoupled_lab_frame
"#;
        let spec: BundleSpec = serde_yml::from_str(dn_yaml).unwrap();
        assert_eq!(spec.symmetry, Symmetry::Dn(2));
        assert_eq!(spec.helices[0].pitch, Some(-120.0));
        assert_eq!(spec.helices[0].twist_mode, TwistMode::UncoupledLabFrame);
        spec.validate().unwrap();

        let het_yaml = r#"
symmetry: none
helices:
  - sequence: "AaAaAa"
    bundle_radius: 6.5
    orientation: parallel
  - sequence: "aAaAaA"
    bundle_radius: 7.0
    z_offset: 2.0
    orientation: antiparallel
"#;
        let het: BundleSpec = serde_yml::from_str(het_yaml).unwrap();
        assert_eq!(het.symmetry, Symmetry::None);
        assert_eq!(het.helices.len(), 2);
        het.validate().unwrap();
    }
}
