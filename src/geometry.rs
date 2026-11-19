//! Crick-style beta-helix backbone geometry, atom placement, and structural metrics.

use std::fmt;

use nalgebra::{Matrix3, Point3, Vector2, Vector3};

use crate::input::{
    CanonicalResidue, Chirality, HelixParams, InputError, Sequence, TwistMode,
};

const BOND_CA_C: f64 = 1.528;
const BOND_C_N: f64 = 1.332;
const BOND_N_CA: f64 = 1.457;
const BOND_C_O: f64 = 1.233;
const BOND_CA_CB: f64 = 1.521;

const ANGLE_CA_C_N_DEG: f64 = 117.3;
const ANGLE_C_N_CA_DEG: f64 = 124.2;
const ANGLE_CA_C_O_DEG: f64 = 120.8;
const ANGLE_N_CA_CB_DEG: f64 = 110.3;

const EPS: f64 = 1e-12;

/// Backbone and Cβ atomic coordinates for one residue.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueAtoms {
    pub seq_id: usize,
    pub res_name: &'static str,
    pub chirality: Chirality,
    pub n: Point3<f64>,
    pub ca: Point3<f64>,
    pub c: Point3<f64>,
    pub o: Point3<f64>,
    pub cb: Option<Point3<f64>>,
}

/// Generated polypeptide chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub chain_id: char,
    pub residues: Vec<ResidueAtoms>,
}

/// Errors produced during structure generation or RMSD evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    Input(InputError),
    DegenerateGeometry(&'static str),
    LengthMismatch { model: usize, reference: usize },
}

impl From<InputError> for BuildError {
    fn from(err: InputError) -> Self {
        Self::Input(err)
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(err) => write!(f, "{err}"),
            Self::DegenerateGeometry(msg) => write!(f, "degenerate geometry: {msg}"),
            Self::LengthMismatch { model, reference } => write!(
                f,
                "coordinate count mismatch: model has {model}, reference has {reference}"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Trans peptide unit in a 2D virtual-bond frame where:
/// - CA(i) is at (0, 0)
/// - CA(i+1) is at (chord_len, 0)
/// - C(i)=O(i) points toward +y
struct IdealPeptideUnit {
    c: Vector2<f64>,
    o: Vector2<f64>,
    n_next: Vector2<f64>,
}

fn trans_peptide_for_angle_offset(
    delta_rad: f64,
) -> (f64, Vector2<f64>, Vector2<f64>, Vector2<f64>) {
    let ang_cacn = ANGLE_CA_C_N_DEG.to_radians() + delta_rad;
    let ang_cnca = ANGLE_C_N_CA_DEG.to_radians() + delta_rad;
    let ang_caco = ANGLE_CA_C_O_DEG.to_radians();

    let p_c = Vector2::new(BOND_CA_C, 0.0);
    let th_cn = std::f64::consts::PI - ang_cacn;
    let p_n = p_c + Vector2::new(BOND_C_N * th_cn.cos(), BOND_C_N * th_cn.sin());
    let th_nca = th_cn - (std::f64::consts::PI - ang_cnca);
    let p_ca1 = p_n + Vector2::new(BOND_N_CA * th_nca.cos(), BOND_N_CA * th_nca.sin());

    let th_co = -(std::f64::consts::PI - ang_caco);
    let p_o = p_c + Vector2::new(BOND_C_O * th_co.cos(), BOND_C_O * th_co.sin());

    let chord = p_ca1.norm();
    let rot = -p_ca1.y.atan2(p_ca1.x);
    let (sin_r, cos_r) = rot.sin_cos();
    let rotate_and_flip_y = |v: Vector2<f64>| -> Vector2<f64> {
        let rx = v.x * cos_r - v.y * sin_r;
        let ry = v.x * sin_r + v.y * cos_r;
        // Negate y so C=O points toward +y.
        Vector2::new(rx, -ry)
    };

    (
        chord,
        rotate_and_flip_y(p_c),
        rotate_and_flip_y(p_o),
        rotate_and_flip_y(p_n),
    )
}

impl IdealPeptideUnit {
    /// Solves peptide-bond angle offset `delta` by bisection so all bond lengths remain exact
    /// while matching `target_chord` between consecutive Cα atoms.
    fn for_chord_length(target_chord: f64) -> Result<Self, BuildError> {
        let mut lo = -35.0_f64.to_radians();
        let mut hi = 35.0_f64.to_radians();
        let l_min = trans_peptide_for_angle_offset(lo).0;
        let l_max = trans_peptide_for_angle_offset(hi).0;
        if target_chord < l_min || target_chord > l_max {
            return Err(BuildError::DegenerateGeometry(
                "Cα-Cα distance outside physical trans peptide range",
            ));
        }
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            if trans_peptide_for_angle_offset(mid).0 < target_chord {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (_, c, o, n_next) = trans_peptide_for_angle_offset(0.5 * (lo + hi));
        Ok(Self { c, o, n_next })
    }
}

pub(crate) fn step_params(parity: Chirality, params: &HelixParams) -> (f64, f64) {
    match parity {
        Chirality::L | Chirality::Achiral => (params.rise_ld, params.tilt_ld_deg),
        Chirality::D => (params.rise_dl, params.tilt_dl_deg),
    }
}

fn radius_for_parity(parity: Chirality, params: &HelixParams) -> f64 {
    match parity {
        Chirality::L | Chirality::Achiral => params.radius_l,
        Chirality::D => params.radius_d,
    }
}

fn opposite_parity(parity: Chirality) -> Chirality {
    match parity {
        Chirality::L | Chirality::Achiral => Chirality::D,
        Chirality::D => Chirality::L,
    }
}

/// Local Darboux frame sample along a 3D helix centerline curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSample {
    /// Centerline position (Å).
    pub center: Point3<f64>,
    /// Unit tangent pointing in the N-to-C progression direction.
    pub tangent: Vector3<f64>,
    /// Reference unit radial vector in the normal plane (points inward toward bundle axis).
    pub radial_in: Vector3<f64>,
    /// Unit binormal completing the right-handed frame (`tangent × radial_in`).
    pub binormal: Vector3<f64>,
}

struct PeptidePlaneAtoms {
    c: Point3<f64>,
    o: Point3<f64>,
    n_next: Point3<f64>,
}

fn place_peptide_plane(
    ca_i: Point3<f64>,
    ca_next: Point3<f64>,
    center_mid: Point3<f64>,
    tilt_deg: f64,
) -> Result<PeptidePlaneAtoms, BuildError> {
    let chord = ca_next - ca_i;
    let chord_len = chord.norm();
    if chord_len < 0.5 {
        return Err(BuildError::DegenerateGeometry(
            "consecutive Cα atoms are too close",
        ));
    }
    let unit_plane = IdealPeptideUnit::for_chord_length(chord_len)?;
    let ex = chord / chord_len;

    let ca_mid = Point3::from(0.5 * (ca_i.coords + ca_next.coords));
    let radial_vec = ca_mid - center_mid;
    let r_perp = radial_vec - ex * radial_vec.dot(&ex);
    let r_norm = r_perp.norm();
    if r_norm < EPS {
        return Err(BuildError::DegenerateGeometry(
            "radial vector is collinear with Cα-Cα chord",
        ));
    }
    let e_rad = r_perp / r_norm;
    let e_perp = ex.cross(&e_rad);

    let tilt = tilt_deg.to_radians();
    let ey = e_rad * tilt.cos() + e_perp * tilt.sin();

    let map_point = |p: Vector2<f64>| -> Point3<f64> { ca_i + ex * p.x + ey * p.y };

    Ok(PeptidePlaneAtoms {
        c: map_point(unit_plane.c),
        o: map_point(unit_plane.o),
        n_next: map_point(unit_plane.n_next),
    })
}

fn place_cb(
    n: Point3<f64>,
    ca: Point3<f64>,
    c: Point3<f64>,
    chirality: Chirality,
) -> Result<Option<Point3<f64>>, BuildError> {
    let sign = match chirality {
        Chirality::L => 1.0,
        Chirality::D => -1.0,
        Chirality::Achiral => return Ok(None),
    };
    let u = (n - ca).normalize();
    let v = (c - ca).normalize();
    let b_sum = u + v;
    let b_norm = b_sum.norm();
    let w_cross = u.cross(&v);
    let w_norm = w_cross.norm();
    if b_norm < EPS || w_norm < EPS {
        return Err(BuildError::DegenerateGeometry(
            "collinear N-Cα-C bond vectors",
        ));
    }
    let b = b_sum / b_norm;
    let w = w_cross / w_norm;

    let cos_gamma = u.dot(&b).clamp(0.1, 0.999);
    let cos_alpha = (-ANGLE_N_CA_CB_DEG.to_radians().cos() / cos_gamma).clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();
    let cb_dir = -b * cos_alpha + w * (sign * sin_alpha);

    Ok(Some(ca + cb_dir * BOND_CA_CB))
}

/// Generates a Crick-style beta-helix backbone along an arbitrary 3D centerline curve.
///
/// - `samples`: local Darboux frame `(center, tangent, radial_in, binormal)` at each residue.
/// - `twist_mode`: selects `CoupledDarboux` (constant `phi_c_alpha` relative to `radial_in`)
///   or `UncoupledLabFrame` (subtracts geometric Darboux twist `dtheta_maj_cos_alpha_steps`).
pub fn build_helix_along_curve(
    sequence: &Sequence,
    params: &HelixParams,
    samples: &[CurveSample],
    twist_mode: TwistMode,
    phi_c_alpha_rad: f64,
    dtheta_maj_cos_alpha_steps: &[f64],
    chain_id: char,
) -> Result<Chain, BuildError> {
    sequence.validate_phase1_backbone()?;
    params.validate(sequence.len())?;

    let n_res = sequence.len();
    if samples.len() != n_res {
        return Err(BuildError::LengthMismatch {
            model: samples.len(),
            reference: n_res,
        });
    }
    if dtheta_maj_cos_alpha_steps.len() != n_res - 1 {
        return Err(BuildError::LengthMismatch {
            model: dtheta_maj_cos_alpha_steps.len(),
            reference: n_res - 1,
        });
    }

    let hand_sign = params.handedness.sign();
    let mut cas = Vec::with_capacity(n_res);
    let mut psis = Vec::with_capacity(n_res);

    let mut psi = phi_c_alpha_rad;
    for (i, (res, sample)) in sequence.residues.iter().zip(samples.iter()).enumerate() {
        let r = radius_for_parity(res.backbone_parity, params);
        let offset = sample.radial_in * (r * psi.cos()) + sample.binormal * (r * psi.sin());
        cas.push(sample.center + offset);
        psis.push(psi);

        if i + 1 < n_res {
            let n_turn = params.step_residues_per_turn(i);
            let dar_lag = match twist_mode {
                TwistMode::CoupledDarboux => 0.0,
                TwistMode::UncoupledLabFrame => dtheta_maj_cos_alpha_steps[i],
            };
            psi += hand_sign * 2.0 * std::f64::consts::PI / n_turn - dar_lag;
        }
    }

    let mut ns = vec![Point3::origin(); n_res];
    let mut cs = vec![Point3::origin(); n_res];
    let mut os = vec![Point3::origin(); n_res];

    for i in 0..(n_res - 1) {
        let (_, tilt_deg) = step_params(sequence.residues[i].backbone_parity, params);
        let center_mid = Point3::from(0.5 * (samples[i].center.coords + samples[i + 1].center.coords));
        let plane = place_peptide_plane(cas[i], cas[i + 1], center_mid, tilt_deg)?;
        cs[i] = plane.c;
        os[i] = plane.o;
        ns[i + 1] = plane.n_next;
    }

    // Virtual extension for terminal N(0) and C(n-1), O(n-1).
    let prev_parity = opposite_parity(sequence.residues[0].backbone_parity);
    let (prev_rise, prev_tilt) = step_params(prev_parity, params);
    let prev_r = radius_for_parity(prev_parity, params);
    let prev_lag = match twist_mode {
        TwistMode::CoupledDarboux => 0.0,
        TwistMode::UncoupledLabFrame => dtheta_maj_cos_alpha_steps[0],
    };
    let prev_psi =
        psis[0] - (hand_sign * 2.0 * std::f64::consts::PI / params.step_residues_per_turn(0) - prev_lag);
    let center_prev = samples[0].center - samples[0].tangent * prev_rise;
    let ca_prev = center_prev
        + samples[0].radial_in * (prev_r * prev_psi.cos())
        + samples[0].binormal * (prev_r * prev_psi.sin());
    let center_mid_prev = Point3::from(0.5 * (center_prev.coords + samples[0].center.coords));
    ns[0] = place_peptide_plane(ca_prev, cas[0], center_mid_prev, prev_tilt)?.n_next;

    let last_parity = sequence.residues[n_res - 1].backbone_parity;
    let next_parity = opposite_parity(last_parity);
    let (last_rise, last_tilt) = step_params(last_parity, params);
    let next_r = radius_for_parity(next_parity, params);
    let next_lag = match twist_mode {
        TwistMode::CoupledDarboux => 0.0,
        TwistMode::UncoupledLabFrame => dtheta_maj_cos_alpha_steps[n_res - 2],
    };
    let next_psi = psis[n_res - 1]
        + (hand_sign * 2.0 * std::f64::consts::PI / params.step_residues_per_turn(n_res - 2)
            - next_lag);
    let center_next = samples[n_res - 1].center + samples[n_res - 1].tangent * last_rise;
    let ca_next = center_next
        + samples[n_res - 1].radial_in * (next_r * next_psi.cos())
        + samples[n_res - 1].binormal * (next_r * next_psi.sin());
    let center_mid_next = Point3::from(0.5 * (samples[n_res - 1].center.coords + center_next.coords));
    let last_plane = place_peptide_plane(cas[n_res - 1], ca_next, center_mid_next, last_tilt)?;
    cs[n_res - 1] = last_plane.c;
    os[n_res - 1] = last_plane.o;

    let mut residues = Vec::with_capacity(n_res);
    for i in 0..n_res {
        let spec = &sequence.residues[i];
        let cb = if spec.residue == CanonicalResidue::Gly {
            None
        } else {
            place_cb(ns[i], cas[i], cs[i], spec.chirality)?
        };
        residues.push(ResidueAtoms {
            seq_id: spec.seq_id,
            res_name: spec.pdb_name(),
            chirality: spec.chirality,
            n: ns[i],
            ca: cas[i],
            c: cs[i],
            o: os[i],
            cb,
        });
    }

    Ok(Chain { chain_id, residues })
}

/// Generates a Crick-style parametric beta-helix backbone for `sequence` and `params`.
///
/// Residues are built along the +Z axis with Cα coordinates placed on alternating cylindrical
/// radii `(radius_l, radius_d)` and variable per-step twist `±2π / residues_per_turn[i]`.
pub fn build_beta_helix(
    sequence: &Sequence,
    params: &HelixParams,
    chain_id: char,
) -> Result<Chain, BuildError> {
    sequence.validate_phase1_backbone()?;
    params.validate(sequence.len())?;

    let n_res = sequence.len();
    let mut z = 0.0;
    let mut samples = Vec::with_capacity(n_res);
    for (i, res) in sequence.residues.iter().enumerate() {
        samples.push(CurveSample {
            center: Point3::new(0.0, 0.0, z),
            tangent: Vector3::z(),
            radial_in: Vector3::x(),
            binormal: Vector3::y(),
        });
        if i + 1 < n_res {
            let (rise, _) = step_params(res.backbone_parity, params);
            z += rise;
        }
    }
    let zeros = vec![0.0; n_res - 1];
    build_helix_along_curve(
        sequence,
        params,
        &samples,
        TwistMode::CoupledDarboux,
        params.phase_offset_rad,
        &zeros,
        chain_id,
    )
}

/// Computes Euclidean distance between two 3D points (Å).
pub fn bond_length(a: Point3<f64>, b: Point3<f64>) -> f64 {
    (b - a).norm()
}

/// Computes bond angle `a-b-c` in degrees.
pub fn bond_angle_deg(a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> f64 {
    let u = (a - b).normalize();
    let v = (c - b).normalize();
    u.dot(&v).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Computes signed dihedral angle `p0-p1-p2-p3` in degrees (`[-180, 180]`).
pub fn dihedral_deg(
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    p3: Point3<f64>,
) -> f64 {
    let b0 = p0 - p1;
    let b1 = (p2 - p1).normalize();
    let b2 = p3 - p2;
    let v = b0 - b1 * b0.dot(&b1);
    let w = b2 - b1 * b2.dot(&b1);
    let x = v.dot(&w);
    let y = b1.cross(&v).dot(&w);
    y.atan2(x).to_degrees()
}

/// Returns per-residue `(phi, psi, omega)` torsion angles in degrees.
pub fn backbone_dihedrals(chain: &Chain) -> Vec<(Option<f64>, Option<f64>, Option<f64>)> {
    let n = chain.residues.len();
    (0..n)
        .map(|i| {
            let cur = &chain.residues[i];
            let phi = if i > 0 {
                let prev = &chain.residues[i - 1];
                Some(dihedral_deg(prev.c, cur.n, cur.ca, cur.c))
            } else {
                None
            };
            let psi = if i + 1 < n {
                let next = &chain.residues[i + 1];
                Some(dihedral_deg(cur.n, cur.ca, cur.c, next.n))
            } else {
                None
            };
            let omega = if i > 0 {
                let prev = &chain.residues[i - 1];
                Some(dihedral_deg(prev.ca, prev.c, cur.n, cur.ca))
            } else {
                None
            };
            (phi, psi, omega)
        })
        .collect()
}

/// Computes the signed improper dihedral `(N, C, Cα, Cβ)` in degrees (~+122° L, ~-122° D).
pub fn improper_chirality_deg(res: &ResidueAtoms) -> Option<f64> {
    res.cb
        .map(|cb| dihedral_deg(res.n, res.c, res.ca, cb))
}

/// Computes optimal Kabsch root-mean-square deviation (Å) between equal-length point sets.
pub fn kabsch_rmsd(model: &[Point3<f64>], reference: &[Point3<f64>]) -> Result<f64, BuildError> {
    if model.len() != reference.len() {
        return Err(BuildError::LengthMismatch {
            model: model.len(),
            reference: reference.len(),
        });
    }
    if model.is_empty() {
        return Err(BuildError::DegenerateGeometry("empty point set for RMSD"));
    }

    let n = model.len() as f64;
    let mean_m = model
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + p.coords)
        / n;
    let mean_r = reference
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + p.coords)
        / n;

    let mut cov = Matrix3::<f64>::zeros();
    for (pm, pr) in model.iter().zip(reference.iter()) {
        let cm = pm.coords - mean_m;
        let cr = pr.coords - mean_r;
        cov += cm * cr.transpose();
    }

    let svd = cov.svd(true, true);
    let u = svd
        .u
        .ok_or(BuildError::DegenerateGeometry("SVD failed to compute U"))?;
    let v_t = svd
        .v_t
        .ok_or(BuildError::DegenerateGeometry("SVD failed to compute V^T"))?;

    let mut diag = Matrix3::identity();
    if (v_t.transpose() * u.transpose()).determinant() < 0.0 {
        diag[(2, 2)] = -1.0;
    }
    let rot = v_t.transpose() * diag * u.transpose();

    let sq_sum: f64 = model
        .iter()
        .zip(reference.iter())
        .map(|(pm, pr)| {
            let aligned = rot * (pm.coords - mean_m);
            (aligned - (pr.coords - mean_r)).norm_squared()
        })
        .sum();

    Ok((sq_sum / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_alternating_ld_geometry_and_chirality() {
        let seq = Sequence::parse("AaAaAaAaAaAaAaA").unwrap();
        let params = HelixParams::beta_6_3_gramicidin();
        let chain = build_beta_helix(&seq, &params, 'A').unwrap();
        assert_eq!(chain.residues.len(), 15);

        let dihedrals = backbone_dihedrals(&chain);
        for (i, res) in chain.residues.iter().enumerate() {
            let imp = improper_chirality_deg(res).expect("Ala must have CB");
            if i % 2 == 0 {
                assert_eq!(res.res_name, "ALA");
                assert!(imp > 115.0 && imp < 130.0, "expected L improper, got {imp}");
            } else {
                assert_eq!(res.res_name, "DAL");
                assert!(
                    imp < -115.0 && imp > -130.0,
                    "expected D improper, got {imp}"
                );
            }
            assert!((bond_length(res.n, res.ca) - BOND_N_CA).abs() < 1e-6);
            assert!((bond_length(res.ca, res.c) - BOND_CA_C).abs() < 1e-6);
            assert!((bond_length(res.c, res.o) - BOND_C_O).abs() < 1e-6);
            assert!((bond_length(res.ca, res.cb.unwrap()) - BOND_CA_CB).abs() < 1e-6);

            if let Some(om) = dihedrals[i].2 {
                assert!(om.abs() > 179.9, "expected trans omega, got {om}");
            }
        }
    }

    #[test]
    fn left_handed_helix_mirrors_twist_and_keeps_outward_cb() {
        let seq = Sequence::parse("AaAaAaAa").unwrap();
        let mut params = HelixParams::beta_6_3();
        params.handedness = crate::input::Handedness::Left;
        let chain = build_beta_helix(&seq, &params, 'A').unwrap();

        let p0 = chain.residues[0].ca;
        let p1 = chain.residues[1].ca;
        let th0 = p0.y.atan2(p0.x);
        let th1 = p1.y.atan2(p1.x);
        let mut dth = th1 - th0;
        if dth > std::f64::consts::PI {
            dth -= 2.0 * std::f64::consts::PI;
        }
        assert!(dth < 0.0, "left-handed helix must have negative Δθ");

        for res in &chain.residues {
            let ca_r = (res.ca.x.powi(2) + res.ca.y.powi(2)).sqrt();
            let cb = res.cb.unwrap();
            let cb_r = (cb.x.powi(2) + cb.y.powi(2)).sqrt();
            assert!(cb_r > ca_r, "Cβ must point radially outward");
        }
    }

    #[test]
    fn kabsch_rmsd_zero_under_rigid_transform() {
        let seq = Sequence::parse("AaAaAa").unwrap();
        let chain = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();
        let pts: Vec<_> = chain.residues.iter().map(|r| r.ca).collect();
        let rot = nalgebra::Rotation3::from_euler_angles(0.4, -0.7, 1.1);
        let shift = Vector3::new(12.0, -5.0, 3.3);
        let moved: Vec<_> = pts.iter().map(|&p| rot * p + shift).collect();
        let rmsd = kabsch_rmsd(&pts, &moved).unwrap();
        assert!(rmsd < 1e-10, "expected ~0 RMSD, got {rmsd}");
    }
}
