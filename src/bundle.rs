//! Oligomeric beta-helix bundle generation, supercoiled Crick trajectories, and symmetry.

use nalgebra::{Point3, Vector3};

use crate::geometry::{
    BuildError, Chain, CurveSample, build_helix_along_curve, step_params,
};
use crate::input::{
    BundleSpec, HelixPlacementParams, InputError, Orientation, Sequence, Symmetry,
};

/// Oligomeric assembly of beta-helix chains.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    pub chains: Vec<Chain>,
}

impl Bundle {
    /// Returns the minimum inter-chain Cα–Cα distance (Å) across all distinct chain pairs.
    pub fn min_interchain_ca_distance(&self) -> Option<f64> {
        let mut min_dist = f64::INFINITY;
        for i in 0..self.chains.len() {
            for j in (i + 1)..self.chains.len() {
                for a in &self.chains[i].residues {
                    for b in &self.chains[j].residues {
                        let d = (a.ca - b.ca).norm();
                        if d < min_dist {
                            min_dist = d;
                        }
                    }
                }
            }
        }
        if min_dist.is_finite() {
            Some(min_dist)
        } else {
            None
        }
    }
}

fn default_chain_id(index: usize) -> Result<char, BuildError> {
    const CHAIN_ALPHABET: &[u8] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";
    CHAIN_ALPHABET
        .get(index)
        .copied()
        .map(char::from)
        .ok_or(BuildError::Input(InputError::InvalidParameter(
            "bundle exceeds 62 chains without explicit unique chain IDs",
        )))
}

/// Computes local Darboux frame samples along a supercoiled (or straight) major helix.
///
/// The curve is centered so its axial midpoint sits at `z = 0` prior to applying `z_shift`.
/// `phi_c_alpha = 0` aligns residue 0 with `radial_in` (pointing toward the bundle Z-axis).
pub fn supercoil_samples(
    sequence: &Sequence,
    placement: &HelixPlacementParams,
    azimuth_rad: f64,
    z_shift: f64,
) -> (Vec<CurveSample>, Vec<f64>) {
    let n_res = sequence.len();
    let mut arc_positions = Vec::with_capacity(n_res);
    let mut s = 0.0;
    for (i, res) in sequence.residues.iter().enumerate() {
        arc_positions.push(s);
        if i + 1 < n_res {
            let (rise, _) = step_params(res.backbone_parity, &placement.minor_params);
            s += rise;
        }
    }
    let s_mid = arc_positions.iter().sum::<f64>() / (n_res as f64);

    let r_maj = placement.bundle_radius;
    let pitch = placement.pitch.unwrap_or(0.0);
    let (alpha, omega_maj) = if r_maj > 1e-12 && pitch.abs() > 1e-12 {
        let alpha = (2.0 * std::f64::consts::PI * r_maj / pitch).atan();
        (alpha, alpha.sin() / r_maj)
    } else {
        (0.0, 0.0)
    };
    let cos_a = alpha.cos();
    let sin_a = alpha.sin();

    let samples: Vec<CurveSample> = arc_positions
        .iter()
        .map(|&s_i| {
            let u = s_i - s_mid;
            let (theta, z, tangent) = match placement.orientation {
                Orientation::Parallel => {
                    let th = azimuth_rad + omega_maj * u;
                    let z = u * cos_a + z_shift;
                    let t = Vector3::new(-sin_a * th.sin(), sin_a * th.cos(), cos_a);
                    (th, z, t)
                }
                Orientation::Antiparallel => {
                    let th = azimuth_rad - omega_maj * u;
                    let z = -u * cos_a + z_shift;
                    let t = Vector3::new(sin_a * th.sin(), -sin_a * th.cos(), -cos_a);
                    (th, z, t)
                }
            };
            let center = Point3::new(r_maj * theta.cos(), r_maj * theta.sin(), z);
            let radial_in = Vector3::new(-theta.cos(), -theta.sin(), 0.0);
            let binormal = tangent.cross(&radial_in);
            CurveSample {
                center,
                tangent,
                radial_in,
                binormal,
            }
        })
        .collect();

    let dtheta_cos_alpha: Vec<f64> = arc_positions
        .windows(2)
        .map(|w| omega_maj * (w[1] - w[0]) * cos_a)
        .collect();

    (samples, dtheta_cos_alpha)
}

fn build_single_placed_helix(
    placement: &HelixPlacementParams,
    keep_gly: bool,
    azimuth_rad: f64,
    z_shift: f64,
    chain_id: char,
) -> Result<Chain, BuildError> {
    let seq = Sequence::parse(&placement.sequence)?.as_poly_ala(keep_gly);
    let (samples, dtheta_cos_alpha) = supercoil_samples(&seq, placement, azimuth_rad, z_shift);
    build_helix_along_curve(
        &seq,
        &placement.minor_params,
        &samples,
        placement.twist_mode,
        placement.phi_c_alpha_deg.to_radians(),
        &dtheta_cos_alpha,
        chain_id,
    )
}

/// Expands the per-chain target [`Sequence`]s for `spec` in the same chain order as [`build_bundle`].
pub fn bundle_target_sequences(spec: &BundleSpec) -> Result<Vec<Sequence>, BuildError> {
    spec.validate()?;
    match spec.symmetry {
        Symmetry::Cn(n) => {
            let seq = Sequence::parse(&spec.helices[0].sequence)?;
            Ok(vec![seq; n])
        }
        Symmetry::Dn(n) => {
            let seq = Sequence::parse(&spec.helices[0].sequence)?;
            Ok(vec![seq; 2 * n])
        }
        Symmetry::None => spec
            .helices
            .iter()
            .map(|h| Sequence::parse(&h.sequence).map_err(BuildError::from))
            .collect(),
    }
}

/// Generates an oligomeric beta-helix [`Bundle`] from a [`BundleSpec`].
///
/// - `Symmetry::Dn(n)`: expands 1 template helix into `2n` alternating parallel (`z = 0`) and
///   antiparallel (`z = z_offset`) helices at `Ω_k = π/(2n) + πk/n`.
/// - `Symmetry::Cn(n)`: expands 1 template helix into `n` parallel helices at `Ω_k = 2πk/n`.
/// - `Symmetry::None`: builds each heteromeric helix with longitudinal shift relative to helix 0.
pub fn build_bundle(spec: &BundleSpec) -> Result<Bundle, BuildError> {
    spec.validate()?;
    let keep_gly = spec.keep_gly || spec.pack;

    let mut chains = Vec::new();
    match spec.symmetry {
        Symmetry::Cn(n) => {
            let tpl = &spec.helices[0];
            let base_az = tpl.azimuth_deg.unwrap_or(0.0).to_radians();
            for k in 0..n {
                let az = base_az + 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                let chain_id = default_chain_id(k)?;
                chains.push(build_single_placed_helix(tpl, keep_gly, az, 0.0, chain_id)?);
            }
        }
        Symmetry::Dn(n) => {
            let tpl = &spec.helices[0];
            let base_az = tpl
                .azimuth_deg
                .map(f64::to_radians)
                .unwrap_or(std::f64::consts::PI / (2.0 * n as f64));
            let total_helices = 2 * n;
            for k in 0..total_helices {
                let mut helix_params = tpl.clone();
                let is_parallel = k % 2 == 0;
                helix_params.orientation = if is_parallel {
                    Orientation::Parallel
                } else {
                    Orientation::Antiparallel
                };
                let z_shift = if is_parallel { 0.0 } else { tpl.z_offset };
                let az = base_az + std::f64::consts::PI * (k as f64) / (n as f64);
                let chain_id = default_chain_id(k)?;
                chains.push(build_single_placed_helix(
                    &helix_params,
                    keep_gly,
                    az,
                    z_shift,
                    chain_id,
                )?);
            }
        }
        Symmetry::None => {
            let n_helices = spec.helices.len();
            let z_ref = spec.helices[0].z_offset;
            for (k, helix) in spec.helices.iter().enumerate() {
                let az = helix
                    .azimuth_deg
                    .map(f64::to_radians)
                    .unwrap_or(2.0 * std::f64::consts::PI * (k as f64) / (n_helices as f64));
                let z_shift = helix.z_offset - z_ref;
                let chain_id = match helix.chain_id {
                    Some(id) => id,
                    None => default_chain_id(k)?,
                };
                chains.push(build_single_placed_helix(
                    helix, keep_gly, az, z_shift, chain_id,
                )?);
            }
        }
    }

    Ok(Bundle { chains })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{
        backbone_dihedrals, bond_length, build_beta_helix, improper_chirality_deg, kabsch_rmsd,
    };
    use crate::input::{HelixParams, TwistMode};

    fn chain_mean_z(chain: &Chain) -> f64 {
        chain.residues.iter().map(|r| r.ca.z).sum::<f64>() / (chain.residues.len() as f64)
    }

    fn all_ca(chain: &Chain) -> Vec<Point3<f64>> {
        chain.residues.iter().map(|r| r.ca).collect()
    }

    #[test]
    fn straight_zero_radius_bundle_matches_monomer_up_to_centering_and_phi() {
        let seq_str = "AaAaAaAaAaAaAaA";
        let seq = Sequence::parse(seq_str).unwrap();
        let monomer = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();

        let spec = BundleSpec {
            symmetry: Symmetry::None,
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: seq_str.to_string(),
                bundle_radius: 0.0,
                pitch: None,
                phi_c_alpha_deg: 0.0,
                z_offset: 0.0,
                azimuth_deg: Some(0.0),
                orientation: Orientation::Parallel,
                twist_mode: TwistMode::CoupledDarboux,
                minor_params: HelixParams::beta_6_3(),
                chain_id: Some('A'),
            }],
        };
        let bundle = build_bundle(&spec).unwrap();
        let rmsd = kabsch_rmsd(&all_ca(&monomer), &all_ca(&bundle.chains[0])).unwrap();
        assert!(rmsd < 1e-10, "expected exact match, got {rmsd}");
    }

    #[test]
    fn dn_symmetry_satisfies_exact_cn_and_c2_point_group() {
        let spec = BundleSpec {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: "AaAaAaAaAaAaAaA".to_string(),
                bundle_radius: 8.5,
                pitch: Some(140.0),
                phi_c_alpha_deg: 25.0,
                z_offset: 0.0,
                ..Default::default()
            }],
        };
        let bundle = build_bundle(&spec).unwrap();
        assert_eq!(bundle.chains.len(), 4);

        // Verify C2(X): (x, y, z) -> (x, -y, -z) maps chain 0 (45°) to chain 3 (-45°).
        let c0_flipped: Vec<Point3<f64>> = bundle.chains[0]
            .residues
            .iter()
            .map(|r| Point3::new(r.ca.x, -r.ca.y, -r.ca.z))
            .collect();
        let c3 = all_ca(&bundle.chains[3]);
        let max_err = c0_flipped
            .iter()
            .zip(c3.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-10,
            "D2 C2(X) symmetry error too large: {max_err}"
        );

        // Verify C2(Z): (x, y, z) -> (-x, -y, z) maps chain 0 to chain 2.
        let c0_rot_z: Vec<Point3<f64>> = bundle.chains[0]
            .residues
            .iter()
            .map(|r| Point3::new(-r.ca.x, -r.ca.y, r.ca.z))
            .collect();
        let c2 = all_ca(&bundle.chains[2]);
        let max_z_err = c0_rot_z
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_z_err < 1e-10,
            "D2 C2(Z) symmetry error too large: {max_z_err}"
        );
    }

    #[test]
    fn z_offset_shifts_antiparallel_relative_to_first_helix() {
        let helix = HelixPlacementParams {
            sequence: "AaAaAaAaAaAa".to_string(),
            bundle_radius: 8.5,
            pitch: Some(150.0),
            z_offset: 3.5,
            ..Default::default()
        };
        let seq = Sequence::parse(&helix.sequence).unwrap();
        let (s0, _) = supercoil_samples(&seq, &helix, 0.0, 0.0);
        let mut anti = helix.clone();
        anti.orientation = Orientation::Antiparallel;
        let (s1, _) = supercoil_samples(&seq, &anti, 0.0, helix.z_offset);

        let mean_c0 = s0.iter().map(|s| s.center.z).sum::<f64>() / (s0.len() as f64);
        let mean_c1 = s1.iter().map(|s| s.center.z).sum::<f64>() / (s1.len() as f64);
        assert!(mean_c0.abs() < 1e-10, "helix 0 centerline must sit at z=0: {mean_c0}");
        assert!(
            (mean_c1 - 3.5).abs() < 1e-10,
            "antiparallel centerline must shift by exact 3.5 Å: {mean_c1}"
        );

        let bundle = build_bundle(&BundleSpec {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![helix],
        })
        .unwrap();
        let z0 = chain_mean_z(&bundle.chains[0]);
        let z1 = chain_mean_z(&bundle.chains[1]);
        let z2 = chain_mean_z(&bundle.chains[2]);
        assert!((z0 - z2).abs() < 1e-10);
        assert!((z1 - z0 - 3.5).abs() < 0.1, "Cα centroid shift={}", z1 - z0);
    }

    #[test]
    fn coupled_darboux_preserves_phi_ca_while_uncoupled_precesses() {
        let base_helix = HelixPlacementParams {
            sequence: "AaAaAaAaAaAaAaAaAaAaAaAa".to_string(),
            bundle_radius: 9.0,
            pitch: Some(80.0),
            phi_c_alpha_deg: 0.0,
            twist_mode: TwistMode::CoupledDarboux,
            ..Default::default()
        };

        let coupled = build_bundle(&BundleSpec {
            symmetry: Symmetry::None,
            keep_gly: false,
            pack: false,
            helices: vec![base_helix.clone()],
        })
        .unwrap();

        let mut uncoupled_helix = base_helix.clone();
        uncoupled_helix.twist_mode = TwistMode::UncoupledLabFrame;
        let uncoupled = build_bundle(&BundleSpec {
            symmetry: Symmetry::None,
            keep_gly: false,
            pack: false,
            helices: vec![uncoupled_helix],
        })
        .unwrap();

        let seq = Sequence::parse(&base_helix.sequence).unwrap();
        let (samples, _) = supercoil_samples(&seq, &base_helix, 0.0, 0.0);

        let local_angle = |chain: &Chain, idx: usize| -> f64 {
            let offset = chain.residues[idx].ca - samples[idx].center;
            let y = offset.dot(&samples[idx].binormal);
            let x = offset.dot(&samples[idx].radial_in);
            y.atan2(x)
        };

        let last = seq.len() - 1;
        let expected_coupled = (base_helix.minor_params.handedness.sign()
            * 2.0
            * std::f64::consts::PI
            * (last as f64)
            / base_helix.minor_params.residues_per_turn)
            .rem_euclid(2.0 * std::f64::consts::PI);
        let obs_coupled = local_angle(&coupled.chains[0], last).rem_euclid(2.0 * std::f64::consts::PI);
        let obs_uncoupled =
            local_angle(&uncoupled.chains[0], last).rem_euclid(2.0 * std::f64::consts::PI);

        let diff_coupled = (obs_coupled - expected_coupled).abs();
        assert!(
            diff_coupled < 1e-10 || (2.0 * std::f64::consts::PI - diff_coupled) < 1e-10,
            "CoupledDarboux deviated from constant φ_Cα: obs={obs_coupled}, exp={expected_coupled}"
        );
        assert!(
            (obs_uncoupled - obs_coupled).abs() > 0.1,
            "UncoupledLabFrame must precess relative to CoupledDarboux"
        );

        // Verify exact bond geometry and chirality under tight supercoiling.
        for chain in [&coupled.chains[0], &uncoupled.chains[0]] {
            let dihedrals = backbone_dihedrals(chain);
            for (i, res) in chain.residues.iter().enumerate() {
                assert!((bond_length(res.n, res.ca) - 1.457).abs() < 1e-6);
                assert!((bond_length(res.ca, res.c) - 1.528).abs() < 1e-6);
                assert!(improper_chirality_deg(res).is_some());
                if let Some(om) = dihedrals[i].2 {
                    assert!(om.abs() > 179.9, "expected trans omega, got {om}");
                }
            }
        }
    }
}
