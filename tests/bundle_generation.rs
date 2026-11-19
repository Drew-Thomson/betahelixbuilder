//! Integration tests for oligomeric beta-helix bundles, Dn symmetry, and heteromeric YAML specs.

use betahelixbuilder::{
    BundleSpec, Symmetry, backbone_dihedrals, bond_length, build_bundle, bundle_to_pdb_string,
    improper_chirality_deg,
};
use nalgebra::Point3;

#[test]
fn d3_left_handed_supercoil_bundle_preserves_symmetry_and_geometry() {
    let yaml = r#"
symmetry: d3
helices:
  - sequence: "AaAaAaAaAaAaAaAaAaAa"
    bundle_radius: 14.0
    pitch: -160.0
    phi_c_alpha_deg: 35.0
    z_offset: 0.0
    twist_mode: coupled_darboux
"#;
    let spec: BundleSpec = serde_yml::from_str(yaml).unwrap();
    assert_eq!(spec.symmetry, Symmetry::Dn(3));
    let bundle = build_bundle(&spec).unwrap();
    assert_eq!(bundle.chains.len(), 6);

    // Check C3(Z) rotation (120°) maps chain 0 -> chain 2.
    let angle = 2.0 * std::f64::consts::PI / 3.0;
    let (s, c) = angle.sin_cos();
    let max_c3_err = bundle.chains[0]
        .residues
        .iter()
        .zip(bundle.chains[2].residues.iter())
        .map(|(r0, r2)| {
            let rot = Point3::new(c * r0.ca.x - s * r0.ca.y, s * r0.ca.x + c * r0.ca.y, r0.ca.z);
            (rot - r2.ca).norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_c3_err < 1e-10, "D3 C3(Z) error: {max_c3_err}");

    // Check C2(X) maps chain 0 (30°) -> chain 5 (-30°).
    let max_c2_err = bundle.chains[0]
        .residues
        .iter()
        .zip(bundle.chains[5].residues.iter())
        .map(|(r0, r5)| (Point3::new(r0.ca.x, -r0.ca.y, -r0.ca.z) - r5.ca).norm())
        .fold(0.0_f64, f64::max);
    assert!(max_c2_err < 1e-10, "D3 C2(X) error: {max_c2_err}");

    let min_d = bundle.min_interchain_ca_distance().unwrap();
    assert!(min_d > 4.5, "unexpected interchain clash: {min_d:.2} Å");

    for chain in &bundle.chains {
        let dihedrals = backbone_dihedrals(chain);
        for (i, res) in chain.residues.iter().enumerate() {
            assert!((bond_length(res.n, res.ca) - 1.457).abs() < 1e-6);
            assert!((bond_length(res.ca, res.c) - 1.528).abs() < 1e-6);
            assert!(improper_chirality_deg(res).is_some());
            if let Some(om) = dihedrals[i].2 {
                assert!(om.abs() > 179.9);
            }
        }
    }
}

#[test]
fn heteromeric_yaml_bundle_applies_relative_offsets_and_exports_pdb() {
    let yaml = r#"
symmetry: none
keep_gly: false
helices:
  - sequence: "AaAaAaAaAaAa"
    bundle_radius: 7.5
    pitch: 140.0
    phi_c_alpha_deg: 0.0
    z_offset: 0.0
    azimuth_deg: 0.0
    orientation: parallel
    twist_mode: coupled_darboux
    chain_id: "A"
  - sequence: "AaAaAaAaAaAa"
    bundle_radius: 8.5
    pitch: -180.0
    phi_c_alpha_deg: 90.0
    z_offset: 4.5
    azimuth_deg: 120.0
    orientation: antiparallel
    twist_mode: uncoupled_lab_frame
    chain_id: "B"
  - sequence: "aAaAaAaAaAaA"
    bundle_radius: 9.0
    phi_c_alpha_deg: 180.0
    z_offset: -2.0
    azimuth_deg: 240.0
    orientation: parallel
    chain_id: "C"
"#;
    let spec: BundleSpec = serde_yml::from_str(yaml).unwrap();
    let bundle = build_bundle(&spec).unwrap();
    assert_eq!(bundle.chains.len(), 3);

    let mean_z = |idx: usize| -> f64 {
        bundle.chains[idx]
            .residues
            .iter()
            .map(|r| r.ca.z)
            .sum::<f64>()
            / (bundle.chains[idx].residues.len() as f64)
    };
    assert!(mean_z(0).abs() < 0.05);
    assert!((mean_z(1) - 4.5).abs() < 0.05);
    assert!((mean_z(2) + 2.0).abs() < 1e-10);

    let pdb = bundle_to_pdb_string(&bundle);
    assert_eq!(pdb.lines().filter(|l| l.starts_with("ATOM  ")).count(), 180);
    assert_eq!(pdb.lines().filter(|l| l.starts_with("TER   ")).count(), 3);
    assert_eq!(pdb.lines().filter(|l| *l == "END").count(), 1);
}
