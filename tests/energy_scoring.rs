//! Integration tests for native Rust BUDEFF energy evaluation on monomers and bundles.

use betahelixbuilder::{
    BudeForceField, BundleSpec, EnergyOptions, HelixPlacementParams, ScoredStructure, Sequence,
    Symmetry, build_beta_helix, build_bundle, bundle_to_pdb_string, evaluate_bundle,
    internal_energy, to_pdb_string, HelixParams,
};

const D_AMINO_TEST_PDB: &str = "\
ATOM      1  N   DAL A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  DAL A   1      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C   DAL A   1      12.000  10.000  10.000  1.00  0.00           C
ATOM      4  O   DAL A   1      13.000  10.000  10.000  1.00  0.00           O
ATOM      5  CB  DAL A   1      11.000  12.000  11.000  1.00  0.00           C
ATOM      6  N   DAS A   2      13.000  11.000  11.000  1.00  0.00           N
ATOM      7  CA  DAS A   2      14.000  12.000  12.000  1.00  0.00           C
ATOM      8  C   DAS A   2      15.000  11.000  11.000  1.00  0.00           C
ATOM      9  O   DAS A   2      16.000  11.000  11.000  1.00  0.00           O
ATOM     10  CB  DAS A   2      14.000  13.000  13.000  1.00  0.00           C
ATOM     11  CG  DAS A   2      15.000  14.000  14.000  1.00  0.00           C
ATOM     12  OD1 DAS A   2      16.000  14.000  14.000  1.00  0.00           O
ATOM     13  OD2 DAS A   2      14.000  15.000  15.000  1.00  0.00           O
";

#[test]
fn matches_budeff_reference_score_on_d_amino_test() {
    let ff = BudeForceField::bude_2016v1().unwrap();
    let structure = ScoredStructure::from_pdb_str(D_AMINO_TEST_PDB).unwrap();

    // Legacy electrostatic mode matches compiled budeff C++ extension 1:1.
    let legacy_opts = EnergyOptions {
        include_interactions: true,
        no_neighbour_backbone: true,
        legacy_unassigned_electrostatic_type: true,
    };
    let legacy = internal_energy(&structure, &ff, &legacy_opts);
    assert!(legacy.unparameterized_atoms.is_empty());
    assert!((legacy.total - 5.448321213720587).abs() < 1e-6);
    assert!((legacy.steric - 6.379563617959967).abs() < 1e-6);
    assert!((legacy.desolvation - (-0.9312424042393792)).abs() < 1e-6);
    assert!((legacy.charge - 0.0).abs() < 1e-6);

    // Fixed electrostatic mode produces valid finite scores and zero unparameterized atoms.
    let fixed = internal_energy(&structure, &ff, &EnergyOptions::standard());
    assert!(fixed.unparameterized_atoms.is_empty());
    assert!(fixed.total.is_finite());
}

#[test]
fn clashing_d2_bundle_has_much_higher_energy_than_spaced_bundle() {
    let ff = BudeForceField::bude_2016v1().unwrap();
    let opts = EnergyOptions::standard();

    let build_with_radius = |bundle_radius: f64| {
        let spec = BundleSpec {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: "AaAaAaAaAaAaAaA".to_string(),
                bundle_radius,
                pitch: Some(140.0),
                ..Default::default()
            }],
        };
        let bundle = build_bundle(&spec).unwrap();
        let report = evaluate_bundle(&ScoredStructure::from_bundle(&bundle), &ff, &opts);
        (bundle, report)
    };

    let (_clash_b, clash_e) = build_with_radius(5.0);
    let (spaced_b, spaced_e) = build_with_radius(8.0);

    assert!(
        clash_e.interaction.steric > 100.0,
        "expected strong steric clash at R=5 Å, got {}",
        clash_e.interaction.steric
    );
    assert!(
        spaced_e.interaction.total < clash_e.interaction.total,
        "spaced bundle ({}) should be lower energy than clashing bundle ({})",
        spaced_e.interaction.total,
        clash_e.interaction.total
    );

    // PDB round-trip gives matching energy within %8.3f coordinate precision.
    let pdb_str = bundle_to_pdb_string(&spaced_b);
    let parsed = ScoredStructure::from_pdb_str(&pdb_str).unwrap();
    let parsed_e = evaluate_bundle(&parsed, &ff, &opts);
    assert!((parsed_e.internal.total - spaced_e.internal.total).abs() < 0.5);
    assert!((parsed_e.interaction.total - spaced_e.interaction.total).abs() < 0.5);
}

#[test]
fn monomer_pdb_roundtrip_energy_matches_in_memory() {
    let ff = BudeForceField::bude_2016v1().unwrap();
    let opts = EnergyOptions::standard();
    let seq = Sequence::parse("AaAaAaAaAaAaAaA").unwrap();
    let chain = build_beta_helix(&seq, &HelixParams::beta_6_3_gramicidin(), 'A').unwrap();

    let mem_score = internal_energy(&ScoredStructure::from_chain(&chain), &ff, &opts);
    let pdb_score = internal_energy(
        &ScoredStructure::from_pdb_str(&to_pdb_string(&chain)).unwrap(),
        &ff,
        &opts,
    );
    assert!((mem_score.total - pdb_score.total).abs() < 0.2);
}
