//! Integration tests for systematic pairwise beta-helix probing.

use betahelixbuilder::{
    AnchorParity, BudeForceField, CanonicalResidue, Chirality, DasprConfig, EnergyOptions,
    ProbeConfig, RankBy, probe_pair_interactions,
};

#[test]
#[ignore = "requires local dASPR binary outside sandbox"]
fn probe_subset_with_daspr_and_budeff_across_both_anchors() {
    let ff = BudeForceField::load_default().unwrap();
    let cfg = ProbeConfig {
        anchor: AnchorParity::Both,
        residues: Some(vec![
            CanonicalResidue::Ala,
            CanonicalResidue::Trp,
            CanonicalResidue::Asp,
            CanonicalResidue::Lys,
            CanonicalResidue::Pro,
        ]),
        ..ProbeConfig::default()
    };

    let report = probe_pair_interactions(
        &cfg,
        &ff,
        &EnergyOptions::standard(),
        &DasprConfig::default(),
    )
    .unwrap();

    // 2 anchors * 5 offsets * 5 * 5 pairs = 250 rows
    assert_eq!(report.rows.len(), 250);
    assert_eq!(report.parity_comparisons.len(), 5);

    // Proline must be present in both L ('P') and D ('p') positions.
    assert!(report.rows.iter().any(|r| r.res_i_case == 'P'));
    assert!(report.rows.iter().any(|r| r.res_i_case == 'p'));

    // Single-Ala reference pairs must satisfy ddE_DMC == 0 identically.
    for r in &report.rows {
        if r.res_i_code == 'A' || r.res_j_code == 'A' {
            assert!(
                r.dde_dmc.abs() < 1e-10,
                "expected ddE_dmc == 0 for ({}, {}), got {}",
                r.res_i_code,
                r.res_j_code,
                r.dde_dmc
            );
        }
        if r.res_i_code == 'A' && r.res_j_code == 'A' {
            assert!(r.de_vs_ala.abs() < 1e-10);
        }
    }

    // Opposite formal charge attraction (K-D at i+6) yields negative direct_charge.
    let kd_p6 = report
        .rows
        .iter()
        .find(|r| {
            r.anchor_chirality == Chirality::L
                && r.offset == 6
                && r.res_i_code == 'K'
                && r.res_j_code == 'D'
        })
        .expect("K-D at +6 row present");
    assert!(
        kd_p6.direct_charge < 0.0,
        "expected attractive K-D charge at +6, got {}",
        kd_p6.direct_charge
    );

    let csv = report.to_csv();
    assert!(csv.lines().count() == 251);
    let mat_dmc = report.matrix_csv(Chirality::L, 6, RankBy::Dmc).unwrap();
    assert!(mat_dmc.starts_with("res_i,A,W,D,K,P\n"));
}
