//! Integration tests for repetitive stability & handedness-specific sequence design.

use betahelixbuilder::{
    AnchorParity, BudeForceField, CanonicalResidue, DasprConfig, DesignConfig, EnergyOptions,
    PairSurrogateModel, ProbeConfig, design_sequences, probe_pair_interactions,
};

#[test]
#[ignore = "requires local dASPR binary outside sandbox"]
fn designs_capped_30mer_with_constraints_and_positive_3d_handedness_gap() {
    let ff = BudeForceField::load_default().unwrap();
    let energy_opts = EnergyOptions::standard();
    let daspr_cfg = DasprConfig::default();

    // Build a fast 6-residue subset surrogate table so the test executes quickly.
    let probe_cfg = ProbeConfig {
        anchor: AnchorParity::Both,
        residues: Some(vec![
            CanonicalResidue::Ala,
            CanonicalResidue::Leu,
            CanonicalResidue::Trp,
            CanonicalResidue::Arg,
            CanonicalResidue::Glu,
            CanonicalResidue::His,
        ]),
        ..ProbeConfig::default()
    };
    let probe_rep = probe_pair_interactions(&probe_cfg, &ff, &energy_opts, &daspr_cfg).unwrap();
    let surrogate =
        PairSurrogateModel::from_probe_report(&probe_rep, betahelixbuilder::Handedness::Right)
            .unwrap();

    let mut cfg = DesignConfig {
        sa_restarts: 8,
        sa_steps: 400,
        top_k_3d: 6,
        refine_termini: true,
        refine_top_n: 2,
        ..DesignConfig::default()
    };
    cfg.constraints.allow_terminal_pro = true;

    let report =
        design_sequences(&cfg, Some(&surrogate), &ff, &energy_opts, &daspr_cfg).unwrap();
    assert!(!report.candidates.is_empty());

    let top = &report.candidates[0];
    assert_eq!(top.sequence_core.len(), 30);
    assert!(top.sequence_with_caps.starts_with("Ac-"));
    assert!(top.sequence_with_caps.ends_with("-NH2"));
    assert!((1..=4).contains(&top.net_charge));
    assert!(top.trp_tyr_count >= 1);
    assert!(!top.sequence_core.contains(['C', 'c', 'M', 'm']));
    assert!(
        top.e_3d_target < report.host_3d_target_energy,
        "designed 30-mer should stabilize relative to poly-Ala host ({} vs {})",
        top.e_3d_target,
        report.host_3d_target_energy
    );
    assert!(
        top.e_3d_hand_gap > 5.0,
        "expected substantial positive handedness gap E_decoy - E_target, got {}",
        top.e_3d_hand_gap
    );
}
