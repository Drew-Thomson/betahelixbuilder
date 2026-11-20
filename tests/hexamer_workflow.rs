use betahelixbuilder::{
    BudeForceField, DasprConfig, EnergyOptions, HexOligomerTarget, HexamerWorkflowConfig,
    LumenMode, PairSurrogateModel, ProbeConfig, ProbeReport, SeamPattern, design_hexamers,
    probe_pair_interactions,
};

#[test]
#[ignore = "requires local dASPR binary outside sandbox"]
fn screens_two_interface_c6_hexamers_with_ab_controls_and_pore_check() {
    let daspr = DasprConfig::default();
    let ff = BudeForceField::load_default().unwrap();
    let energy_opts = EnergyOptions::default();

    let pair_csv = std::path::Path::new("/tmp/beta_probe_test/pairs.csv");
    let surrogate = if pair_csv.is_file() {
        PairSurrogateModel::from_csv(pair_csv, betahelixbuilder::Handedness::Right).unwrap()
    } else {
        let probe_cfg = ProbeConfig {
            offsets: vec![1, 5, 6, 7],
            ..ProbeConfig::default()
        };
        let report: ProbeReport = probe_pair_interactions(&probe_cfg, &ff, &energy_opts, &daspr)
            .expect("probe pairs requires dASPR");
        PairSurrogateModel::from_probe_report(&report, betahelixbuilder::Handedness::Right).unwrap()
    };

    let states = ["c2_aa", "c2_bb", "c2", "c3", "c6", "d3"]
        .into_iter()
        .map(|s| HexOligomerTarget::parse(s).unwrap())
        .collect();

    let cfg = HexamerWorkflowConfig {
        seam_patterns: vec![SeamPattern::Strict6],
        lumen_modes: vec![LumenMode::Aromatic, LumenMode::SaltBridge],
        sa_restarts_per_class: 3,
        sa_steps: 600,
        coarse_pool_size: 4,
        top_de: 2,
        states,
        de_pop_size: 8,
        de_generations: 4,
        ..HexamerWorkflowConfig::default()
    };

    let report = design_hexamers(&cfg, Some(&surrogate), &ff, &energy_opts, &daspr).unwrap();
    assert_eq!(report.monomers.len(), 4);
    assert_eq!(report.coarse_evaluations.len(), 24);
    assert_eq!(report.de_evaluations.len(), 12);

    let promoted: Vec<_> = report
        .monomers
        .iter()
        .filter(|m| m.promoted_to_de)
        .collect();
    assert_eq!(promoted.len(), 2);
    for m in promoted {
        assert!(m.q_interface_a > 0 && m.q_interface_b < 0);
        assert!(m.target_pore_diameter_a.is_finite());
        assert!(m.best_homotypic_c2_assoc.is_finite());
    }
    assert!(report.coarse_csv().contains("coarse_de_assoc_per_chain"));
    assert!(report.hexamer_csv().contains("min_pore_diameter_a"));
}
