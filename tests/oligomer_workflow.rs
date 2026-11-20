//! Integration tests for oligomeric beta-helix design and multi-state DE screening.

use betahelixbuilder::{
    AnchorParity, BudeForceField, CanonicalResidue, DasprConfig, EdgePattern, EnergyOptions,
    Handedness, OligomerState, OligomerWorkflowConfig, PairSurrogateModel, ProbeConfig,
    SeamComposition, SeamPattern, SeamWidth, design_oligomers, probe_pair_interactions,
};

#[test]
#[ignore = "requires local dASPR binary outside sandbox"]
fn screens_parallel_seams_across_oligomer_states_with_de_convergence_flags() {
    let ff = BudeForceField::load_default().unwrap();
    let energy_opts = EnergyOptions::standard();
    let daspr_cfg = DasprConfig::default();

    let probe_cfg = ProbeConfig {
        anchor: AnchorParity::Both,
        residues: Some(vec![
            CanonicalResidue::Ala,
            CanonicalResidue::Leu,
            CanonicalResidue::Ile,
            CanonicalResidue::Trp,
            CanonicalResidue::Lys,
            CanonicalResidue::Glu,
            CanonicalResidue::Gln,
        ]),
        ..ProbeConfig::default()
    };
    let probe_rep = probe_pair_interactions(&probe_cfg, &ff, &energy_opts, &daspr_cfg).unwrap();
    let surrogate = PairSurrogateModel::from_probe_report(&probe_rep, Handedness::Right).unwrap();

    let cfg = OligomerWorkflowConfig {
        length: 30,
        seam_patterns: vec![SeamPattern::Strict6, SeamPattern::Staggered6676],
        seam_widths: vec![SeamWidth::Single],
        seam_compositions: vec![SeamComposition::MixedAliphaticAromatic],
        edge_patterns: vec![EdgePattern::ParallelBias],
        sa_restarts_per_class: 3,
        sa_steps: 250,
        top_monomers: 2,
        states: vec![
            OligomerState::parse("c2").unwrap(),
            OligomerState::parse("d1").unwrap(),
            OligomerState::parse("c3").unwrap(),
        ],
        run_coarse_screen: true,
        de_pop_size: 6,
        de_generations: 3,
        ..OligomerWorkflowConfig::default()
    };

    let rep = design_oligomers(&cfg, Some(&surrogate), &ff, &energy_opts, &daspr_cfg).unwrap();
    assert_eq!(rep.monomers.len(), 2);
    assert_eq!(rep.evaluations.len(), 6);
    for ev in &rep.evaluations {
        assert!(ev.e_per_chain.is_finite());
        assert!(ev.de_assoc_per_chain.is_finite());
        assert!(!ev.best_pdb.is_empty());
    }
    let mcsv = rep.monomer_csv();
    let ocsv = rep.oligomer_csv();
    assert!(mcsv.lines().count() == 3);
    assert!(ocsv.lines().count() == 7);
}
