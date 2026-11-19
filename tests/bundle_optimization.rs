//! Integration tests for Differential Evolution bundle optimisation.

use betahelixbuilder::{
    BudeForceField, DasprConfig, DeConfig, EnergyOptions, OptObjective, OptimizeSpec,
    ParamDescriptor, ScoredStructure, Symmetry, differential_evolution, evaluate_bundle,
    optimize_bundle,
};

#[test]
fn heteromeric_optimize_spec_decodes_and_boundary_warns() {
    let yaml = r#"
symmetry: none
objective: interaction
de:
  pop_size: 12
  generations: 12
  seed: 42
helices:
  - sequence: "VGAlAvVv"
    bundle_radius: { mean: 8.0, range: 2.0 }
    pitch: 140.0
    phi_c_alpha_deg: { mean: 0.0, range: 90.0 }
    z_offset: 0.0
    orientation: parallel
  - sequence: "VGAlAvVv"
    bundle_radius: 8.5
    pitch: 140.0
    phi_c_alpha_deg: 0.0
    z_offset: { mean: 0.0, range: 4.0 }
    orientation: antiparallel
"#;
    let spec: OptimizeSpec = serde_yml::from_str(yaml).unwrap();
    assert_eq!(spec.symmetry, Symmetry::None);
    assert_eq!(spec.objective, OptObjective::Interaction);

    let desc = spec.validate_and_descriptors().unwrap();
    assert_eq!(desc.len(), 3);
    assert_eq!(desc[0].label, "helix0.bundle_radius");
    assert_eq!(desc[1].label, "helix0.phi_c_alpha_deg");
    assert_eq!(desc[2].label, "helix1.z_offset");

    // Drive optimum to the upper edge (+1.0) to verify boundary convergence.
    let edge_desc = vec![ParamDescriptor {
        label: "edge".to_string(),
        mean: 0.0,
        half_range: 10.0,
    }];
    let edge_cfg = DeConfig {
        pop_size: 12,
        generations: 15,
        seed: 42,
        store_history: true,
        ..Default::default()
    };
    let res = differential_evolution(1, &edge_desc, &edge_cfg, |x| (1.0 - x[0]).powi(2)).unwrap();
    assert!(res.best_norm[0] > 0.95);
    assert_eq!(res.history.len(), 12 * (15 + 1));
}

#[test]
#[ignore = "requires external dASPR binary outside sandbox mount"]
fn optimize_d2_gramicidin_bundle_with_daspr_and_budeff() {
    let yaml = r#"
symmetry: d2
objective: internal
de:
  pop_size: 8
  generations: 3
  seed: 42
helices:
  - sequence: "VGAlAvVvWlWlWlW"
    bundle_radius: { mean: 9.0, range: 3.0 }
    pitch: { mean: 140.0, range: 40.0 }
    phi_c_alpha_deg: { mean: 0.0, range: 90.0 }
"#;
    let spec: OptimizeSpec = serde_yml::from_str(yaml).unwrap();
    let ff = BudeForceField::bude_2016v1().unwrap();
    let opts = EnergyOptions::standard();
    let daspr = DasprConfig::default();

    let result = optimize_bundle(&spec, &ff, &opts, &daspr).unwrap();
    assert!(result.best_pdb.contains("REMARK param bundle_radius"));
    assert!(result.best_energy.internal.total < result.logbook[0].avg);

    let rescored = evaluate_bundle(
        &ScoredStructure::from_pdb_str(&result.best_pdb).unwrap(),
        &ff,
        &opts,
    );
    assert!((rescored.internal.total - result.best_energy.internal.total).abs() < 0.5);
    assert!(!result.history_to_csv().is_empty());
}
