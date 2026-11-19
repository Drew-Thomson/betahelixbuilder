//! Integration tests verifying poly-D/L-alanine backbone reproduction of 1NRM and 2RQO.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use betahelixbuilder::{
    HelixParams, Sequence, backbone_dihedrals, bond_length, build_beta_helix,
    improper_chirality_deg, kabsch_rmsd,
};
use nalgebra::Point3;

type ResidueAtomsMap = BTreeMap<String, Point3<f64>>;

fn parse_cif_backbone(
    rel_path: &str,
    chain_id: &str,
    seq_start: i32,
    seq_end: i32,
) -> BTreeMap<i32, ResidueAtomsMap> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let content = fs::read_to_string(&path).expect("reference mmCIF file must exist");

    let mut cols = Vec::new();
    let mut in_atom_site = false;
    let mut residues: BTreeMap<i32, ResidueAtomsMap> = BTreeMap::new();

    for line in content.lines() {
        let s = line.trim();
        if let Some(col) = s.strip_prefix("_atom_site.") {
            cols.push(col.to_string());
            in_atom_site = true;
            continue;
        }
        if !in_atom_site {
            continue;
        }
        if s.starts_with('#') || s.starts_with("loop_") {
            if !residues.is_empty() {
                break;
            }
            continue;
        }
        if !(s.starts_with("ATOM") || s.starts_with("HETATM")) {
            continue;
        }

        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < cols.len() {
            continue;
        }

        let col_idx = |name: &str| cols.iter().position(|c| c == name);
        let model_num = col_idx("pdbx_PDB_model_num")
            .map(|i| parts[i])
            .unwrap_or("1");
        if model_num != "1" {
            continue;
        }

        let ch = parts[col_idx("auth_asym_id").expect("auth_asym_id column")];
        if ch != chain_id {
            continue;
        }

        let seq_id: i32 = parts[col_idx("auth_seq_id").expect("auth_seq_id column")]
            .parse()
            .expect("integer auth_seq_id");
        if seq_id < seq_start || seq_id > seq_end {
            continue;
        }

        let atom_name = parts[col_idx("label_atom_id").expect("label_atom_id column")];
        if !matches!(atom_name, "N" | "CA" | "C" | "O") {
            continue;
        }

        let x: f64 = parts[col_idx("Cartn_x").unwrap()].parse().unwrap();
        let y: f64 = parts[col_idx("Cartn_y").unwrap()].parse().unwrap();
        let z: f64 = parts[col_idx("Cartn_z").unwrap()].parse().unwrap();

        residues
            .entry(seq_id)
            .or_default()
            .insert(atom_name.to_string(), Point3::new(x, y, z));
    }

    residues
}

fn collect_coords(
    chain: &betahelixbuilder::Chain,
    reference: &BTreeMap<i32, ResidueAtomsMap>,
    atom_names: &[&str],
) -> (Vec<Point3<f64>>, Vec<Point3<f64>>) {
    assert_eq!(chain.residues.len(), reference.len());
    let mut mod_pts = Vec::new();
    let mut ref_pts = Vec::new();

    for (res_mod, (_, res_ref)) in chain.residues.iter().zip(reference.iter()) {
        for &atom in atom_names {
            let p_mod = match atom {
                "N" => res_mod.n,
                "CA" => res_mod.ca,
                "C" => res_mod.c,
                "O" => res_mod.o,
                _ => unreachable!(),
            };
            let p_ref = res_ref
                .get(atom)
                .unwrap_or_else(|| panic!("missing reference atom {atom}"));
            mod_pts.push(p_mod);
            ref_pts.push(*p_ref);
        }
    }
    (mod_pts, ref_pts)
}

#[test]
fn reproduces_1nrm_gramicidin_backbone_as_poly_dl_ala() {
    let ref_res = parse_cif_backbone("examples/1NRM.cif", "A", 1, 15);
    assert_eq!(ref_res.len(), 15);

    let seq = Sequence::parse("AaAaAaAaAaAaAaA").unwrap();
    let params = HelixParams::beta_6_3_gramicidin();
    let chain = build_beta_helix(&seq, &params, 'A').unwrap();

    let (mod_ca, ref_ca) = collect_coords(&chain, &ref_res, &["CA"]);
    let (mod_bb, ref_bb) = collect_coords(&chain, &ref_res, &["N", "CA", "C", "O"]);

    let ca_rmsd = kabsch_rmsd(&mod_ca, &ref_ca).unwrap();
    let bb_rmsd = kabsch_rmsd(&mod_bb, &ref_bb).unwrap();
    println!("1NRM (1..15): CA_RMSD={ca_rmsd:.3} A, BB_RMSD={bb_rmsd:.3} A");

    assert!(ca_rmsd < 0.8, "1NRM CA RMSD too high: {ca_rmsd:.3} Å");
    assert!(bb_rmsd < 0.8, "1NRM backbone RMSD too high: {bb_rmsd:.3} Å");
}

#[test]
fn reproduces_2rqo_polytheonamide_core_as_poly_dl_ala() {
    // Residues 2..=48 form the strictly alternating L/D beta-helix core of 2RQO.
    let ref_res = parse_cif_backbone("examples/2RQO.cif", "A", 2, 48);
    assert_eq!(ref_res.len(), 47);

    let seq_str: String = (0..47)
        .map(|i| if i % 2 == 0 { 'A' } else { 'a' })
        .collect();
    let seq = Sequence::parse(&seq_str).unwrap();
    let params = HelixParams::beta_6_3_polytheonamide();
    let chain = build_beta_helix(&seq, &params, 'A').unwrap();

    let (mod_ca, ref_ca) = collect_coords(&chain, &ref_res, &["CA"]);
    let (mod_bb, ref_bb) = collect_coords(&chain, &ref_res, &["N", "CA", "C", "O"]);

    let ca_rmsd = kabsch_rmsd(&mod_ca, &ref_ca).unwrap();
    let bb_rmsd = kabsch_rmsd(&mod_bb, &ref_bb).unwrap();
    println!("2RQO (2..48): CA_RMSD={ca_rmsd:.3} A, BB_RMSD={bb_rmsd:.3} A");

    assert!(ca_rmsd < 0.9, "2RQO (2..48) CA RMSD too high: {ca_rmsd:.3} Å");
    assert!(
        bb_rmsd < 0.9,
        "2RQO (2..48) backbone RMSD too high: {bb_rmsd:.3} Å"
    );
}

#[test]
fn reproduces_2rqo_full_48mer_as_poly_dl_ala() {
    // Full 48-mer (residues 1..=48, with pos 1 GLY replaced by L-Ala matching its negative phi).
    let ref_res = parse_cif_backbone("examples/2RQO.cif", "A", 1, 48);
    assert_eq!(ref_res.len(), 48);

    let seq = Sequence::parse("AAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaA").unwrap();
    let params = HelixParams::beta_6_3_polytheonamide();
    let chain = build_beta_helix(&seq, &params, 'A').unwrap();

    let (mod_ca, ref_ca) = collect_coords(&chain, &ref_res, &["CA"]);
    let (mod_bb, ref_bb) = collect_coords(&chain, &ref_res, &["N", "CA", "C", "O"]);

    let ca_rmsd = kabsch_rmsd(&mod_ca, &ref_ca).unwrap();
    let bb_rmsd = kabsch_rmsd(&mod_bb, &ref_bb).unwrap();
    println!("2RQO (1..48): CA_RMSD={ca_rmsd:.3} A, BB_RMSD={bb_rmsd:.3} A");

    assert!(ca_rmsd < 1.2, "2RQO (1..48) CA RMSD too high: {ca_rmsd:.3} Å");
    assert!(
        bb_rmsd < 1.3,
        "2RQO (1..48) backbone RMSD too high: {bb_rmsd:.3} Å"
    );
}

#[test]
fn supports_varying_residues_per_turn_profile() {
    let seq = Sequence::parse("AaAaAaAaAaAa").unwrap();
    let mut params = HelixParams::beta_6_3();
    params.residues_per_turn_profile = Some(vec![
        6.10, 6.15, 6.20, 6.25, 6.30, 6.35, 6.40, 6.35, 6.30, 6.25, 6.20,
    ]);
    let chain = build_beta_helix(&seq, &params, 'A').unwrap();
    let dihedrals = backbone_dihedrals(&chain);

    for i in 0..(chain.residues.len() - 1) {
        let c_n = bond_length(chain.residues[i].c, chain.residues[i + 1].n);
        assert!((c_n - 1.332).abs() < 1e-6, "peptide C-N distorted: {c_n}");
        if let Some(om) = dihedrals[i + 1].2 {
            assert!(om.abs() > 175.0, "non-trans omega: {om}");
        }
        assert!(improper_chirality_deg(&chain.residues[i]).is_some());
    }
}
