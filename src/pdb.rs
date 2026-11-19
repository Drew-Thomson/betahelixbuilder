//! PDB format output for generated peptide chains.

use std::io::{self, Write};

use nalgebra::Point3;

use crate::bundle::Bundle;
use crate::geometry::Chain;

pub(crate) struct PdbAtomRecord<'a> {
    pub(crate) serial: usize,
    pub(crate) atom_name: &'a str,
    pub(crate) res_name: &'a str,
    pub(crate) chain_id: char,
    pub(crate) seq_id: usize,
    pub(crate) pos: Point3<f64>,
    pub(crate) element: &'a str,
}

pub(crate) fn write_atom_record(
    writer: &mut (impl Write + ?Sized),
    atom: &PdbAtomRecord<'_>,
) -> io::Result<()> {
    writeln!(
        writer,
        "ATOM  {:>5} {:<4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}  ",
        atom.serial,
        atom.atom_name,
        atom.res_name,
        atom.chain_id,
        atom.seq_id,
        atom.pos.x,
        atom.pos.y,
        atom.pos.z,
        atom.element
    )
}

fn write_chain_records(
    chain: &Chain,
    serial: &mut usize,
    include_caps_and_oxt: bool,
    writer: &mut impl Write,
) -> io::Result<()> {
    let mut last_res_name = "ALA";
    let mut last_seq_id = 1usize;

    let emit_single = |w: &mut dyn Write,
                       serial: &mut usize,
                       atom_name: &str,
                       res_name: &str,
                       seq_id: usize,
                       pos: Point3<f64>,
                       element: &str|
     -> io::Result<()> {
        write_atom_record(
            w,
            &PdbAtomRecord {
                serial: *serial,
                atom_name,
                res_name,
                chain_id: chain.chain_id,
                seq_id,
                pos,
                element,
            },
        )?;
        *serial += 1;
        Ok(())
    };

    if include_caps_and_oxt && let Some(ace) = &chain.n_cap {
        emit_single(writer, serial, " CH3", "ACE", ace.seq_id, ace.ch3, "C")?;
        emit_single(writer, serial, " C  ", "ACE", ace.seq_id, ace.c, "C")?;
        emit_single(writer, serial, " O  ", "ACE", ace.seq_id, ace.o, "O")?;
    }

    for res in &chain.residues {
        emit_single(writer, serial, " N  ", res.res_name, res.seq_id, res.n, "N")?;
        emit_single(writer, serial, " CA ", res.res_name, res.seq_id, res.ca, "C")?;
        emit_single(writer, serial, " C  ", res.res_name, res.seq_id, res.c, "C")?;
        emit_single(writer, serial, " O  ", res.res_name, res.seq_id, res.o, "O")?;
        if let Some(cb) = res.cb {
            emit_single(writer, serial, " CB ", res.res_name, res.seq_id, cb, "C")?;
        }
        if include_caps_and_oxt && let Some(oxt) = res.oxt {
            emit_single(writer, serial, " OXT", res.res_name, res.seq_id, oxt, "O")?;
        }
        last_res_name = res.res_name;
        last_seq_id = res.seq_id;
    }

    if include_caps_and_oxt && let Some(nh2) = &chain.c_cap {
        emit_single(writer, serial, " N  ", "NH2", nh2.seq_id, nh2.n, "N")?;
        last_res_name = "NH2";
        last_seq_id = nh2.seq_id;
    }

    if !chain.residues.is_empty() {
        writeln!(
            writer,
            "TER   {:>5}      {:>3} {}{:>4}",
            *serial, last_res_name, chain.chain_id, last_seq_id
        )?;
        *serial += 1;
    }
    Ok(())
}

/// Formats a [`Chain`] without terminal caps or `OXT` (used as input for `dASPR`).
pub(crate) fn to_daspr_input_pdb(chain: &Chain) -> String {
    let mut buf = Vec::new();
    let mut serial = 1usize;
    write_chain_records(chain, &mut serial, false, &mut buf).expect("Vec<u8> write cannot fail");
    writeln!(buf, "END").expect("Vec<u8> write cannot fail");
    String::from_utf8(buf).expect("PDB output is valid ASCII/UTF-8")
}

/// Formats a [`Bundle`] without terminal caps or `OXT` (used as input for `dASPR`).
pub(crate) fn bundle_to_daspr_input_pdb(bundle: &Bundle) -> String {
    let mut buf = Vec::new();
    let mut serial = 1usize;
    for chain in &bundle.chains {
        write_chain_records(chain, &mut serial, false, &mut buf)
            .expect("Vec<u8> write cannot fail");
    }
    writeln!(buf, "END").expect("Vec<u8> write cannot fail");
    String::from_utf8(buf).expect("PDB output is valid ASCII/UTF-8")
}

/// Writes a generated [`Chain`] in fixed-column PDB format (`ATOM`, `TER`, `END`).
pub fn write_pdb(chain: &Chain, writer: &mut impl Write) -> io::Result<()> {
    let mut serial = 1usize;
    write_chain_records(chain, &mut serial, true, writer)?;
    writeln!(writer, "END")
}

/// Writes a multi-chain [`Bundle`] in fixed-column PDB format (`ATOM`, `TER` per chain, `END`).
pub fn write_bundle_pdb(bundle: &Bundle, writer: &mut impl Write) -> io::Result<()> {
    let mut serial = 1usize;
    for chain in &bundle.chains {
        write_chain_records(chain, &mut serial, true, writer)?;
    }
    writeln!(writer, "END")
}

/// Formats a [`Chain`] directly to a PDB-formatted `String`.
pub fn to_pdb_string(chain: &Chain) -> String {
    let mut buf = Vec::new();
    write_pdb(chain, &mut buf).expect("writing to Vec<u8> cannot fail");
    String::from_utf8(buf).expect("PDB output is valid ASCII/UTF-8")
}

/// Formats a [`Bundle`] directly to a PDB-formatted `String`.
pub fn bundle_to_pdb_string(bundle: &Bundle) -> String {
    let mut buf = Vec::new();
    write_bundle_pdb(bundle, &mut buf).expect("writing to Vec<u8> cannot fail");
    String::from_utf8(buf).expect("PDB output is valid ASCII/UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::build_beta_helix;
    use crate::input::{HelixParams, Sequence};

    #[test]
    fn pdb_output_has_valid_columns_and_roundtrips_coords() {
        let seq = Sequence::parse("Aa").unwrap();
        let chain = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();
        let pdb = to_pdb_string(&chain);
        let atom_lines: Vec<_> = pdb.lines().filter(|l| l.starts_with("ATOM  ")).collect();
        assert_eq!(atom_lines.len(), 11);
        for line in &atom_lines {
            assert_eq!(line.len(), 80, "expected 80-column PDB line: {line}");
        }
        assert!(atom_lines[0].contains("ALA A   1"));
        assert!(atom_lines[5].contains("DAL A   2"));
        assert!(atom_lines[10].contains(" OXT DAL A   2"));
        assert!(pdb.lines().any(|l| l.starts_with("TER   ")));
        assert_eq!(pdb.lines().last(), Some("END"));
    }

    #[test]
    fn pdb_output_supports_acetyl_and_amide_caps() {
        let seq = Sequence::parse("Ac-Aa-NH2").unwrap();
        let chain = build_beta_helix(&seq, &HelixParams::beta_6_3(), 'A').unwrap();
        let pdb = to_pdb_string(&chain);
        let atom_lines: Vec<_> = pdb.lines().filter(|l| l.starts_with("ATOM  ")).collect();
        // 3 (ACE) + 5 (ALA) + 5 (DAL) + 1 (NH2) = 14 atoms
        assert_eq!(atom_lines.len(), 14);
        assert!(atom_lines[0].contains(" CH3 ACE A   0"));
        assert!(atom_lines[3].contains("ALA A   1"));
        assert!(atom_lines[13].contains(" N   NH2 A   3"));
        assert!(!pdb.contains(" OXT "));
        assert!(pdb.lines().any(|l| l.contains("TER      15      NH2 A   3")));
    }

    #[test]
    fn bundle_pdb_writes_all_chains_with_ter_and_single_end() {
        use crate::bundle::build_bundle;
        use crate::input::{BundleSpec, HelixPlacementParams, Symmetry};

        let spec = BundleSpec {
            symmetry: Symmetry::Dn(2),
            keep_gly: false,
            pack: false,
            helices: vec![HelixPlacementParams {
                sequence: "Aa".to_string(),
                bundle_radius: 7.5,
                ..Default::default()
            }],
        };
        let bundle = build_bundle(&spec).unwrap();
        let pdb = bundle_to_pdb_string(&bundle);
        assert_eq!(pdb.lines().filter(|l| l.starts_with("ATOM  ")).count(), 44);
        assert_eq!(pdb.lines().filter(|l| l.starts_with("TER   ")).count(), 4);
        assert_eq!(pdb.lines().filter(|l| *l == "END").count(), 1);
    }
}
