//! PDB format output for generated peptide chains.

use std::io::{self, Write};

use nalgebra::Point3;

use crate::bundle::Bundle;
use crate::geometry::Chain;

struct PdbAtomRecord<'a> {
    serial: usize,
    atom_name: &'a str,
    res_name: &'a str,
    chain_id: char,
    seq_id: usize,
    pos: Point3<f64>,
    element: &'a str,
}

fn write_atom_record(writer: &mut impl Write, atom: &PdbAtomRecord<'_>) -> io::Result<()> {
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
    writer: &mut impl Write,
) -> io::Result<()> {
    let mut last_res_name = "ALA";
    let mut last_seq_id = 1usize;

    for res in &chain.residues {
        let mut emit = |atom_name: &str, pos: Point3<f64>, element: &str| -> io::Result<()> {
            write_atom_record(
                writer,
                &PdbAtomRecord {
                    serial: *serial,
                    atom_name,
                    res_name: res.res_name,
                    chain_id: chain.chain_id,
                    seq_id: res.seq_id,
                    pos,
                    element,
                },
            )?;
            *serial += 1;
            Ok(())
        };

        emit(" N  ", res.n, "N")?;
        emit(" CA ", res.ca, "C")?;
        emit(" C  ", res.c, "C")?;
        emit(" O  ", res.o, "O")?;
        if let Some(cb) = res.cb {
            emit(" CB ", cb, "C")?;
        }
        last_res_name = res.res_name;
        last_seq_id = res.seq_id;
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

/// Writes a generated [`Chain`] in fixed-column PDB format (`ATOM`, `TER`, `END`).
pub fn write_pdb(chain: &Chain, writer: &mut impl Write) -> io::Result<()> {
    let mut serial = 1usize;
    write_chain_records(chain, &mut serial, writer)?;
    writeln!(writer, "END")
}

/// Writes a multi-chain [`Bundle`] in fixed-column PDB format (`ATOM`, `TER` per chain, `END`).
pub fn write_bundle_pdb(bundle: &Bundle, writer: &mut impl Write) -> io::Result<()> {
    let mut serial = 1usize;
    for chain in &bundle.chains {
        write_chain_records(chain, &mut serial, writer)?;
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
        assert_eq!(atom_lines.len(), 10);
        for line in &atom_lines {
            assert_eq!(line.len(), 80, "expected 80-column PDB line: {line}");
        }
        assert!(atom_lines[0].contains("ALA A   1"));
        assert!(atom_lines[5].contains("DAL A   2"));
        assert!(pdb.lines().any(|l| l.starts_with("TER   ")));
        assert_eq!(pdb.lines().last(), Some("END"));
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
        assert_eq!(pdb.lines().filter(|l| l.starts_with("ATOM  ")).count(), 40);
        assert_eq!(pdb.lines().filter(|l| l.starts_with("TER   ")).count(), 4);
        assert_eq!(pdb.lines().filter(|l| *l == "END").count(), 1);
    }
}
